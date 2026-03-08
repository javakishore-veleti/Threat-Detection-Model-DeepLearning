"""PyTorch training loop for the autoencoder with validation and checkpointing.

All hyperparameters (learning rate, batch size, epochs) come from
req.config["training"] so they can be tuned from YAML.

For the BETH dataset — what happens during training:
-----------------------------------------------------
The autoencoder is trained to reconstruct NORMAL syscall events.
Training data has 763,144 rows × 23 features, all with evil=0 (no attacks).

Each epoch:
  1. TRAINING PASS — shuffle the 763K normal rows into batches of 256.
     For each batch:
       a) Feed 256 rows through the autoencoder → get 256 reconstructed rows
       b) Compute MSE loss: how different is the reconstruction from the input?
       c) Backpropagate: compute gradients (which weights caused the error?)
       d) Update weights with Adam optimizer (move them to reduce the error)
     After all batches: average training loss for this epoch.

  2. VALIDATION PASS — feed the 188,967 validation rows through (NO gradient
     computation, just measure how well the model reconstructs unseen normal data).
     The validation set also has evil=0 (all normal) but 1,269 rows are sus=1
     (suspicious).  If the model learns meaningful patterns, sus=1 rows should
     have slightly higher reconstruction error than sus=0 rows.

  3. CHECKPOINT — save the model weights:
       last.pt  — always overwritten (so we can resume if training is interrupted)
       best.pt  — overwritten ONLY when val_loss improves (this is the model
                  we'll use for inference — it generalises best to unseen data)

  Over 50 epochs the losses should decrease: the model gets better at
  reconstructing normal patterns.  When we later feed ATTACK data through
  the trained model, reconstruction error will be high → anomaly detected.

Resume support:
  If the pipeline was interrupted mid-training (machine crashed, timeout),
  passing --resume <path-to-last.pt> reloads the model weights, optimizer
  state, and epoch counter, and continues training from where it left off.

Reads from ctx_data: model, model_device, train_X, val_X
Writes to ctx_data:  model (with trained weights), checkpoint_dir,
                     train_losses, val_losses, best_val_loss
"""

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from core.common.wfs.dtos import WfReq, WfResp
from core.common.wfs.interfaces import WfTask
from core.config import get_cfg
from core.logger import get_logger

log = get_logger(__name__)


class Trainer(WfTask):

    def execute(self, req: WfReq, resp: WfResp) -> WfResp:
        cfg = req.config

        # ---------- Retrieve model and data from ctx_data ----------------------
        model: nn.Module | None = resp.ctx_data.get("model")
        if model is None:
            resp.success = False
            resp.message = "trainer requires model in ctx_data — run autoencoder first"
            return resp

        device_str = resp.ctx_data.get("model_device", "cpu")
        device = torch.device(device_str)

        # For BETH: train_X shape = (763144, 23), val_X shape = (188967, 23)
        train_X: np.ndarray | None = resp.ctx_data.get("train_X")
        val_X: np.ndarray | None = resp.ctx_data.get("val_X")

        if train_X is None or val_X is None:
            resp.success = False
            resp.message = "trainer requires train_X and val_X in ctx_data — run scaling first"
            return resp

        # ---------- Read training hyperparameters from YAML --------------------
        lr = get_cfg(cfg, "training.lr", 0.001)
        batch_size = get_cfg(cfg, "training.batch_size", 256)
        num_epochs = get_cfg(cfg, "training.num_epochs", 50)

        # Where to save checkpoints (best.pt, last.pt)
        checkpoint_dir = Path(get_cfg(
            cfg, "training.checkpoint_dir",
            str(Path(get_cfg(cfg, "dataset.paths.artifact_dir", ".")) / "checkpoints"),
        ))
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # ---------- Create PyTorch DataLoaders ---------------------------------
        # Convert numpy arrays to PyTorch tensors and wrap in DataLoader.
        # TensorDataset: pairs each row with itself (autoencoder target = input).
        # shuffle=True for training so each epoch sees rows in a different order
        # (helps the model generalise, avoids learning batch-specific patterns).
        train_tensor = torch.from_numpy(train_X).to(device)
        val_tensor = torch.from_numpy(val_X).to(device)

        train_dataset = TensorDataset(train_tensor, train_tensor)
        val_dataset = TensorDataset(val_tensor, val_tensor)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        log.debug("DataLoaders: train=%d batches, val=%d batches (batch_size=%d)",
                  len(train_loader), len(val_loader), batch_size)

        # ---------- Loss function and optimizer --------------------------------
        # MSELoss: Mean Squared Error between input and reconstruction.
        # For each sample: average of (input_feature_i - output_feature_i)^2
        # across all 23 features.  Lower = better reconstruction.
        criterion = nn.MSELoss()

        # Adam: adaptive learning rate optimizer.  Adjusts the step size for
        # each weight individually based on recent gradient history.
        # lr=0.001 is a standard starting point.
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        # ---------- Resume from checkpoint if requested ------------------------
        start_epoch = 0
        best_val_loss = float("inf")
        best_path = checkpoint_dir / "best.pt"
        last_path = checkpoint_dir / "last.pt"

        if req.resume and Path(req.resume).exists():
            checkpoint = torch.load(req.resume, map_location=device, weights_only=False)
            model.load_state_dict(checkpoint["model_state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            start_epoch = checkpoint.get("epoch", 0) + 1
            best_val_loss = checkpoint.get("best_val_loss", float("inf"))
            log.debug("Resumed from %s — starting at epoch %d (best_val_loss=%.6f)",
                      req.resume, start_epoch, best_val_loss)

        # ---------- Training loop ----------------------------------------------
        train_losses: list[float] = []
        val_losses: list[float] = []

        log.debug("Starting training: %d epochs, lr=%s, device=%s", num_epochs, lr, device)

        for epoch in range(start_epoch, num_epochs):

            # --- Training pass ---
            # model.train() enables training-mode behaviors (e.g. dropout,
            # batch norm — our autoencoder doesn't use these, but it's best
            # practice to always call it).
            model.train()
            epoch_train_loss = 0.0

            for batch_input, batch_target in train_loader:
                # Forward pass: feed batch through autoencoder
                reconstruction = model(batch_input)
                loss = criterion(reconstruction, batch_target)

                # Backward pass: compute gradients
                optimizer.zero_grad()  # clear old gradients from last batch
                loss.backward()        # compute new gradients
                optimizer.step()       # update weights

                epoch_train_loss += loss.item() * batch_input.size(0)

            avg_train_loss = epoch_train_loss / len(train_dataset)
            train_losses.append(avg_train_loss)

            # --- Validation pass ---
            # model.eval() disables training-mode behaviors.
            # torch.no_grad() tells PyTorch not to track gradients — saves
            # memory and computation since we're only measuring, not learning.
            model.eval()
            epoch_val_loss = 0.0

            with torch.no_grad():
                for batch_input, batch_target in val_loader:
                    reconstruction = model(batch_input)
                    loss = criterion(reconstruction, batch_target)
                    epoch_val_loss += loss.item() * batch_input.size(0)

            avg_val_loss = epoch_val_loss / len(val_dataset)
            val_losses.append(avg_val_loss)

            # --- Log progress ---
            improved = avg_val_loss < best_val_loss
            marker = " ★ new best" if improved else ""
            log.debug("Epoch %3d/%d — train_loss: %.6f  val_loss: %.6f%s",
                      epoch + 1, num_epochs, avg_train_loss, avg_val_loss, marker)

            # --- Checkpoint: always save last.pt for resume ---
            _save_checkpoint(last_path, model, optimizer, epoch,
                             avg_train_loss, avg_val_loss, best_val_loss)

            # --- Checkpoint: save best.pt only when val_loss improves ---
            if improved:
                best_val_loss = avg_val_loss
                _save_checkpoint(best_path, model, optimizer, epoch,
                                 avg_train_loss, avg_val_loss, best_val_loss)

        # ---------- Load best weights for downstream tasks ---------------------
        # The model currently has the LAST epoch's weights.  Load the BEST
        # weights so the evaluation/inference tasks use the model that
        # generalised best to the validation set.
        if best_path.exists():
            best_ckpt = torch.load(best_path, map_location=device, weights_only=False)
            model.load_state_dict(best_ckpt["model_state_dict"])
            log.debug("Loaded best model from epoch %d (val_loss=%.6f)",
                      best_ckpt["epoch"] + 1, best_ckpt["best_val_loss"])

        # ---------- Publish to ctx_data for evaluation tasks -------------------
        resp.ctx_data["model"] = model
        resp.ctx_data["checkpoint_dir"] = str(checkpoint_dir)
        resp.ctx_data["train_losses"] = train_losses
        resp.ctx_data["val_losses"] = val_losses
        resp.ctx_data["best_val_loss"] = best_val_loss

        resp.message = (
            f"Training complete — {num_epochs} epochs, "
            f"best val_loss={best_val_loss:.6f}, "
            f"checkpoints at {checkpoint_dir}"
        )
        log.debug(resp.message)
        return resp


def _save_checkpoint(path: Path, model: nn.Module, optimizer: torch.optim.Optimizer,
                     epoch: int, train_loss: float, val_loss: float,
                     best_val_loss: float) -> None:
    """Save a training checkpoint that contains everything needed to resume."""
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "train_loss": train_loss,
        "val_loss": val_loss,
        "best_val_loss": best_val_loss,
    }, path)


Task = Trainer
