"""Config-driven PyTorch autoencoder for anomaly detection.

This task creates the neural network model that will learn to reconstruct
normal host-level event patterns.  The architecture and hyperparameters are
read from req.config["model"] so they can be tuned from YAML without
touching Python code.

For the BETH dataset — how the autoencoder works:
--------------------------------------------------
An autoencoder is a neural network that learns to COPY its input to its
output through a narrow bottleneck.  During training it only sees NORMAL
syscall events (763K rows, 23 features).  It gets very good at
reconstructing normal patterns.

When an ATTACK event is fed through the trained autoencoder, the output
won't match the input well because the model has never seen attack
patterns.  The difference (reconstruction error) is high → anomaly.

Architecture for BETH (default config):

  INPUT (23 features)
    ↓
  Linear(23 → 64) + ReLU      ← encoder: compress
    ↓
  Linear(64 → 32) + ReLU      ← encoder: compress more
    ↓
  Linear(32 → 16) + ReLU      ← BOTTLENECK: tightest compression
    ↓                            (the model must distill 23 features
    ↓                             into just 16 numbers — forces it
    ↓                             to learn only the most important
    ↓                             patterns of normal behavior)
    ↓
  Linear(16 → 32) + ReLU      ← decoder: expand
    ↓
  Linear(32 → 64) + ReLU      ← decoder: expand more
    ↓
  Linear(64 → 23)             ← output: reconstruct original 23 features
                                 (NO activation — raw values, because
                                  our features include negative numbers
                                  from StandardScaler)

  Loss = MSE (Mean Squared Error) between input and output.
  High MSE on a sample → the model couldn't reconstruct it → anomaly.

Why these layer sizes?
  23 → 64: expand first to give the network room to learn combinations
  64 → 32 → 16: progressively compress — each layer must keep only the
                 most important patterns, dropping noise
  16 → 32 → 64 → 23: mirror the encoder to reconstruct
  The bottleneck (16) is the key — too wide and the model memorizes
  everything including noise; too narrow and it can't represent normal
  patterns well.  16 is a good starting point for 23 input features.

Reads from ctx_data: input_dim (23 for BETH)
Writes to ctx_data:  model (Autoencoder nn.Module instance),
                     model_device ("cpu" or "cuda")
"""

import torch
import torch.nn as nn

from core.common.wfs.dtos import WfReq, WfResp
from core.common.wfs.interfaces import WfTask
from core.config import get_cfg
from core.logger import get_logger

log = get_logger(__name__)


class Autoencoder(nn.Module):
    """Symmetric autoencoder: encoder → bottleneck → decoder.

    For BETH default config:
      encoder:  23 → 64 → 32 → 16  (each with ReLU)
      decoder:  16 → 32 → 64 → 23  (ReLU on hidden, NO activation on output)
    """

    def __init__(self, input_dim: int, hidden_dims: list[int],
                 bottleneck_dim: int):
        super().__init__()

        # --- Build encoder: input → hidden_dims → bottleneck ---
        # For BETH: [Linear(23,64)+ReLU, Linear(64,32)+ReLU, Linear(32,16)+ReLU]
        encoder_layers: list[nn.Module] = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            encoder_layers.append(nn.Linear(prev_dim, h_dim))
            encoder_layers.append(nn.ReLU())
            prev_dim = h_dim
        encoder_layers.append(nn.Linear(prev_dim, bottleneck_dim))
        encoder_layers.append(nn.ReLU())
        self.encoder = nn.Sequential(*encoder_layers)

        # --- Build decoder: bottleneck → reversed hidden_dims → input ---
        # For BETH: [Linear(16,32)+ReLU, Linear(32,64)+ReLU, Linear(64,23)]
        # The final layer has NO activation so the model can output any value
        # (important because StandardScaler produces negative values).
        decoder_layers: list[nn.Module] = []
        prev_dim = bottleneck_dim
        for h_dim in reversed(hidden_dims):
            decoder_layers.append(nn.Linear(prev_dim, h_dim))
            decoder_layers.append(nn.ReLU())
            prev_dim = h_dim
        decoder_layers.append(nn.Linear(prev_dim, input_dim))
        self.decoder = nn.Sequential(*decoder_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Pass input through encoder then decoder to reconstruct it.

        For BETH: x has shape (batch_size, 23).
        Returns: reconstructed x with same shape (batch_size, 23).
        The training loss is MSE between input x and this output.
        """
        encoded = self.encoder(x)   # (batch, 23) → (batch, 16)
        decoded = self.decoder(encoded)  # (batch, 16) → (batch, 23)
        return decoded


class AutoencoderTask(WfTask):

    def execute(self, req: WfReq, resp: WfResp) -> WfResp:
        cfg = req.config

        # ---------- Read input_dim from ctx_data (set by scaling task) ---------
        # For BETH: 23 features after all preprocessing
        input_dim = resp.ctx_data.get("input_dim")
        if input_dim is None:
            resp.success = False
            resp.message = "autoencoder requires input_dim in ctx_data — run scaling first"
            return resp

        # ---------- Read architecture from YAML config -------------------------
        # For BETH defaults: hidden_dims=[64, 32], bottleneck=16
        hidden_dims = get_cfg(cfg, "model.hidden_dims", [64, 32])
        bottleneck_dim = get_cfg(cfg, "model.bottleneck_dim", 16)

        # Device selection: prefer CUDA (GPU) if available, fall back to MPS
        # (Apple Silicon) or CPU.
        device_cfg = get_cfg(cfg, "model.device", "auto")
        if device_cfg == "auto":
            if torch.cuda.is_available():
                device = torch.device("cuda")
            elif torch.backends.mps.is_available():
                device = torch.device("mps")
            else:
                device = torch.device("cpu")
        else:
            device = torch.device(device_cfg)

        # ---------- Create the autoencoder model -------------------------------
        # For BETH: Autoencoder(23, [64,32], 16)
        #   encoder: 23→64→32→16 (with ReLU between each)
        #   decoder: 16→32→64→23 (ReLU on hidden, no activation on output)
        model = Autoencoder(input_dim, hidden_dims, bottleneck_dim)
        model = model.to(device)

        # Count trainable parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        log.debug("Autoencoder created on %s:", device)
        log.debug("  Architecture: %d → %s → %d → %s → %d",
                  input_dim, hidden_dims, bottleneck_dim,
                  list(reversed(hidden_dims)), input_dim)
        log.debug("  Total parameters: %d (all trainable)", total_params)
        log.debug("  Model:\n%s", model)

        # ---------- Publish to ctx_data for training task ----------------------
        resp.ctx_data["model"] = model
        resp.ctx_data["model_device"] = str(device)

        resp.message = (
            f"Autoencoder ready — {input_dim}→{hidden_dims}→{bottleneck_dim}"
            f"→{list(reversed(hidden_dims))}→{input_dim} "
            f"({total_params:,} params) on {device}"
        )
        log.debug(resp.message)
        return resp


Task = AutoencoderTask
