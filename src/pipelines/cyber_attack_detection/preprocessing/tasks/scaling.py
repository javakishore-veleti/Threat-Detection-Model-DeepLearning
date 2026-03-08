"""Config-driven feature scaling using scikit-learn StandardScaler.

Scales only the TRUE NUMERIC columns — columns where arithmetic distance
is meaningful.  Label-encoded categorical columns and binary engineered
features are left untouched.

For the BETH dataset:

  Columns that GET scaled (3):
    timestamp    — seconds since epoch start. Range spans thousands; without
                   scaling it would dominate the autoencoder's loss function.
    argsNum      — count of syscall arguments (0–20+). Meaningful quantity.
    returnValue  — syscall return code. Spans a huge range (−4095 to +millions).
                   Negative = error, 0 = success, positive = info/fd number.

  Columns that do NOT get scaled:
    processId, threadId, parentProcessId, userId, mountNamespace, eventId
      → now label-encoded integers (0, 1, 2, ...). Scaling would distort
        the categorical encoding — encoded value 5 becoming −0.3 implies a
        relationship between categories that doesn't exist.
    processName, hostName, eventName
      → also label-encoded integers. Same reason.
    is_root, return_negative, is_child_of_init, is_orphan, is_high_args,
    args_touches_proc, args_touches_etc, args_is_hidden_path,
    args_has_write_flag, args_has_pathname
      → binary 0/1 features. Already in a small range; scaling would
        destroy the interpretable 0/1 boundary.
    return_category
      → ordinal 0/1/2 bucket. Small range, no scaling needed.

  Why StandardScaler?
    StandardScaler transforms each column to mean=0, std=1.  This ensures
    the autoencoder treats all numeric columns equally — without it,
    timestamp (range ~0–86400) would dominate returnValue (often 0).

  Data leakage prevention:
    The scaler is fit on TRAINING data only, then applied (transform) to
    val/test.  This prevents test-set statistics from leaking into the
    preprocessing pipeline.

  Final output:
    All DataFrames are converted to numpy float32 arrays and saved as .npy
    files.  The scaler is saved as a pickle artifact for inference reuse.
    input_dim (number of features) is recorded in ctx_data so the
    autoencoder knows its input size.

Reads from ctx_data: train_df, val_df, test_df,
                     train_labels, val_labels, test_labels,
                     true_numeric_columns
Writes to ctx_data:  train_X, val_X, test_X (numpy float32 arrays),
                     train_labels, val_labels, test_labels (numpy arrays),
                     input_dim, scaler, scaler_artifact_path,
                     scaled_columns, feature_names
"""

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from core.common.wfs.dtos import WfReq, WfResp
from core.common.wfs.interfaces import WfTask
from core.config import get_cfg
from core.logger import get_logger

log = get_logger(__name__)


class Scaling(WfTask):

    def execute(self, req: WfReq, resp: WfResp) -> WfResp:
        cfg = req.config
        df_keys = req.df_keys        # ("train_df", "val_df", "test_df")
        label_keys = req.label_keys  # ("train_labels", "val_labels", "test_labels")
        train_key = df_keys[0]       # fit scaler on training only

        # ---------- Determine which columns to scale ---------------------------
        # For BETH: ["timestamp", "argsNum", "returnValue"] — the only 3
        # columns where arithmetic relationships are meaningful.
        scale_cols = resp.ctx_data.get(
            "true_numeric_columns",
            get_cfg(cfg, "columns.true_numeric.names", []),
        )

        # Where to save scaler.pkl and .npy arrays
        artifact_dir = Path(get_cfg(
            cfg, "dataset.paths.artifact_dir",
            str(Path(get_cfg(cfg, "dataset.paths.data_dir", ".")) / "artifacts"),
        ))

        # ---------- Retrieve DataFrames from ctx_data --------------------------
        frames = {k: resp.ctx_data.get(k) for k in df_keys}
        label_frames = {k: resp.ctx_data.get(k) for k in label_keys}

        if any(v is None for v in frames.values()):
            resp.success = False
            resp.message = f"scaling requires {df_keys} in ctx_data"
            return resp

        train_df: pd.DataFrame = frames[train_key]

        # Filter to columns that actually exist in the DataFrame
        # (in case a true-numeric column was dropped earlier)
        scale_cols = [c for c in scale_cols if c in train_df.columns]
        all_columns = list(train_df.columns)

        if not scale_cols:
            log.debug("No columns to scale — converting directly to numpy")
        else:
            log.debug("Scaling %d columns: %s", len(scale_cols), scale_cols)

        # ---------- Fit StandardScaler on TRAINING data only -------------------
        # For BETH: computes mean and std of timestamp, argsNum, returnValue
        # from the 763K training rows.  These same statistics are used to
        # transform val and test — no data leakage.
        scaler = StandardScaler()
        if scale_cols:
            scaler.fit(train_df[scale_cols])
            log.debug("Scaler fit on training — means: %s, stds: %s",
                      dict(zip(scale_cols, scaler.mean_.round(4))),
                      dict(zip(scale_cols, scaler.scale_.round(4))))

        # ---------- Transform each split and convert to numpy float32 ----------
        arrays: dict[str, np.ndarray] = {}
        label_arrays: dict[str, np.ndarray] = {}

        for df_key, lbl_key in zip(df_keys, label_keys):
            df = frames[df_key].copy()

            # Scale only the true numeric columns, leave everything else intact
            if scale_cols:
                df[scale_cols] = scaler.transform(df[scale_cols])

            # Convert entire DataFrame to float32 numpy array for PyTorch
            arr = df.values.astype(np.float32)
            arrays[df_key] = arr

            # Convert labels to numpy as well (evil + sus columns)
            lbl_df = label_frames.get(lbl_key)
            if lbl_df is not None:
                label_arrays[lbl_key] = lbl_df.values.astype(np.float32)

            log.debug("%s: scaled %d cols → numpy float32 shape %s",
                      df_key, len(scale_cols), arr.shape)

        # ---------- Save artifacts to disk -------------------------------------
        artifact_dir.mkdir(parents=True, exist_ok=True)

        # Save the fitted scaler for inference reuse
        scaler_path = artifact_dir / "scaler.pkl"
        with open(scaler_path, "wb") as f:
            pickle.dump({"scaler": scaler, "columns": scale_cols}, f)
        log.debug("Saved scaler to %s", scaler_path)

        # Save numpy arrays as .npy files
        # For BETH: train_X.npy (~763K × 20), val_X.npy (~76K × 20),
        # test_X.npy (~717K × 20), plus corresponding label arrays.
        prefix_map = dict(zip(df_keys, ["train", "val", "test"]))
        label_prefix_map = dict(zip(label_keys, ["train", "val", "test"]))

        for df_key, prefix in prefix_map.items():
            npy_path = artifact_dir / f"{prefix}_X.npy"
            np.save(npy_path, arrays[df_key])
            log.debug("Saved %s → %s", df_key, npy_path)

        for lbl_key, prefix in label_prefix_map.items():
            if lbl_key in label_arrays:
                npy_path = artifact_dir / f"{prefix}_labels.npy"
                np.save(npy_path, label_arrays[lbl_key])
                log.debug("Saved %s → %s", lbl_key, npy_path)

        # ---------- Publish to ctx_data for downstream tasks -------------------
        # The autoencoder and trainer tasks will read these numpy arrays.
        input_dim = arrays[train_key].shape[1]

        for df_key in df_keys:
            # Replace DataFrame with numpy array under a new key
            x_key = df_key.replace("_df", "_X")  # train_df → train_X
            resp.ctx_data[x_key] = arrays[df_key]

        for lbl_key in label_keys:
            if lbl_key in label_arrays:
                resp.ctx_data[lbl_key] = label_arrays[lbl_key]

        resp.ctx_data["input_dim"] = input_dim
        resp.ctx_data["scaler"] = scaler
        resp.ctx_data["scaler_artifact_path"] = str(scaler_path)
        resp.ctx_data["scaled_columns"] = scale_cols
        resp.ctx_data["feature_names"] = all_columns

        total_rows = sum(arr.shape[0] for arr in arrays.values())
        resp.message = (
            f"Scaling complete — {total_rows:,} rows, {input_dim} features "
            f"({len(scale_cols)} scaled, {input_dim - len(scale_cols)} untouched), "
            f"numpy float32 arrays + artifacts at {artifact_dir}"
        )
        log.debug(resp.message)
        return resp


Task = Scaling
