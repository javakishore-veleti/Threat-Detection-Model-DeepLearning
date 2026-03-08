"""Config-driven categorical encoding using scikit-learn LabelEncoder.

All column names come from ctx_data (published by data_analysis) or
from req.config.  No dataset-specific values are hardcoded here.

For the BETH dataset this task label-encodes all 9 categorical columns:

  String columns (3):
    processName  — program name ("systemd", "sshd", "bash", etc.)
    hostName     — server identifier
    eventName    — human-readable syscall name ("prctl", "open", etc.)

  Integer ID columns (6) — LOOK numeric but ARE categorical:
    processId         — OS-assigned PID (PID 500 is not "twice" PID 250)
    threadId          — OS-assigned TID
    parentProcessId   — parent PID (1 = init, 0 = orphan/kernel)
    userId            — Linux UID (0 = root, not "less than" UID 100)
    mountNamespace    — kernel namespace ID (opaque container boundary)
    eventId           — syscall type code (157 = prctl, 2 = open — an enum)

  Why label-encode the 6 integer IDs?
    They are stored as int64 so pandas treats them as numeric, but they are
    identifiers with NO ordinal relationship.  If we leave processId as raw
    integers, the model learns "PID 30000 is 30x more important than PID
    1000" — which is wrong.  Label encoding converts each unique value to a
    sequential integer (0, 1, 2, ...) that says "these are different
    categories" without implying any ordering.

  Handling unseen values in val/test:
    The test set contains attack traffic with process IDs, event types, and
    process names never seen during normal training.  These unseen values are
    mapped to a special UNKNOWN token (integer 0) so the encoder doesn't
    crash and the model sees a consistent "never seen before" signal.

  Artifact:
    The fitted LabelEncoders are saved as a pickle file so that the same
    encoding can be applied during inference on new data.

Reads from ctx_data: train_df, val_df, test_df,
                     categorical_id_columns, string_categorical_columns
Writes to ctx_data:  train_df, val_df, test_df (columns now integers),
                     encoders (dict of {column_name: LabelEncoder}),
                     encoder_artifact_path
"""

import pickle
from pathlib import Path

import pandas as pd
from sklearn.preprocessing import LabelEncoder

from core.common.wfs.dtos import WfReq, WfResp
from core.common.wfs.interfaces import WfTask
from core.config import get_cfg
from core.logger import get_logger

log = get_logger(__name__)


class Encoding(WfTask):

    def execute(self, req: WfReq, resp: WfResp) -> WfResp:
        cfg = req.config
        # ("train_df", "val_df", "test_df")
        df_keys = req.df_keys
        train_key = df_keys[0]  # thresholds / vocabularies come from training only

        # ---------- Determine which columns to encode --------------------------
        # For BETH: 6 integer IDs + 3 string categorical = 9 columns total.
        # These lists were published to ctx_data by data_analysis.
        cat_id_cols = resp.ctx_data.get("categorical_id_columns", [])
        string_cat_cols = resp.ctx_data.get("string_categorical_columns", [])
        all_cat_cols = string_cat_cols + cat_id_cols

        if not all_cat_cols:
            log.debug("No categorical columns to encode — skipping")
            resp.message = "Encoding skipped — no categorical columns found"
            return resp

        # The token used for unseen categories.  Must match what cleaning used
        # to fill NaN strings (cleaning.categorical_fill = "UNKNOWN" in YAML).
        unknown_token = get_cfg(cfg, "encoding.unknown_token",
                                get_cfg(cfg, "cleaning.categorical_fill", "UNKNOWN"))

        # Where to save the label_encoders.pkl artifact
        artifact_dir = Path(get_cfg(
            cfg, "dataset.paths.artifact_dir",
            str(Path(get_cfg(cfg, "dataset.paths.data_dir", ".")) / "artifacts"),
        ))

        # ---------- Retrieve DataFrames from ctx_data --------------------------
        frames = {k: resp.ctx_data.get(k) for k in df_keys}
        if any(v is None for v in frames.values()):
            resp.success = False
            resp.message = f"encoding requires {df_keys} in ctx_data"
            return resp

        train_df: pd.DataFrame = frames[train_key]
        encoders: dict[str, LabelEncoder] = {}
        total_unseen = 0

        for col in all_cat_cols:
            if col not in train_df.columns:
                log.debug("Skipping %s — column not in DataFrame", col)
                continue

            le = LabelEncoder()

            # Convert training values to strings so integer IDs and string
            # columns go through the same encoding path.
            # e.g. processId 1234 → "1234", processName "sshd" → "sshd"
            train_vals = train_df[col].astype(str).tolist()

            # Inject UNKNOWN into the training vocabulary so it gets a
            # dedicated integer code.  When val/test have unseen values
            # (e.g. a processName that only appears during attacks), they
            # get mapped to this UNKNOWN integer.
            if unknown_token not in train_vals:
                train_vals.append(unknown_token)

            # Fit on training vocabulary only — no data leakage from val/test
            le.fit(train_vals)
            known_classes = set(le.classes_)

            # Transform every split, replacing unseen values with UNKNOWN
            for key in df_keys:
                df = frames[key]
                col_str = df[col].astype(str)

                unseen_mask = ~col_str.isin(known_classes)
                n_unseen = int(unseen_mask.sum())
                if n_unseen > 0:
                    col_str = col_str.where(~unseen_mask, unknown_token)
                    total_unseen += n_unseen
                    log.debug("  %s/%s: %d unseen values → %s",
                              key, col, n_unseen, unknown_token)

                df[col] = le.transform(col_str)

            encoders[col] = le
            log.debug("Encoded %s: %d classes (incl. %s)",
                      col, len(le.classes_), unknown_token)

        # ---------- Save encoders to disk as pickle ----------------------------
        # Used later during inference to apply the same encoding to new data.
        artifact_dir.mkdir(parents=True, exist_ok=True)
        encoder_path = artifact_dir / "label_encoders.pkl"
        with open(encoder_path, "wb") as f:
            pickle.dump(encoders, f)
        log.debug("Saved %d encoders to %s", len(encoders), encoder_path)

        # ---------- Publish updated DataFrames back to ctx_data ----------------
        for key in df_keys:
            resp.ctx_data[key] = frames[key]
        resp.ctx_data["encoders"] = encoders
        resp.ctx_data["encoder_artifact_path"] = str(encoder_path)

        total_encoded = len(encoders)
        total_classes = sum(len(le.classes_) for le in encoders.values())
        resp.message = (
            f"Encoding complete — {total_encoded} columns label-encoded "
            f"({total_classes} total classes, {total_unseen} unseen replacements), "
            f"artifact at {encoder_path}"
        )
        log.debug(resp.message)
        return resp


Task = Encoding
