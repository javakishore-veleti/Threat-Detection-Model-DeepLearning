"""Clean raw DataFrames: drop unusable columns, handle missing values, separate labels.

All column names, split info, and label columns are read from ctx_data
(populated by the data_analysis task).  DataFrames are reused from
ctx_data["raw_frames"] to avoid re-reading CSVs from disk.

For the BETH dataset, this task does three things per split
(training / validation / testing):

  1. Drop columns  — removes "stackAddresses" (binary call-stack data we
     can't use).  The "args" column is NOT dropped here; it is kept so
     feature_engineering can parse file paths and flags from it first.

  2. Fill missing values — numeric columns (timestamp, argsNum, returnValue)
     get the TRAINING-SET median.  Categorical/string columns get "UNKNOWN"
     (configurable via cleaning.categorical_fill in the YAML).

  3. Separate labels — moves "evil" (attack label, 0/1) and "sus"
     (suspicious flag, 0/1) into their own DataFrames so they are never
     accidentally fed to the model as input features.

After cleaning, ctx_data contains:
  train_df / val_df / test_df          — feature DataFrames (no labels)
  train_labels / val_labels / test_labels — label DataFrames (evil + sus)

Reads from ctx_data: raw_frames, raw_data_path, splits_config,
                     drop_columns, label_columns
Writes to ctx_data:  train_df, val_df, test_df,
                     train_labels, val_labels, test_labels,
                     feature_columns, cleaned_splits
"""

from pathlib import Path

import pandas as pd

from core.common.wfs.dtos import WfReq, WfResp
from core.common.wfs.interfaces import WfTask
from core.config import get_cfg
from core.logger import get_logger

log = get_logger(__name__)


class Cleaning(WfTask):

    def execute(self, req: WfReq, resp: WfResp) -> WfResp:
        # ---------- Gather everything the data_analysis task published ----------
        # raw_frames: {"training": DataFrame, "validation": DataFrame, "testing": DataFrame}
        # Already loaded from CSV by data_analysis — reusing avoids 3 redundant file reads.
        raw_frames: dict[str, pd.DataFrame] | None = resp.ctx_data.get("raw_frames")
        raw_data_path = resp.ctx_data.get("raw_data_path")

        # splits_config: {"training": "labelled_training_data.csv", ...}
        splits_config = resp.ctx_data.get("splits_config")

        # For BETH: drop_columns = ["stackAddresses"] (binary call-stack, unusable).
        # Note: "args" is in parse_then_drop, NOT drop — it stays for feature_engineering.
        drop_columns = resp.ctx_data.get("drop_columns", [])

        # For BETH: label_columns = ["evil", "sus"]
        # evil = main attack label, sus = weak suspicious signal for threshold tuning.
        label_columns = resp.ctx_data.get("label_columns", [])

        if not splits_config:
            resp.success = False
            resp.message = (
                "cleaning requires splits_config in ctx_data "
                "— ensure data_analysis ran first"
            )
            return resp

        frames: dict[str, pd.DataFrame] = {}
        labels: dict[str, pd.DataFrame] = {}
        fill_medians: pd.Series | None = None

        # From YAML cleaning.categorical_fill — for BETH this is "UNKNOWN".
        # Used when a string column (processName, hostName, eventName) has NaN.
        categorical_fill = get_cfg(req.config, "cleaning.categorical_fill", "UNKNOWN")

        for split_name, filename in splits_config.items():
            # ---------- Step 1: Get the DataFrame (prefer memory, fallback disk) ---
            if raw_frames and split_name in raw_frames:
                df = raw_frames[split_name].copy()
                log.debug("Reusing pre-loaded %s (%d rows, %d cols)", split_name, len(df), len(df.columns))
            elif raw_data_path:
                filepath = Path(raw_data_path) / filename
                log.debug("Loading %s from disk: %s", split_name, filepath.name)
                df = pd.read_csv(filepath)
            else:
                resp.success = False
                resp.message = f"No data source for split '{split_name}' — need raw_frames or raw_data_path"
                return resp

            # ---------- Step 2: Drop unusable columns --------------------------
            # For BETH: drops "stackAddresses" (memory addresses from call stack).
            existing_drops = [c for c in drop_columns if c in df.columns]
            if existing_drops:
                df = df.drop(columns=existing_drops)
                log.debug("  Dropped %s", existing_drops)

            # ---------- Step 3: Handle missing values --------------------------
            # Strategy: numeric NaNs → training median, string NaNs → "UNKNOWN".
            # Medians are computed once from the FIRST split (training) to prevent
            # data leakage from val/test into the fill values.
            missing_counts = df.isnull().sum()
            cols_with_missing = missing_counts[missing_counts > 0]
            if len(cols_with_missing) > 0:
                log.debug("  Missing values found: %s", cols_with_missing.to_dict())

                if fill_medians is None:
                    # Compute medians from this split (training comes first in YAML
                    # key order), reused for val/test to avoid leakage.
                    fill_medians = df.select_dtypes(include="number").median()

                for col in cols_with_missing.index:
                    if df[col].dtype in ("float64", "int64"):
                        # e.g. if argsNum has NaN → fill with training median of argsNum
                        df[col] = df[col].fillna(fill_medians[col])
                    else:
                        # e.g. if processName has NaN → fill with "UNKNOWN"
                        df[col] = df[col].fillna(categorical_fill)

                log.debug("  Filled missing values (numeric=train median, categorical=%s)", categorical_fill)
            else:
                log.debug("  No missing values")

            # ---------- Step 4: Separate labels from features ------------------
            # For BETH: moves "evil" and "sus" into a separate DataFrame so they
            # never become model input.  Training has evil=0 for ALL rows (no attacks),
            # while testing has ~84% attacks.
            existing_labels = [c for c in label_columns if c in df.columns]
            split_labels = df[existing_labels].copy() if existing_labels else pd.DataFrame(index=df.index)
            df = df.drop(columns=existing_labels, errors="ignore")

            frames[split_name] = df
            labels[split_name] = split_labels
            log.debug("  %s cleaned: %d rows, %d feature cols, labels=%s",
                      split_name, len(df), len(df.columns), list(split_labels.columns))

        # Free the original raw DataFrames — cleaning has its own copies now
        del resp.ctx_data["raw_frames"]
        log.debug("Released raw_frames from ctx_data to free memory")

        # ---------- Publish cleaned data into ctx_data under standard keys -----
        # Maps split names → standard ctx_data keys:
        #   "training"   → ctx_data["train_df"]   + ctx_data["train_labels"]
        #   "validation"  → ctx_data["val_df"]     + ctx_data["val_labels"]
        #   "testing"    → ctx_data["test_df"]     + ctx_data["test_labels"]
        # These keys come from req.df_keys / req.label_keys (derived from YAML).
        split_names = list(splits_config.keys())
        df_keys = req.df_keys       # ("train_df", "val_df", "test_df")
        label_keys = req.label_keys  # ("train_labels", "val_labels", "test_labels")
        for df_key, lbl_key, split_name in zip(df_keys, label_keys, split_names):
            resp.ctx_data[df_key] = frames[split_name]
            resp.ctx_data[lbl_key] = labels[split_name]

        resp.ctx_data["feature_columns"] = list(frames[split_names[0]].columns)
        resp.ctx_data["cleaned_splits"] = split_names

        total_rows = sum(len(f) for f in frames.values())
        n_features = len(frames[split_names[0]].columns)
        resp.message = (
            f"Cleaning complete — {total_rows:,} total rows, "
            f"{n_features} feature columns (reused pre-loaded data, zero CSV re-reads)"
        )
        log.debug(resp.message)
        return resp


Task = Cleaning
