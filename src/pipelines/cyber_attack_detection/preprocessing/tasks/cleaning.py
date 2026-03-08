"""Clean raw DataFrames: drop unusable columns, handle missing values, separate labels.

All column names, split info, and label columns are read from ctx_data
(populated by the data_analysis task). DataFrames are reused from
ctx_data["raw_frames"] to avoid re-reading CSVs from disk.
"""

from pathlib import Path

import pandas as pd

from core.common.wfs.dtos import WfReq, WfResp
from core.common.wfs.interfaces import WfTask
from core.logger import get_logger

log = get_logger(__name__)


class Cleaning(WfTask):

    def execute(self, req: WfReq, resp: WfResp) -> WfResp:
        raw_frames: dict[str, pd.DataFrame] | None = resp.ctx_data.get("raw_frames")
        raw_data_path = resp.ctx_data.get("raw_data_path")
        splits_config = resp.ctx_data.get("splits_config")
        drop_columns = resp.ctx_data.get("drop_columns", [])
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
        categorical_fill = "UNKNOWN"

        for split_name, filename in splits_config.items():
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

            existing_drops = [c for c in drop_columns if c in df.columns]
            if existing_drops:
                df = df.drop(columns=existing_drops)
                log.debug("  Dropped %s", existing_drops)

            missing_counts = df.isnull().sum()
            cols_with_missing = missing_counts[missing_counts > 0]
            if len(cols_with_missing) > 0:
                log.debug("  Missing values found: %s", cols_with_missing.to_dict())

                if fill_medians is None:
                    fill_medians = df.select_dtypes(include="number").median()

                for col in cols_with_missing.index:
                    if df[col].dtype in ("float64", "int64"):
                        df[col] = df[col].fillna(fill_medians[col])
                    else:
                        df[col] = df[col].fillna(categorical_fill)

                log.debug("  Filled missing values (numeric=train median, categorical=%s)", categorical_fill)
            else:
                log.debug("  No missing values")

            existing_labels = [c for c in label_columns if c in df.columns]
            split_labels = df[existing_labels].copy() if existing_labels else pd.DataFrame(index=df.index)
            df = df.drop(columns=existing_labels, errors="ignore")

            frames[split_name] = df
            labels[split_name] = split_labels
            log.debug("  %s cleaned: %d rows, %d feature cols, labels=%s",
                      split_name, len(df), len(df.columns), list(split_labels.columns))

        del resp.ctx_data["raw_frames"]
        log.debug("Released raw_frames from ctx_data to free memory")

        split_names = list(splits_config.keys())
        for i, name in enumerate(split_names):
            key_prefix = "train" if i == 0 else ("val" if "val" in name else "test")
            resp.ctx_data[f"{key_prefix}_df"] = frames[name]
            resp.ctx_data[f"{key_prefix}_labels"] = labels[name]

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
