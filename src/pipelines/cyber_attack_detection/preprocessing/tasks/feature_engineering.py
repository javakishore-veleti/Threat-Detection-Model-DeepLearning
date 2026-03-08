"""Engineer domain-informed features from cleaned DataFrames.

Creates 11 features (6 from structured columns, 5 from args parsing),
then drops the raw args column. All thresholds are computed from the
training split only to prevent data leakage.

Reads from ctx_data: train_df, val_df, test_df, parse_then_drop_columns
Writes to ctx_data:  train_df, val_df, test_df (updated in-place), feature_columns
"""

import ast

import pandas as pd

from core.common.wfs.dtos import WfReq, WfResp
from core.common.wfs.interfaces import WfTask
from core.logger import get_logger

log = get_logger(__name__)

DF_KEYS = ("train_df", "val_df", "test_df")


class FeatureEngineering(WfTask):

    def execute(self, req: WfReq, resp: WfResp) -> WfResp:
        frames = {k: resp.ctx_data.get(k) for k in DF_KEYS}
        if any(v is None for v in frames.values()):
            resp.success = False
            resp.message = "feature_engineering requires train_df/val_df/test_df in ctx_data"
            return resp

        train_df: pd.DataFrame = frames["train_df"]

        args_threshold_95 = float(train_df["argsNum"].quantile(0.95))
        log.debug("argsNum 95th-percentile threshold (training): %.1f", args_threshold_95)

        for key in DF_KEYS:
            df: pd.DataFrame = frames[key]
            log.debug("Engineering features for %s (%d rows)", key, len(df))
            df = self._add_structured_features(df, args_threshold_95)
            df = self._add_args_features(df)
            frames[key] = df

        parse_then_drop = resp.ctx_data.get("parse_then_drop_columns", ["args"])
        for key in DF_KEYS:
            existing = [c for c in parse_then_drop if c in frames[key].columns]
            if existing:
                frames[key] = frames[key].drop(columns=existing)
                log.debug("Dropped raw columns %s from %s", existing, key)

        for key in DF_KEYS:
            resp.ctx_data[key] = frames[key]

        resp.ctx_data["feature_columns"] = list(frames["train_df"].columns)
        resp.ctx_data["engineered_features"] = [
            "is_root", "return_negative", "return_category",
            "is_child_of_init", "is_orphan", "is_high_args",
            "args_touches_proc", "args_touches_etc", "args_has_write_flag",
            "args_is_hidden_path", "args_has_pathname",
        ]

        n_features = len(frames["train_df"].columns)
        total_rows = sum(len(frames[k]) for k in DF_KEYS)
        resp.message = (
            f"Feature engineering complete — {total_rows:,} rows, "
            f"{n_features} features (11 new, args column dropped)"
        )
        log.debug(resp.message)
        return resp

    @staticmethod
    def _add_structured_features(df: pd.DataFrame, args_threshold: float) -> pd.DataFrame:
        df["is_root"] = (df["userId"] == 0).astype(int)
        df["return_negative"] = (df["returnValue"] < 0).astype(int)
        df["return_category"] = df["returnValue"].apply(
            lambda v: 0 if v == 0 else (1 if v < 0 else 2)
        )
        df["is_child_of_init"] = (df["parentProcessId"] == 1).astype(int)
        df["is_orphan"] = (df["parentProcessId"] == 0).astype(int)
        df["is_high_args"] = (df["argsNum"] > args_threshold).astype(int)
        return df

    @staticmethod
    def _parse_single_args(args_str: str) -> dict[str, int]:
        result = {
            "args_touches_proc": 0,
            "args_touches_etc": 0,
            "args_has_write_flag": 0,
            "args_is_hidden_path": 0,
            "args_has_pathname": 0,
        }
        try:
            args_list = ast.literal_eval(args_str)
        except (ValueError, SyntaxError, TypeError):
            return result

        for arg in args_list:
            name = arg.get("name", "")
            val = str(arg.get("value", ""))

            if name in ("pathname", "filename"):
                result["args_has_pathname"] = 1
                if "/proc/" in val:
                    result["args_touches_proc"] = 1
                if "/etc/" in val:
                    result["args_touches_etc"] = 1
                if "/." in val:
                    result["args_is_hidden_path"] = 1

            if name == "flags" and any(
                f in val for f in ("WRONLY", "RDWR", "CREAT")
            ):
                result["args_has_write_flag"] = 1

        return result

    @classmethod
    def _add_args_features(cls, df: pd.DataFrame) -> pd.DataFrame:
        if "args" not in df.columns:
            for col in ("args_touches_proc", "args_touches_etc",
                        "args_has_write_flag", "args_is_hidden_path",
                        "args_has_pathname"):
                df[col] = 0
            log.debug("  args column not present — filled args features with 0")
            return df

        parsed = df["args"].apply(cls._parse_single_args)
        args_df = pd.DataFrame(parsed.tolist(), index=df.index)
        df = pd.concat([df, args_df], axis=1)
        return df


Task = FeatureEngineering
