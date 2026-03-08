"""Config-driven feature engineering engine.

Reads feature definitions from req.config["features"] and applies them
generically. Supports structured features (eq, lt, gt_quantile, binned)
and args-column parsing (path signals, flag signals). No dataset-specific
column names are hardcoded — everything comes from YAML config.

Reads from ctx_data: train_df, val_df, test_df
Writes to ctx_data:  train_df, val_df, test_df (updated), feature_columns
"""

import ast

import pandas as pd

from core.common.wfs.dtos import WfReq, WfResp
from core.common.wfs.interfaces import WfTask
from core.config import get_cfg
from core.logger import get_logger

log = get_logger(__name__)


class FeatureEngineering(WfTask):

    def execute(self, req: WfReq, resp: WfResp) -> WfResp:
        cfg = req.config
        df_keys = req.df_keys
        train_key = df_keys[0]

        frames = {k: resp.ctx_data.get(k) for k in df_keys}
        if any(v is None for v in frames.values()):
            resp.success = False
            resp.message = f"feature_engineering requires {df_keys} in ctx_data"
            return resp

        structured_specs = get_cfg(cfg, "features.structured", [])
        args_cfg = get_cfg(cfg, "features.args_parsing", {})
        parse_then_drop = resp.ctx_data.get("parse_then_drop_columns",
                                            get_cfg(cfg, "columns.parse_then_drop.names", []))

        train_df: pd.DataFrame = frames[train_key]
        thresholds = self._compute_thresholds(train_df, structured_specs)

        engineered_names: list[str] = []
        for key in df_keys:
            df: pd.DataFrame = frames[key]
            log.debug("Engineering features for %s (%d rows)", key, len(df))
            df = self._apply_structured(df, structured_specs, thresholds)
            if args_cfg:
                df = self._apply_args_parsing(df, args_cfg)
            frames[key] = df

        for spec in structured_specs:
            engineered_names.append(spec["name"])
        if args_cfg:
            for sig in args_cfg.get("path_signals", []):
                engineered_names.append(sig["name"])
            for sig in args_cfg.get("flag_signals", []):
                engineered_names.append(sig["name"])
            pn_feat = args_cfg.get("has_pathname_feature")
            if pn_feat:
                engineered_names.append(pn_feat)

        for key in df_keys:
            existing = [c for c in parse_then_drop if c in frames[key].columns]
            if existing:
                frames[key] = frames[key].drop(columns=existing)
                log.debug("Dropped raw columns %s from %s", existing, key)

        for key in df_keys:
            resp.ctx_data[key] = frames[key]

        resp.ctx_data["feature_columns"] = list(frames[train_key].columns)
        resp.ctx_data["engineered_features"] = engineered_names

        n_features = len(frames[train_key].columns)
        total_rows = sum(len(frames[k]) for k in df_keys)
        n_new = len(engineered_names)
        resp.message = (
            f"Feature engineering complete — {total_rows:,} rows, "
            f"{n_features} features ({n_new} new, raw parsed columns dropped)"
        )
        log.debug(resp.message)
        return resp

    @staticmethod
    def _compute_thresholds(train_df: pd.DataFrame, specs: list[dict]) -> dict:
        thresholds: dict[str, float] = {}
        for spec in specs:
            if spec["type"] == "gt_quantile":
                col = spec["column"]
                q = spec.get("quantile", 0.95)
                thresholds[spec["name"]] = float(train_df[col].quantile(q))
                log.debug("%s %s-percentile threshold (training): %.1f",
                          col, q, thresholds[spec["name"]])
        return thresholds

    @staticmethod
    def _apply_structured(df: pd.DataFrame, specs: list[dict],
                          thresholds: dict) -> pd.DataFrame:
        for spec in specs:
            name = spec["name"]
            col = spec["column"]
            feat_type = spec["type"]

            if col not in df.columns:
                log.debug("  Skipping %s — column %s not in DataFrame", name, col)
                continue

            if feat_type == "eq":
                df[name] = (df[col] == spec["value"]).astype(int)
            elif feat_type == "lt":
                df[name] = (df[col] < spec["value"]).astype(int)
            elif feat_type == "gt":
                df[name] = (df[col] > spec["value"]).astype(int)
            elif feat_type == "gt_quantile":
                df[name] = (df[col] > thresholds[name]).astype(int)
            elif feat_type == "binned":
                df[name] = df[col].apply(
                    lambda v: 0 if v == 0 else (1 if v < 0 else 2)
                )
            else:
                log.debug("  Unknown feature type '%s' for %s — skipping", feat_type, name)

        return df

    @staticmethod
    def _apply_args_parsing(df: pd.DataFrame, args_cfg: dict) -> pd.DataFrame:
        source_col = args_cfg.get("source_column", "args")
        if source_col not in df.columns:
            for sig in args_cfg.get("path_signals", []):
                df[sig["name"]] = 0
            for sig in args_cfg.get("flag_signals", []):
                df[sig["name"]] = 0
            pn = args_cfg.get("has_pathname_feature")
            if pn:
                df[pn] = 0
            log.debug("  %s column not present — filled args features with 0", source_col)
            return df

        path_arg_names = set(args_cfg.get("path_arg_names", []))
        path_signals = args_cfg.get("path_signals", [])
        flag_signals = args_cfg.get("flag_signals", [])
        has_pathname_feature = args_cfg.get("has_pathname_feature")

        all_feature_names = [s["name"] for s in path_signals]
        all_feature_names += [s["name"] for s in flag_signals]
        if has_pathname_feature:
            all_feature_names.append(has_pathname_feature)

        def parse_row(args_str: str) -> dict[str, int]:
            result = {n: 0 for n in all_feature_names}
            try:
                args_list = ast.literal_eval(args_str)
            except (ValueError, SyntaxError, TypeError):
                return result

            for arg in args_list:
                arg_name = arg.get("name", "")
                val = str(arg.get("value", ""))

                if arg_name in path_arg_names:
                    if has_pathname_feature:
                        result[has_pathname_feature] = 1
                    for sig in path_signals:
                        if sig["pattern"] in val:
                            result[sig["name"]] = 1

                for fsig in flag_signals:
                    if arg_name == fsig.get("arg_name", "flags"):
                        if any(p in val for p in fsig.get("patterns", [])):
                            result[fsig["name"]] = 1

            return result

        parsed = df[source_col].apply(parse_row)
        args_df = pd.DataFrame(parsed.tolist(), index=df.index)
        df = pd.concat([df, args_df], axis=1)
        return df


Task = FeatureEngineering
