"""Config-driven feature engineering engine.

Reads feature definitions from req.config["features"] and applies them
generically.  No dataset-specific column names are hardcoded — everything
comes from configs/cyber_attack_detection/default.yaml.

For the BETH dataset the YAML defines 11 new binary/categorical features
derived from the cleaned columns:

  Structured features (from numeric / ID columns)
  ------------------------------------------------
  is_root           – 1 when userId == 0 (process runs as root)
  return_negative   – 1 when returnValue < 0 (syscall error / permission denied)
  return_category   – buckets returnValue into 0=zero, 1=negative, 2=positive
  is_child_of_init  – 1 when parentProcessId == 1 (spawned by init)
  is_orphan         – 1 when parentProcessId == 0 (no parent — kernel or zombie)
  is_high_args      – 1 when argsNum > 95th-percentile of training data
                      (unusually many syscall arguments → complex/suspicious call)

  Args-parsed features (from the raw `args` JSON-like string column)
  -------------------------------------------------------------------
  args_touches_proc – 1 when any pathname contains "/proc/" (attackers read
                      /proc/self/maps, /proc/version, etc.)
  args_touches_etc  – 1 when pathname contains "/etc/" (attackers read
                      /etc/passwd, /etc/shadow, etc.)
  args_is_hidden_path – 1 when pathname contains "/." (dotfile = hidden file,
                      often used by rootkits to hide payloads)
  args_has_write_flag – 1 when syscall flags contain WRONLY, RDWR, or CREAT
                      (write/create operations → data exfil or dropper)
  args_has_pathname – 1 when the args list contains *any* pathname/filename arg
                      (file-touching syscall vs memory-only syscall)

After creating these features the raw `args` column is dropped (it was
only kept through cleaning so this task could parse it).

Reads from ctx_data: train_df, val_df, test_df
Writes to ctx_data:  train_df, val_df, test_df (with new features, args dropped),
                     feature_columns, engineered_features
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
        # e.g. ("train_df", "val_df", "test_df") — derived from YAML splits
        df_keys = req.df_keys
        train_key = df_keys[0]  # "train_df" — we compute thresholds from training only

        frames = {k: resp.ctx_data.get(k) for k in df_keys}
        if any(v is None for v in frames.values()):
            resp.success = False
            resp.message = f"feature_engineering requires {df_keys} in ctx_data"
            return resp

        # YAML config["features"]["structured"] defines the 6 structured features
        # (is_root, return_negative, return_category, etc.)
        structured_specs = get_cfg(cfg, "features.structured", [])

        # YAML config["features"]["args_parsing"] defines the 5 features extracted
        # from the `args` JSON column (args_touches_proc, args_has_write_flag, etc.)
        args_cfg = get_cfg(cfg, "features.args_parsing", {})

        # Columns to drop AFTER we've extracted features from them.
        # For BETH this is ["args"] — we keep it through cleaning just so we
        # can parse path/flag signals here, then drop the raw string column.
        parse_then_drop = resp.ctx_data.get("parse_then_drop_columns",
                                            get_cfg(cfg, "columns.parse_then_drop.names", []))

        # ---------- Phase 1: Compute thresholds from training data only ----------
        # e.g. for is_high_args we need the 95th-percentile of argsNum in training
        # so the same cutoff applies consistently to val and test splits.
        train_df: pd.DataFrame = frames[train_key]
        thresholds = self._compute_thresholds(train_df, structured_specs)

        # ---------- Phase 2: Apply features to every split -----------------------
        engineered_names: list[str] = []
        for key in df_keys:
            df: pd.DataFrame = frames[key]
            log.debug("Engineering features for %s (%d rows)", key, len(df))

            # Add the 6 structured binary/bucketed columns
            df = self._apply_structured(df, structured_specs, thresholds)

            # Parse the `args` column into 5 binary signal columns
            if args_cfg:
                df = self._apply_args_parsing(df, args_cfg)
            frames[key] = df

        # ---------- Collect the names of all new features we created -------------
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

        # ---------- Phase 3: Drop the raw columns we've finished parsing ---------
        # For BETH: drops the raw `args` string column now that we've extracted
        # args_touches_proc, args_touches_etc, args_is_hidden_path,
        # args_has_write_flag, and args_has_pathname from it.
        for key in df_keys:
            existing = [c for c in parse_then_drop if c in frames[key].columns]
            if existing:
                frames[key] = frames[key].drop(columns=existing)
                log.debug("Dropped raw columns %s from %s", existing, key)

        # ---------- Publish updated DataFrames back to ctx_data ------------------
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
        """Pre-compute any data-dependent thresholds using TRAINING data only.

        For BETH: the `is_high_args` feature uses the 95th percentile of
        argsNum in the training split.  If argsNum's p95 is say 8, then any
        row with argsNum > 8 is flagged as "unusually complex syscall".
        We compute this once from training and apply the same cutoff to val/test
        so the model never peeks at evaluation data.
        """
        thresholds: dict[str, float] = {}
        for spec in specs:
            if spec["type"] == "gt_quantile":
                col = spec["column"]       # e.g. "argsNum"
                q = spec.get("quantile", 0.95)
                thresholds[spec["name"]] = float(train_df[col].quantile(q))
                log.debug("%s %s-percentile threshold (training): %.1f",
                          col, q, thresholds[spec["name"]])
        return thresholds

    @staticmethod
    def _apply_structured(df: pd.DataFrame, specs: list[dict],
                          thresholds: dict) -> pd.DataFrame:
        """Create new columns from existing numeric columns using simple rules.

        Each spec in the YAML list becomes one new column.  BETH examples:

          type="eq"   → is_root:          userId == 0?              (1/0)
                      → is_child_of_init: parentProcessId == 1?     (1/0)
                      → is_orphan:        parentProcessId == 0?     (1/0)
          type="lt"   → return_negative:   returnValue < 0?          (1/0)
          type="binned" → return_category: returnValue → 0/1/2
                        (0 = zero / success, 1 = negative / error, 2 = positive / info)
          type="gt_quantile" → is_high_args: argsNum > p95 threshold  (1/0)
        """
        for spec in specs:
            name = spec["name"]
            col = spec["column"]
            feat_type = spec["type"]

            if col not in df.columns:
                log.debug("  Skipping %s — column %s not in DataFrame", name, col)
                continue

            # "eq" — exact match, e.g. userId == 0 → is_root
            if feat_type == "eq":
                df[name] = (df[col] == spec["value"]).astype(int)
            # "lt" — less than, e.g. returnValue < 0 → return_negative
            elif feat_type == "lt":
                df[name] = (df[col] < spec["value"]).astype(int)
            # "gt" — greater than a fixed value
            elif feat_type == "gt":
                df[name] = (df[col] > spec["value"]).astype(int)
            # "gt_quantile" — greater than a training-set percentile
            # e.g. argsNum > p95 → is_high_args (uses pre-computed threshold)
            elif feat_type == "gt_quantile":
                df[name] = (df[col] > thresholds[name]).astype(int)
            # "binned" — bucket into 3 categories: zero / negative / positive
            # For returnValue: 0=success, 1=error(negative), 2=info(positive)
            elif feat_type == "binned":
                df[name] = df[col].apply(
                    lambda v: 0 if v == 0 else (1 if v < 0 else 2)
                )
            else:
                log.debug("  Unknown feature type '%s' for %s — skipping", feat_type, name)

        return df

    @staticmethod
    def _apply_args_parsing(df: pd.DataFrame, args_cfg: dict) -> pd.DataFrame:
        """Parse the raw `args` string column into binary signal features.

        Each row in BETH has an `args` column that looks like a Python list of
        dicts, e.g.:  [{"name":"pathname","value":"/proc/self/maps"}, ...]

        This method:
        1. Parses that string with ast.literal_eval (safe: no code execution).
        2. Checks each arg dict for file-path arguments (pathname, filename)
           and flag arguments (flags like WRONLY, RDWR, CREAT).
        3. Sets binary 0/1 columns based on pattern matches.

        For BETH the 5 features extracted are:
          args_touches_proc  – pathname contains "/proc/"  (90x more in attacks!)
          args_touches_etc   – pathname contains "/etc/"   (credential harvesting)
          args_is_hidden_path – pathname contains "/."     (dotfile = rootkit hide)
          args_has_write_flag – flags contain WRONLY/RDWR/CREAT (file modification)
          args_has_pathname  – args list has any pathname/filename at all
        """
        source_col = args_cfg.get("source_column", "args")

        # If the `args` column was already dropped by a prior task, fill all
        # args-derived features with 0 so downstream tasks still see the columns.
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

        # Config-driven: which arg names carry file paths?
        # For BETH: {"pathname", "filename"}
        path_arg_names = set(args_cfg.get("path_arg_names", []))

        # Config-driven: which substrings in a path indicate attack signals?
        # e.g. [{"name":"args_touches_proc", "pattern":"/proc/"}, ...]
        path_signals = args_cfg.get("path_signals", [])

        # Config-driven: which flag values indicate suspicious file operations?
        # e.g. [{"name":"args_has_write_flag", "arg_name":"flags",
        #         "patterns":["WRONLY","RDWR","CREAT"]}]
        flag_signals = args_cfg.get("flag_signals", [])

        # A catch-all feature: does this syscall touch any file at all?
        has_pathname_feature = args_cfg.get("has_pathname_feature")

        all_feature_names = [s["name"] for s in path_signals]
        all_feature_names += [s["name"] for s in flag_signals]
        if has_pathname_feature:
            all_feature_names.append(has_pathname_feature)

        def parse_row(args_str: str) -> dict[str, int]:
            """Parse a single row's args string into binary features."""
            result = {n: 0 for n in all_feature_names}
            try:
                # e.g. [{"name":"pathname","value":"/proc/self/maps"},
                #        {"name":"flags","value":"O_RDONLY|O_CLOEXEC"}]
                args_list = ast.literal_eval(args_str)
            except (ValueError, SyntaxError, TypeError):
                return result

            for arg in args_list:
                arg_name = arg.get("name", "")
                val = str(arg.get("value", ""))

                # Check path-type arguments (pathname, filename)
                if arg_name in path_arg_names:
                    # This syscall touches a file → set the general flag
                    if has_pathname_feature:
                        result[has_pathname_feature] = 1
                    # Check each path pattern — e.g. does the path contain "/proc/"?
                    for sig in path_signals:
                        if sig["pattern"] in val:
                            result[sig["name"]] = 1

                # Check flag-type arguments (e.g. open() flags)
                for fsig in flag_signals:
                    if arg_name == fsig.get("arg_name", "flags"):
                        # e.g. "O_WRONLY|O_CREAT" → check if WRONLY, RDWR, or CREAT appears
                        if any(p in val for p in fsig.get("patterns", [])):
                            result[fsig["name"]] = 1

            return result

        # Apply row-by-row parsing and expand into new columns
        parsed = df[source_col].apply(parse_row)
        args_df = pd.DataFrame(parsed.tolist(), index=df.index)
        df = pd.concat([df, args_df], axis=1)
        return df


Task = FeatureEngineering
