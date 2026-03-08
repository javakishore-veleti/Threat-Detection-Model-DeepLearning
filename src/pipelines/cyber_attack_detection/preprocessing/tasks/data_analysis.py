"""Programmatic EDA — generates versioned JSON + HTML reports with cybersecurity analyst insights."""

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from core.common.wfs.dtos import WfReq, WfResp
from core.common.wfs.interfaces import WfTask
from core.logger import get_logger

log = get_logger(__name__)

REPORT_DIR = Path.home() / "python_venvs" / "datasets" / "kaggle" / "beth-dataset" / "reports"
REPORT_VERSION = "v04"
JSON_FILE = REPORT_DIR / f"data_analysis_report_{REPORT_VERSION}.json"
HTML_FILE = REPORT_DIR / f"data_analysis_report_{REPORT_VERSION}.html"

SPLITS = {
    "training": "labelled_training_data.csv",
    "validation": "labelled_validation_data.csv",
    "testing": "labelled_testing_data.csv",
}

# -----------------------------------------------------------------------
# Column classification — the analyst's view of what each column REALLY is
# -----------------------------------------------------------------------
# Columns that pandas reads as int64/float64 but are actually categorical
# identifiers (the numeric value has no arithmetic meaning).
CATEGORICAL_IDS = ["processId", "threadId", "parentProcessId", "userId",
                   "mountNamespace", "eventId"]

# Columns that are truly numeric (arithmetic on them is meaningful)
TRUE_NUMERIC = ["timestamp", "argsNum", "returnValue"]

# String columns (already detected as object dtype by pandas)
STRING_CATEGORICAL = ["processName", "hostName", "eventName"]

# Columns to drop (stackAddresses only — args is parsed for features in feature_engineering)
COMPLEX_DROP = ["stackAddresses"]

# Columns to parse in feature engineering then drop (raw string not usable by model directly)
PARSE_THEN_DROP = ["args"]

# Labels (not features)
LABEL_COLS = ["evil", "sus"]

COLUMN_CLASSIFICATION = {
    "true_numeric": {
        "columns": TRUE_NUMERIC,
        "treatment": "Scale with StandardScaler. Arithmetic relationships are meaningful.",
        "detail": (
            "timestamp = seconds since start (ordering matters). "
            "argsNum = count of arguments (more args = more complex call). "
            "returnValue = syscall result (0 = success, negative = error, positive = info)."
        ),
    },
    "categorical_id": {
        "columns": CATEGORICAL_IDS,
        "treatment": "Label-encode. Do NOT scale as numeric — the values are identifiers, "
                     "not quantities.",
        "detail": (
            "processId/threadId/parentProcessId = OS-assigned instance IDs, recycled over "
            "time. PID 500 is not 'more' than PID 250. "
            "userId = Linux UID (0=root, 65534=nobody, etc.) — each number is a distinct "
            "identity, not a quantity. "
            "mountNamespace = kernel namespace ID (container boundary). The billion-range "
            "numbers are opaque identifiers. "
            "eventId = maps to a specific syscall type (e.g., 157=prctl). It's an enum, "
            "not a measurement."
        ),
    },
    "string_categorical": {
        "columns": STRING_CATEGORICAL,
        "treatment": "Label-encode with UNKNOWN handling for unseen values in val/test.",
        "detail": (
            "processName = program name (systemd, sshd, etc.). "
            "hostName = server identifier. "
            "eventName = human-readable syscall name."
        ),
    },
    "complex_drop": {
        "columns": COMPLEX_DROP,
        "treatment": "Drop — not usable without deep binary analysis.",
        "detail": (
            "stackAddresses = list of memory addresses from the call stack. "
            "Requires binary analysis expertise to extract useful features."
        ),
    },
    "parse_then_drop": {
        "columns": PARSE_THEN_DROP,
        "treatment": "Parse in feature engineering to extract binary features, then drop raw string.",
        "detail": (
            "args = nested JSON-like string with syscall argument details. Contains file paths, "
            "flags, and argument values. Data analysis revealed STRONG attack signals: "
            "normal ops access /proc/ 34% of the time vs 0.4% in attacks (90x difference), "
            "normal has 13x more write flags than attacks. Extract binary features "
            "(args_touches_proc, args_touches_etc, args_has_write_flag, args_is_hidden_path, "
            "args_has_pathname) then drop the raw column."
        ),
    },
    "labels": {
        "columns": LABEL_COLS,
        "treatment": "Separate from features before any transformation. evil = target, "
                     "sus = auxiliary signal for threshold tuning.",
        "detail": (
            "evil: 0 = normal, 1 = attack. Training has 0 attacks (anomaly detection setup). "
            "sus: 0 = normal, 1 = suspicious. Weak label available even in training."
        ),
    },
}


class DataAnalysis(WfTask):

    def execute(self, req: WfReq, resp: WfResp) -> WfResp:
        raw_data_path = resp.ctx_data.get("raw_data_path")
        if not raw_data_path:
            resp.success = False
            resp.message = "raw_data_path not found in ctx_data — run download first"
            return resp

        data_dir = Path(raw_data_path)
        frames: dict[str, pd.DataFrame] = {}
        split_stats: dict[str, dict] = {}

        for split_name, filename in SPLITS.items():
            filepath = data_dir / filename
            if not filepath.exists():
                log.debug("Skipping %s — file not found: %s", split_name, filepath)
                continue
            log.debug("Analyzing %s (%s)", split_name, filepath.name)
            df = pd.read_csv(filepath)
            frames[split_name] = df
            split_stats[split_name] = self._analyze_split(df, split_name)

        col_class_stats = self._column_classification_stats(frames)
        insights = self._generate_insights(frames, split_stats)
        recommendations = self._generate_recommendations(frames, split_stats)
        cross_split = self._cross_split_analysis(frames)

        analyst_thinking = self._analyst_thinking(frames, split_stats)

        report = {
            "report_version": REPORT_VERSION,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "column_classification": COLUMN_CLASSIFICATION,
            "column_classification_stats": col_class_stats,
            "splits": split_stats,
            "cross_split_analysis": cross_split,
            "analyst_insights": insights,
            "recommendations": recommendations,
            "analyst_thinking": analyst_thinking,
        }

        REPORT_DIR.mkdir(parents=True, exist_ok=True)
        JSON_FILE.write_text(json.dumps(report, indent=2, default=str))
        log.debug("JSON report saved to %s", JSON_FILE)

        html = self._render_html(report, frames)
        HTML_FILE.write_text(html)
        log.debug("HTML report saved to %s", HTML_FILE)

        repo_root = Path(__file__).resolve()
        for parent in repo_root.parents:
            if (parent / ".git").exists():
                repo_root = parent
                break
        repo_html = repo_root / "data_analysis_report.html"
        repo_html.write_text(html)
        log.debug("Latest HTML report copied to repo root: %s", repo_html)

        resp.message = f"Data analysis {REPORT_VERSION} complete — reports at {REPORT_DIR}"
        resp.ctx_data["analysis_report_path"] = str(JSON_FILE)
        resp.ctx_data["analysis_html_path"] = str(HTML_FILE)
        resp.ctx_data["analysis_report"] = report

        resp.ctx_data["splits_config"] = dict(SPLITS)
        resp.ctx_data["raw_frames"] = frames
        resp.ctx_data["column_classification"] = COLUMN_CLASSIFICATION
        resp.ctx_data["drop_columns"] = list(COMPLEX_DROP)
        resp.ctx_data["parse_then_drop_columns"] = list(PARSE_THEN_DROP)
        resp.ctx_data["label_columns"] = list(LABEL_COLS)
        resp.ctx_data["target_col"] = LABEL_COLS[0]
        resp.ctx_data["aux_col"] = LABEL_COLS[1]
        resp.ctx_data["true_numeric_columns"] = list(TRUE_NUMERIC)
        resp.ctx_data["categorical_id_columns"] = list(CATEGORICAL_IDS)
        resp.ctx_data["string_categorical_columns"] = list(STRING_CATEGORICAL)
        resp.ctx_data["all_categorical_columns"] = list(STRING_CATEGORICAL) + list(CATEGORICAL_IDS)

        log.debug("Published to ctx_data: splits_config=%s, drop=%s, labels=%s, "
                  "numeric=%s, cat_id=%s, string_cat=%s",
                  list(SPLITS.keys()), COMPLEX_DROP, LABEL_COLS,
                  TRUE_NUMERIC, CATEGORICAL_IDS, STRING_CATEGORICAL)
        return resp

    # ------------------------------------------------------------------
    # Column classification statistics
    # ------------------------------------------------------------------

    def _column_classification_stats(self, frames: dict[str, pd.DataFrame]) -> dict:
        """Compute per-column stats that prove ID columns are categorical, not numeric."""
        if "training" not in frames:
            return {}

        train = frames["training"]
        result = {}

        for col in CATEGORICAL_IDS:
            if col not in train.columns:
                continue
            unique = train[col].nunique()
            total = len(train)
            top_10 = train[col].value_counts().head(10)
            top_10_pct = round(float(top_10.sum()) / total * 100, 2)
            result[col] = {
                "unique_values": int(unique),
                "cardinality_ratio": round(unique / total * 100, 4),
                "top_10_values": {str(k): int(v) for k, v in top_10.items()},
                "top_10_coverage_pct": top_10_pct,
                "why_categorical": self._why_categorical(col, train),
            }

        return result

    @staticmethod
    def _why_categorical(col: str, df: pd.DataFrame) -> str:
        unique = df[col].nunique()
        total = len(df)

        if col == "userId":
            root_pct = round(float((df[col] == 0).mean() * 100), 1)
            return (f"{unique} distinct users. {root_pct}% are root (uid=0). "
                    "Each UID is a distinct identity — uid 100 is not 'more' than uid 0.")
        if col == "eventId":
            return (f"{unique} distinct event types (syscall codes). "
                    "eventId 157 (prctl) is not 'greater than' eventId 2 (open). "
                    "These are enum-like codes that map to specific operations.")
        if col == "mountNamespace":
            return (f"{unique} distinct namespaces. Values like 4026532231 are "
                    "kernel-assigned opaque identifiers. Arithmetic on them is meaningless.")
        if col in ("processId", "parentProcessId", "threadId"):
            ratio = round(unique / total * 100, 2)
            return (f"{unique:,} unique values out of {total:,} rows ({ratio}% cardinality). "
                    "OS-assigned instance IDs, recycled over time. No ordinal relationship.")
        return f"{unique} unique values — identifier, not a measurement."

    # ------------------------------------------------------------------
    # Per-split statistics
    # ------------------------------------------------------------------

    def _analyze_split(self, df: pd.DataFrame, split_name: str) -> dict:
        numeric_cols = df.select_dtypes(include="number").columns.tolist()
        categorical_cols = df.select_dtypes(include="object").columns.tolist()

        analysis = {
            "row_count": len(df),
            "column_count": len(df.columns),
            "columns": list(df.columns),
            "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
            "memory_mb": round(df.memory_usage(deep=True).sum() / 1024 / 1024, 2),
            "missing_values": {
                col: int(count)
                for col, count in df.isnull().sum().items()
                if count > 0
            },
            "missing_pct": {
                col: round(count / len(df) * 100, 2)
                for col, count in df.isnull().sum().items()
                if count > 0
            },
            "duplicate_rows": int(df.duplicated().sum()),
            "class_distribution": {
                "evil": df["evil"].value_counts().to_dict() if "evil" in df.columns else {},
                "sus": df["sus"].value_counts().to_dict() if "sus" in df.columns else {},
            },
            "true_numeric_stats": {
                col: self._numeric_col_stats(df[col])
                for col in TRUE_NUMERIC
                if col in df.columns
            },
            "categorical_id_stats": {
                col: self._id_col_stats(df[col])
                for col in CATEGORICAL_IDS
                if col in df.columns
            },
            "string_categorical_stats": {
                col: {
                    "unique_count": int(df[col].nunique()),
                    "top_10": df[col].value_counts().head(10).to_dict(),
                    "least_common_5": df[col].value_counts().tail(5).to_dict(),
                }
                for col in categorical_cols
            },
        }

        if "sus" in df.columns and "evil" in df.columns:
            sus_evil_cross = pd.crosstab(df["sus"], df["evil"]).to_dict()
            analysis["sus_evil_crosstab"] = {
                str(k): {str(k2): v2 for k2, v2 in v.items()}
                for k, v in sus_evil_cross.items()
            }

        log.debug(
            "  %s: %d rows, %d cols, evil=%s, sus=%s, duplicates=%d",
            split_name, analysis["row_count"], analysis["column_count"],
            analysis["class_distribution"]["evil"],
            analysis["class_distribution"]["sus"],
            analysis["duplicate_rows"],
        )
        return analysis

    @staticmethod
    def _numeric_col_stats(s: pd.Series) -> dict:
        return {
            "min": float(s.min()),
            "max": float(s.max()),
            "mean": round(float(s.mean()), 4),
            "std": round(float(s.std()), 4),
            "median": float(s.median()),
            "q25": float(s.quantile(0.25)),
            "q75": float(s.quantile(0.75)),
            "iqr": round(float(s.quantile(0.75) - s.quantile(0.25)), 4),
            "zeros_pct": round(float((s == 0).mean() * 100), 2),
            "negative_pct": round(float((s < 0).mean() * 100), 2),
        }

    @staticmethod
    def _id_col_stats(s: pd.Series) -> dict:
        top_10 = s.value_counts().head(10)
        return {
            "unique_count": int(s.nunique()),
            "cardinality_pct": round(s.nunique() / len(s) * 100, 4),
            "top_10": {str(k): int(v) for k, v in top_10.items()},
            "top_10_coverage_pct": round(float(top_10.sum()) / len(s) * 100, 2),
        }

    # ------------------------------------------------------------------
    # Cross-split analysis
    # ------------------------------------------------------------------

    def _cross_split_analysis(self, frames: dict[str, pd.DataFrame]) -> dict:
        result = {}
        if "training" not in frames:
            return result

        train = frames["training"]

        all_cat_cols = STRING_CATEGORICAL + CATEGORICAL_IDS
        vocab_overlap = {}
        for col in all_cat_cols:
            if col not in train.columns:
                continue
            train_vals = set(train[col].dropna().astype(str).unique())
            for sname, sdf in frames.items():
                if sname == "training":
                    continue
                other_vals = set(sdf[col].dropna().astype(str).unique())
                unseen = other_vals - train_vals
                vocab_overlap[f"{col}_{sname}_unseen"] = {
                    "count": len(unseen),
                    "examples": sorted(list(unseen))[:10],
                }
        result["category_unseen_in_training"] = vocab_overlap

        drift = {}
        for col in TRUE_NUMERIC:
            if col not in train.columns:
                continue
            train_mean = float(train[col].mean())
            train_std = float(train[col].std()) or 1.0
            for sname, sdf in frames.items():
                if sname == "training":
                    continue
                other_mean = float(sdf[col].mean())
                z_shift = abs(other_mean - train_mean) / train_std
                drift[f"{col}_{sname}"] = {
                    "train_mean": round(train_mean, 4),
                    "other_mean": round(other_mean, 4),
                    "z_shift": round(z_shift, 4),
                    "significant": z_shift > 2.0,
                }
        result["distribution_drift"] = drift
        return result

    # ------------------------------------------------------------------
    # Cybersecurity analyst insights
    # ------------------------------------------------------------------

    def _generate_insights(self, frames: dict[str, pd.DataFrame],
                           stats: dict[str, dict]) -> list[dict]:
        insights = []

        # --- anomaly detection framing ---
        if "training" in stats:
            evil_dist = stats["training"]["class_distribution"]["evil"]
            attack_count = evil_dist.get(1, 0)
            normal_count = evil_dist.get(0, 0)
            insights.append({
                "id": "anomaly_framing",
                "severity": "critical",
                "title": "This Is Anomaly Detection, Not Classification",
                "detail": (
                    f"The training set has {normal_count:,} normal samples and "
                    f"{attack_count:,} attack samples. With zero attacks in training, "
                    "a standard classifier cannot learn what attacks look like. "
                    "The model must learn the distribution of NORMAL behavior and "
                    "flag deviations as anomalies. An autoencoder trained on normal-only "
                    "data is the recommended approach."
                ),
            })

        # --- COLUMN MISCLASSIFICATION (key new insight) ---
        insights.append({
            "id": "column_misclassification",
            "severity": "critical",
            "title": "6 Columns Look Numeric But Are Actually Categorical Identifiers",
            "detail": (
                "processId, threadId, parentProcessId, userId, mountNamespace, and eventId "
                "are stored as int64 in the CSV, but they are IDENTIFIERS, not quantities. "
                "PID 500 is not 'twice' PID 250 — they're just different processes. "
                "userId 0 (root) is not 'less than' userId 100 (a service account). "
                "eventId 157 (prctl) is not 'greater than' eventId 2 (open). "
                "mountNamespace values like 4026532231 are opaque kernel IDs. "
                "TREATING THESE AS NUMERIC TELLS THE MODEL THAT BIGGER = MORE, WHICH IS "
                "WRONG. Label-encode them like any other categorical column. Only timestamp, "
                "argsNum, and returnValue are truly numeric."
            ),
        })

        # --- test set class flip ---
        if "testing" in stats:
            test_evil = stats["testing"]["class_distribution"]["evil"]
            test_attack = test_evil.get(1, 0)
            test_normal = test_evil.get(0, 0)
            test_total = test_attack + test_normal
            attack_pct = round(test_attack / test_total * 100, 1) if test_total else 0
            insights.append({
                "id": "test_class_imbalance",
                "severity": "warning",
                "title": "Test Set Is Attack-Heavy (Inverted Distribution)",
                "detail": (
                    f"The test set is {attack_pct}% attacks ({test_attack:,} of "
                    f"{test_total:,}). This is the OPPOSITE of the training set. "
                    "This means accuracy alone is misleading — a model that predicts "
                    f"'attack' for everything would score ~{attack_pct}%. Use precision, "
                    "recall, F1, and ROC-AUC as evaluation metrics instead."
                ),
            })

        # --- suspicious as weak signal ---
        if "training" in stats:
            sus_dist = stats["training"]["class_distribution"]["sus"]
            sus_count = sus_dist.get(1, 0)
            if sus_count > 0:
                insights.append({
                    "id": "sus_weak_label",
                    "severity": "info",
                    "title": f"'sus' Column Provides Weak Signal ({sus_count:,} Rows)",
                    "detail": (
                        f"Even though all training rows are evil=0, {sus_count:,} rows "
                        "are flagged as suspicious (sus=1). These represent borderline "
                        "behavior. During validation, compare reconstruction errors of "
                        "sus=0 vs sus=1 rows — if the model assigns higher error to "
                        "sus=1, it's learning meaningful patterns. Use this to tune "
                        "the anomaly threshold."
                    ),
                })

        # --- true numeric analysis ---
        if "training" in stats:
            rv = stats["training"]["true_numeric_stats"].get("returnValue", {})
            neg_pct = rv.get("negative_pct", 0)
            if neg_pct > 0:
                insights.append({
                    "id": "return_value_signal",
                    "severity": "info",
                    "title": f"returnValue Has {neg_pct}% Negative Values in Training",
                    "detail": (
                        "Negative return values typically indicate system call failures. "
                        "In an attack scenario, adversaries may trigger more errors "
                        "(permission denied, file not found, etc.). Engineer features: "
                        "'is_return_negative' (binary), and consider binning returnValue "
                        "into categories: success (0), error (negative), info (positive)."
                    ),
                })

        # --- userId / root ---
        if "training" in frames:
            train = frames["training"]
            root_pct = round(float((train["userId"] == 0).mean() * 100), 2)
            unique_users = train["userId"].nunique()
            insights.append({
                "id": "root_activity",
                "severity": "info" if root_pct < 50 else "warning",
                "title": f"{root_pct}% of Training Events Run as Root ({unique_users} Distinct Users)",
                "detail": (
                    f"Root (uid 0) has unrestricted system access. {root_pct}% root in "
                    f"training with only {unique_users} distinct user IDs confirms userId "
                    "should be treated as a categorical column, not numeric. The model "
                    "needs to learn that uid 0 is QUALITATIVELY different from uid 100, "
                    "not that it's 'less than' uid 100."
                ),
            })

        # --- eventId cardinality ---
        if "training" in frames:
            train = frames["training"]
            eid_unique = train["eventId"].nunique()
            insights.append({
                "id": "eventid_categorical",
                "severity": "info",
                "title": f"eventId Has Only {eid_unique} Distinct Values — It's an Enum",
                "detail": (
                    f"Despite being stored as int64, eventId has only {eid_unique} distinct "
                    "values across {0:,} rows. Each maps to a specific syscall: 157=prctl, "
                    "2=open, etc. This is clearly categorical. Label-encoding preserves the "
                    "identity while preventing the model from assuming ordinal relationships."
                ).format(len(train)),
            })

        # --- duplicate rows ---
        for sname, sstat in stats.items():
            dup = sstat.get("duplicate_rows", 0)
            if dup > 0:
                pct = round(dup / sstat["row_count"] * 100, 2)
                insights.append({
                    "id": f"duplicates_{sname}",
                    "severity": "warning",
                    "title": f"{sname.title()} Has {dup:,} Duplicate Rows ({pct}%)",
                    "detail": (
                        "Duplicate rows can bias the model — it sees these patterns "
                        "more often, giving them more weight. Decide: if duplicates "
                        "represent real repeated events (e.g., heartbeat signals), keep "
                        "them. If they're data collection artifacts, drop them."
                    ),
                })

        # --- data drift (only on TRUE numeric) ---
        if "training" in frames and "testing" in frames:
            train = frames["training"]
            test = frames["testing"]
            drifted = []
            for col in TRUE_NUMERIC:
                if col not in train.columns:
                    continue
                t_std = float(train[col].std()) or 1.0
                z = abs(float(test[col].mean()) - float(train[col].mean())) / t_std
                if z > 2.0:
                    drifted.append((col, round(z, 2)))
            if drifted:
                cols_str = ", ".join(f"{c} (z={z})" for c, z in drifted)
                insights.append({
                    "id": "data_drift",
                    "severity": "warning",
                    "title": f"{len(drifted)} True-Numeric Features Show Distribution Drift",
                    "detail": (
                        f"These truly numeric columns have significantly different means "
                        f"in test vs training (>2σ): {cols_str}. "
                        "This is EXPECTED because the test set contains attacks."
                    ),
                })

        # --- unseen categories (strings + IDs) ---
        if "training" in frames:
            for col in STRING_CATEGORICAL + CATEGORICAL_IDS:
                if col not in frames["training"].columns:
                    continue
                train_vals = set(frames["training"][col].dropna().astype(str).unique())
                for sname, sdf in frames.items():
                    if sname == "training":
                        continue
                    other_vals = set(sdf[col].dropna().astype(str).unique())
                    unseen = other_vals - train_vals
                    if unseen:
                        insights.append({
                            "id": f"unseen_{col}_{sname}",
                            "severity": "warning",
                            "title": f"{len(unseen)} Unseen '{col}' Values in {sname.title()}",
                            "detail": (
                                f"Values in {sname} not seen during training: "
                                f"{sorted(list(unseen))[:10]}"
                                f"{'...' if len(unseen) > 10 else ''}. "
                                "The label encoder must handle unknown values gracefully "
                                "(map to UNKNOWN token). For ID columns, unseen IDs in the "
                                "test set may themselves be attack indicators."
                            ),
                        })

        return insights

    # ------------------------------------------------------------------
    # Recommendations
    # ------------------------------------------------------------------

    def _generate_recommendations(self, frames: dict[str, pd.DataFrame],
                                  stats: dict[str, dict]) -> list[dict]:
        return [
            {
                "step": "cleaning",
                "priority": 1,
                "action": "Drop 'stackAddresses'. KEEP 'args' for feature extraction.",
                "reason": (
                    "stackAddresses needs binary analysis — drop it. args contains file "
                    "paths and flags with strong attack signal (90x /proc/ access difference "
                    "between normal and attacks). Parse in feature engineering."
                ),
            },
            {
                "step": "cleaning",
                "priority": 2,
                "action": "Check and handle missing values using TRAINING medians/modes",
                "reason": "Prevents information leakage from val/test sets.",
            },
            {
                "step": "cleaning",
                "priority": 3,
                "action": "Separate labels (evil, sus) from features before any transformation",
                "reason": "Labels must not be fed as input to the model.",
            },
            {
                "step": "encoding",
                "priority": 4,
                "action": (
                    "Label-encode ALL categorical columns: processName, hostName, eventName "
                    "(strings) AND processId, threadId, parentProcessId, userId, "
                    "mountNamespace, eventId (numeric IDs). Use UNKNOWN handling."
                ),
                "reason": (
                    "These 9 columns are all identifiers. The model should not assume "
                    "ordinal relationships between IDs. Label encoding treats each "
                    "unique value as a distinct category."
                ),
            },
            {
                "step": "feature_engineering",
                "priority": 5,
                "action": (
                    "Create: is_root (userId==0), return_negative (returnValue<0), "
                    "return_category (success/error/info), is_child_of_init "
                    "(parentProcessId==1), is_orphan (parentProcessId==0), "
                    "is_high_args (argsNum > training 95th percentile). "
                    "No arithmetic on categorical ID columns."
                ),
                "reason": (
                    "Domain-informed features from MITRE ATT&CK: privilege escalation "
                    "(T1068), failed syscall probing, process tree analysis (T1059), "
                    "process injection (T1055), and anomalous syscall complexity. "
                    "All features use binary/categorical derivations consistent with "
                    "column classification — no arithmetic on categorical IDs."
                ),
            },
            {
                "step": "scaling",
                "priority": 6,
                "action": (
                    "StandardScaler on TRUE NUMERIC columns ONLY (timestamp, argsNum, "
                    "returnValue + engineered numeric features). Do NOT scale label-encoded "
                    "categorical columns."
                ),
                "reason": (
                    "Scaling categorical encodings distorts them — encoded value 5 becoming "
                    "-0.3 implies a relationship between categories that doesn't exist. "
                    "Only scale columns where arithmetic distance is meaningful."
                ),
            },
            {
                "step": "model",
                "priority": 7,
                "action": "Start with Autoencoder (e.g., input→64→32→16→32→64→input)",
                "reason": (
                    "Zero attacks in training = anomaly detection. Autoencoder learns "
                    "to reconstruct normal patterns; high reconstruction error = anomaly."
                ),
            },
            {
                "step": "evaluation",
                "priority": 8,
                "action": (
                    "Use ROC-AUC and F1 as primary metrics, set threshold using "
                    "validation 95th percentile, validate with sus column"
                ),
                "reason": (
                    "Accuracy is meaningless with 84% attack test set. Threshold tuning "
                    "on validation sus=1 rows provides a sanity check."
                ),
            },
        ]

    # ------------------------------------------------------------------
    # Analyst thinking — educational deep-dive content
    # ------------------------------------------------------------------

    def _analyst_thinking(self, frames: dict[str, pd.DataFrame],
                          stats: dict[str, dict]) -> dict:
        test_evil = stats.get("testing", {}).get("class_distribution", {}).get("evil", {})
        test_attack = test_evil.get(1, 0)
        test_normal = test_evil.get(0, 0)
        test_total = test_attack + test_normal
        attack_pct = round(test_attack / test_total * 100, 1) if test_total else 0

        return {
            "the_core_problem": {
                "title": "Why Can't We Use a Normal Classifier?",
                "summary": (
                    "A normal classifier (cat vs dog) needs lots of examples of BOTH "
                    "classes. Our training data has ZERO attacks — all 763,144 rows are "
                    "normal. This is realistic: in the real world, you have months of "
                    "normal logs and attacks are rare. We must learn what 'normal' looks "
                    "like and flag anything that deviates."
                ),
            },
            "anomaly_detection_approaches": {
                "title": "Industry Approaches When You Have No Attack Examples",
                "approaches": [
                    {
                        "name": "Autoencoder (our choice)",
                        "how": "Learns to compress and reconstruct normal data. Can't "
                               "reconstruct attacks well — high reconstruction error = anomaly.",
                        "used_in": "Cybersecurity, fraud detection. Very common.",
                    },
                    {
                        "name": "Variational Autoencoder (VAE)",
                        "how": "Like autoencoder but learns a probability distribution of "
                               "normal. Attacks fall outside that distribution.",
                        "used_in": "Advanced anomaly detection, generative modeling.",
                    },
                    {
                        "name": "Isolation Forest",
                        "how": "Randomly splits data into partitions. Anomalies are 'easier "
                               "to isolate' (need fewer splits to separate from normal).",
                        "used_in": "Very popular baseline. Works out of the box. Classical ML.",
                    },
                    {
                        "name": "One-Class SVM",
                        "how": "Draws a boundary around normal data in high-dimensional "
                               "space. Anything outside = anomaly.",
                        "used_in": "Network intrusion detection. Classical ML.",
                    },
                    {
                        "name": "GANs (Generative Adversarial Networks)",
                        "how": "Train a generator to produce realistic normal data. A "
                               "discriminator learns to tell real from fake. Attacks look "
                               "'fake' to the discriminator.",
                        "used_in": "Research-heavy, less common in production.",
                    },
                    {
                        "name": "LSTM / Transformer Sequence Models",
                        "how": "Treat events as a time sequence. Learn to predict the next "
                               "event. Attacks break the predicted pattern.",
                        "used_in": "AWS, Azure, Google for real-time log anomaly detection.",
                    },
                    {
                        "name": "Statistical Methods",
                        "how": "Z-scores, percentile thresholds, Mahalanobis distance — "
                               "flag values that are statistically unusual.",
                        "used_in": "Always used as a baseline. Very simple.",
                    },
                    {
                        "name": "Clustering (DBSCAN, k-means)",
                        "how": "Group similar events. Attacks don't fit into any normal cluster.",
                        "used_in": "Good for exploration and initial analysis.",
                    },
                ],
            },
            "autoencoder_explained": {
                "title": "How Our Autoencoder Works — The Music Teacher Analogy",
                "analogy": (
                    "Imagine a music teacher who has ONLY ever heard classical music — "
                    "thousands of pieces by Mozart, Beethoven, Bach. They know the patterns: "
                    "tempo, instruments, structure.\n\n"
                    "Someone plays a song. The teacher's brain tries to 'reconstruct' it "
                    "from classical music knowledge:\n"
                    "• If they play Mozart → brain easily reconstructs it → 'Sounds familiar' "
                    "→ NORMAL\n"
                    "• If they play death metal → brain tries to reconstruct using classical "
                    "patterns → reconstruction sounds nothing like the original → "
                    "'No idea what this is' → ANOMALY"
                ),
                "technical": (
                    "Step 1: COMPRESS — squeeze all features into a small 'summary' "
                    "(e.g., 14 features → 16 numbers).\n"
                    "Step 2: DECOMPRESS — try to rebuild the original features from that "
                    "summary.\n"
                    "Step 3: COMPARE — how different is the rebuild from the original?\n\n"
                    "During training, it only sees normal data, so it gets REALLY good at "
                    "compressing and rebuilding normal patterns. When an attack comes in, "
                    "the compressor doesn't know how to summarize it (never seen this pattern), "
                    "the decompressor builds something 'normal-ish' (that's all it knows), "
                    "and the comparison shows a BIG difference → ANOMALY DETECTED."
                ),
            },
            "cardinality_explained": {
                "title": "What Does 'Cardinality' Mean?",
                "analogy": (
                    "Think of a deck of playing cards:\n"
                    "• The 'suit' column has cardinality 4 (hearts, diamonds, clubs, spades)\n"
                    "• The 'value' column has cardinality 13 (Ace through King)\n"
                    "• A 'card ID' column has cardinality 52 (every card is unique)\n\n"
                    "Cardinality = how many UNIQUE/DISTINCT values a column has."
                ),
                "in_our_data": (
                    "• eventId has cardinality ~50 — only 50 distinct event types. LOW "
                    "cardinality → clearly categorical (like card suits).\n"
                    "• processName has cardinality ~200 — 200 distinct programs. MEDIUM "
                    "cardinality → categorical, needs efficient encoding.\n"
                    "• processId has cardinality ~tens of thousands — HIGH cardinality. "
                    "Still categorical (it's a name tag, not a measurement), but encoding "
                    "must handle many categories.\n\n"
                    "LOW cardinality = easy to encode. HIGH cardinality = needs label encoding "
                    "(not one-hot) and UNKNOWN handling for unseen values."
                ),
            },
            "evaluation_metrics_explained": {
                "title": "Evaluation Metrics — The Security Guard Analogy",
                "scenario": (
                    "Imagine you're a security guard reviewing 100 people entering a "
                    "building. 80 are intruders, 20 are employees."
                ),
                "metrics": {
                    "precision": {
                        "question": "Of everyone I stopped, how many were ACTUAL intruders?",
                        "example": "You stopped 50 people. 45 were intruders, 5 were "
                                   "employees you wrongly accused. Precision = 45/50 = 90%.",
                        "meaning": "High precision = few false accusations. Important when "
                                   "the cost of a false alarm is high (e.g., shutting down "
                                   "a legitimate server).",
                    },
                    "recall": {
                        "question": "Of ALL the actual intruders, how many did I catch?",
                        "example": "There were 80 intruders. You caught 45. 35 slipped "
                                   "through. Recall = 45/80 = 56%.",
                        "meaning": "High recall = few intruders escape. IN CYBERSECURITY, "
                                   "RECALL MATTERS MOST — you'd rather have some false "
                                   "alarms than let attacks through.",
                    },
                    "f1_score": {
                        "question": "What's my balanced grade between precision and recall?",
                        "example": "F1 = 2 × (0.90 × 0.56) / (0.90 + 0.56) = 0.69",
                        "meaning": "F1 punishes you when precision and recall are unbalanced. "
                                   "A model with 99% precision but 1% recall gets a terrible "
                                   "F1 score.",
                    },
                    "roc_auc": {
                        "question": "How good is my detection ability OVERALL, regardless of "
                                    "where I set the alarm threshold?",
                        "example": "If I pick a random intruder and a random employee, "
                                   "what's the probability my system gives the intruder a "
                                   "higher suspicion score? AUC=1.0 → perfect. AUC=0.5 → "
                                   "random coin flip.",
                        "meaning": "AUC measures separation quality across ALL thresholds. "
                                   "It's the single best number for 'is my model learning "
                                   "anything useful?'",
                    },
                },
                "why_not_accuracy": {
                    "explanation": (
                        f"Our test set: {test_attack:,} attacks + {test_normal:,} normal = "
                        f"{test_total:,} total. A dumb model that says 'ATTACK!' for "
                        f"everything would be {attack_pct}% accurate. Looks great — but "
                        "it's completely useless (it caught zero real patterns). "
                        "Precision, recall, F1, and AUC expose this: the dumb model has "
                        f"{attack_pct}% recall but terrible precision, and AUC = 0.5 (random)."
                    ),
                },
            },
            "dl_vs_ml_vs_metrics": {
                "title": "Deep Learning vs Classical ML vs Metrics — They're Different Things",
                "summary": (
                    "Precision, Recall, F1, and ROC-AUC are EVALUATION METRICS — they are "
                    "NOT deep learning or classical ML. They can evaluate ANY model: a neural "
                    "network, a random forest, a simple if-statement, or even a human expert."
                ),
                "categories": {
                    "metrics": {
                        "what": "How you MEASURE performance",
                        "examples": "Precision, Recall, F1, ROC-AUC, Accuracy",
                        "analogy": "The exam scoring rubric. It doesn't care if the student "
                                   "is a PhD or a high schooler.",
                    },
                    "classical_ml": {
                        "what": "Algorithms with handcrafted math (no neural networks)",
                        "examples": "Isolation Forest, SVM, Random Forest, k-means, "
                                    "Logistic Regression",
                        "analogy": "YOU (the human) decide what patterns to look for. "
                                   "You engineer features, pick a distance metric. "
                                   "The model applies your rules.",
                    },
                    "deep_learning": {
                        "what": "Neural networks that learn features automatically",
                        "examples": "Autoencoder, VAE, LSTM, Transformer, GAN, CNN",
                        "analogy": "The model discovers patterns on its own from raw data. "
                                   "You design the architecture; it learns the features.",
                    },
                },
                "our_project": (
                    "We use a DEEP LEARNING model (autoencoder) evaluated with "
                    "MODEL-AGNOSTIC metrics (F1, AUC), processing data with CLASSICAL ML "
                    "tools (StandardScaler, LabelEncoder from scikit-learn). Real-world "
                    "systems almost always mix both."
                ),
            },
            "args_column_analysis": self._args_column_deep_dive(frames),
            "analyst_perspective": {
                "title": "Data Analyst's Synthesis",
                "thinking": (
                    "Looking at this dataset as a cybersecurity analyst, here is what "
                    "stands out:\n\n"
                    "1. THE DATA TELLS A STORY: Training data is a peaceful honeypot — "
                    "normal server operations. Test data is the honeypot UNDER ATTACK. "
                    "The validation set is a clean control group. This three-way split "
                    "is intentional and well-designed.\n\n"
                    "2. THE TRAP: 6 columns LOOK numeric but ARE categorical identifiers. "
                    "A naive analyst would normalize processId and mountNamespace, teaching "
                    "the model that 'PID 30000 is 30x more important than PID 1000'. "
                    "This is wrong. PID 1 (init) is arguably the most important process.\n\n"
                    "3. THE HIDDEN GOLD MINE: The args column was almost dropped as 'too "
                    "complex'. But a deeper look revealed it contains the strongest attack "
                    "signal in the dataset — a 90x difference in /proc/ access patterns "
                    "between normal and attack traffic. Always look at the data before "
                    "discarding columns.\n\n"
                    "4. THE SIGNAL IS IN THE COMBINATIONS: No single column screams 'attack'. "
                    "Attacks manifest as unusual COMBINATIONS — a rare process running as "
                    "root at an unusual time with unusual arguments. The autoencoder's power "
                    "is learning these multi-dimensional patterns.\n\n"
                    "5. THE WEAK LABELS ARE GOLD: 1,269 'suspicious' rows in training "
                    "(sus=1) are borderline normal. If the autoencoder assigns them higher "
                    "reconstruction error, it's finding real signal. Use this for threshold "
                    "calibration.\n\n"
                    "6. THE REAL-WORLD LESSON: In production, you'd retrain periodically "
                    "as 'normal' evolves (new services, new processes). The model's concept "
                    "of normal must evolve too. This is called concept drift."
                ),
            },
        }

    # ------------------------------------------------------------------
    # Args column deep-dive analysis
    # ------------------------------------------------------------------

    @staticmethod
    def _safe_separation(train_pct: float, attack_pct: float) -> str:
        if attack_pct > 0 and train_pct > 0:
            return f"{round(train_pct / attack_pct, 1)}x"
        if train_pct > 0 and attack_pct == 0:
            return "inf"
        return "~"

    def _args_column_deep_dive(self, frames: dict[str, pd.DataFrame]) -> dict:
        """Analyze the args column to prove it contains strong attack signal."""
        import ast

        def compute_signals(df: pd.DataFrame, sample_size: int = 50000) -> dict:
            sample = df.sample(min(sample_size, len(df)), random_state=42)
            total = len(sample)
            counts = {
                "touches_proc": 0, "touches_etc": 0, "touches_tmp": 0,
                "has_write_flag": 0, "is_hidden_path": 0, "has_pathname": 0,
            }
            example_paths: list[str] = []
            for args_str in sample["args"]:
                try:
                    args_list = ast.literal_eval(args_str)
                except (ValueError, SyntaxError):
                    continue
                for a in args_list:
                    name = a.get("name", "")
                    val = str(a.get("value", ""))
                    if name in ("pathname", "filename"):
                        counts["has_pathname"] += 1
                        if "/proc/" in val:
                            counts["touches_proc"] += 1
                        if "/etc/" in val:
                            counts["touches_etc"] += 1
                        if "/tmp/" in val:
                            counts["touches_tmp"] += 1
                        if "/." in val:
                            counts["is_hidden_path"] += 1
                            if len(example_paths) < 5:
                                example_paths.append(val)
                    if name == "flags" and ("WRONLY" in val or "RDWR" in val or "CREAT" in val):
                        counts["has_write_flag"] += 1

            pcts = {k: round(v / total * 100, 2) for k, v in counts.items()}
            return {"counts": counts, "pcts": pcts, "sample_size": total,
                    "hidden_path_examples": example_paths}

        result: dict = {
            "title": "The Args Column — Why We Almost Made a Big Mistake",
            "story": (
                "Initial instinct: drop the args column — it's a messy JSON-like string, "
                "too complex to parse. But a data analyst's job is to LOOK at the data "
                "before making decisions. When we parsed the file paths and flags buried "
                "inside args, we found the STRONGEST attack signal in the entire dataset."
            ),
        }

        split_signals: dict[str, dict] = {}
        if "training" in frames and "args" in frames["training"].columns:
            split_signals["training_normal"] = compute_signals(frames["training"])

        if "testing" in frames and "args" in frames["testing"].columns:
            test = frames["testing"]
            if "evil" in test.columns:
                attacks = test[test["evil"] == 1]
                normals = test[test["evil"] == 0]
                split_signals["test_attacks"] = compute_signals(attacks)
                split_signals["test_normal"] = compute_signals(normals)

        result["split_signals"] = split_signals

        train_proc = split_signals.get("training_normal", {}).get("pcts", {}).get("touches_proc", 0)
        attack_proc = split_signals.get("test_attacks", {}).get("pcts", {}).get("touches_proc", 0)

        train_write = split_signals.get("training_normal", {}).get("pcts", {}).get("has_write_flag", 0)
        attack_write = split_signals.get("test_attacks", {}).get("pcts", {}).get("has_write_flag", 0)

        result["key_findings"] = [
            {
                "signal": "args_touches_proc",
                "description": "Does the syscall access /proc/ (process info filesystem)?",
                "training_pct": train_proc,
                "attack_pct": attack_proc,
                "separation": self._safe_separation(train_proc, attack_proc),
                "analyst_thinking": (
                    "Normal servers CONSTANTLY read /proc/ — checking CPU usage, memory, "
                    "process lists, health monitoring. It's like a doctor checking vital signs "
                    "every minute. Attackers DON'T do this — they're busy executing commands, "
                    "stealing data, installing backdoors. The ABSENCE of /proc/ access is "
                    "itself a powerful signal that something unusual is happening."
                ),
            },
            {
                "signal": "args_has_write_flag",
                "description": "Does the syscall open files for writing (O_WRONLY, O_RDWR, O_CREAT)?",
                "training_pct": train_write,
                "attack_pct": attack_write,
                "separation": self._safe_separation(train_write, attack_write),
                "analyst_thinking": (
                    "Normal operations involve regular log writes, temp files, config updates. "
                    "Attacks in this dataset are focused on READING and EXFILTRATING — they're "
                    "in reconnaissance/data-theft mode, not file-creation mode. Fewer writes "
                    "= suspicious behavior."
                ),
            },
            {
                "signal": "args_touches_etc",
                "description": "Does the syscall access /etc/ (system config files)?",
                "training_pct": split_signals.get("training_normal", {}).get("pcts", {}).get("touches_etc", 0),
                "attack_pct": split_signals.get("test_attacks", {}).get("pcts", {}).get("touches_etc", 0),
                "separation": self._safe_separation(
                    split_signals.get("training_normal", {}).get("pcts", {}).get("touches_etc", 0),
                    split_signals.get("test_attacks", {}).get("pcts", {}).get("touches_etc", 0),
                ),
                "analyst_thinking": (
                    "Normal services read their config files on startup (/etc/nginx.conf, "
                    "/etc/ssh/sshd_config). Attacks may also read /etc/passwd or /etc/shadow "
                    "to steal credentials, but the PATTERN of access is different — targeted "
                    "reads vs routine config loading."
                ),
            },
            {
                "signal": "args_is_hidden_path",
                "description": "Does the pathname contain '/.' (hidden directory)?",
                "training_pct": split_signals.get("training_normal", {}).get("pcts", {}).get("is_hidden_path", 0),
                "attack_pct": split_signals.get("test_attacks", {}).get("pcts", {}).get("is_hidden_path", 0),
                "separation": self._safe_separation(
                    split_signals.get("training_normal", {}).get("pcts", {}).get("is_hidden_path", 0),
                    split_signals.get("test_attacks", {}).get("pcts", {}).get("is_hidden_path", 0),
                ),
                "analyst_thinking": (
                    "Hidden directories (starting with .) are a classic attacker technique "
                    "(MITRE T1564.001). Malware staging in /tmp/.X25-unix/.rsync/ or similar "
                    "paths is common. Low frequency but HIGH specificity — when it appears, "
                    "it's almost certainly malicious."
                ),
                "real_examples": split_signals.get("test_attacks", {}).get("hidden_path_examples", []),
            },
        ]

        result["lesson"] = (
            "NEVER drop a column just because it looks complex. A 5-minute parsing exercise "
            "revealed the strongest signal in the dataset. The analyst's rule: if a column "
            "contains domain-relevant information (file paths, network addresses, command "
            "arguments), ALWAYS inspect it before discarding. Extract simple binary features "
            "from complex strings — you don't need to parse everything, just the signals "
            "that matter."
        )

        result["features_to_extract"] = [
            {"name": "args_touches_proc", "formula": "'/proc/' in pathname", "type": "binary"},
            {"name": "args_touches_etc", "formula": "'/etc/' in pathname", "type": "binary"},
            {"name": "args_has_write_flag", "formula": "O_WRONLY|O_RDWR|O_CREAT in flags", "type": "binary"},
            {"name": "args_is_hidden_path", "formula": "'/.' in pathname", "type": "binary"},
            {"name": "args_has_pathname", "formula": "any pathname arg exists", "type": "binary"},
        ]

        return result

    # ------------------------------------------------------------------
    # HTML report rendering
    # ------------------------------------------------------------------

    def _render_html(self, report: dict, frames: dict[str, pd.DataFrame]) -> str:
        insights = report.get("analyst_insights", [])
        recs = report.get("recommendations", [])
        splits = report.get("splits", {})
        cross = report.get("cross_split_analysis", {})
        col_class = report.get("column_classification", {})
        col_class_stats = report.get("column_classification_stats", {})
        thinking = report.get("analyst_thinking", {})

        severity_colors = {
            "critical": "#dc3545", "warning": "#ffc107", "info": "#17a2b8",
        }
        severity_bg = {
            "critical": "#fff5f5", "warning": "#fff8e1", "info": "#e8f4f8",
        }

        # --- Split summary rows ---
        split_rows = ""
        for sname, sstat in splits.items():
            evil = sstat["class_distribution"].get("evil", {})
            sus = sstat["class_distribution"].get("sus", {})
            split_rows += f"""
            <tr>
                <td><strong>{sname.title()}</strong></td>
                <td>{sstat['row_count']:,}</td>
                <td>{sstat['column_count']}</td>
                <td>{evil.get(0, 0):,}</td>
                <td style="color:{'#dc3545' if evil.get(1,0) > 0 else '#28a745'};
                    font-weight:bold">{evil.get(1, 0):,}</td>
                <td>{sus.get(1, 0):,}</td>
                <td>{sstat.get('duplicate_rows', 0):,}</td>
                <td>{sstat.get('memory_mb', 0)} MB</td>
            </tr>"""

        # --- Column classification table ---
        col_class_rows = ""
        type_colors = {
            "true_numeric": "#28a745",
            "categorical_id": "#dc3545",
            "string_categorical": "#6f42c1",
            "complex_drop": "#6c757d",
            "labels": "#fd7e14",
        }
        type_labels = {
            "true_numeric": "True Numeric",
            "categorical_id": "Categorical ID (looks numeric!)",
            "string_categorical": "String Categorical",
            "complex_drop": "Complex — Drop",
            "labels": "Labels (not features)",
        }
        for ctype, info in col_class.items():
            color = type_colors.get(ctype, "#333")
            label = type_labels.get(ctype, ctype)
            cols_str = ", ".join(f"<code>{c}</code>" for c in info["columns"])
            col_class_rows += f"""
            <tr>
                <td><span style="background:{color};color:white;padding:2px 8px;
                    border-radius:3px;font-size:0.8em">{label}</span></td>
                <td>{cols_str}</td>
                <td style="font-size:0.9em">{info['treatment']}</td>
            </tr>"""

        # --- Categorical ID evidence table ---
        id_evidence_rows = ""
        for col, cstat in col_class_stats.items():
            id_evidence_rows += f"""
            <tr>
                <td><code>{col}</code></td>
                <td>{cstat['unique_values']:,}</td>
                <td>{cstat['cardinality_ratio']}%</td>
                <td>{cstat['top_10_coverage_pct']}%</td>
                <td style="font-size:0.85em">{cstat['why_categorical']}</td>
            </tr>"""

        # --- True numeric stats table ---
        numeric_html = ""
        if "training" in splits:
            nstats = splits["training"].get("true_numeric_stats", {})
            for col, st in nstats.items():
                numeric_html += f"""
                <tr>
                    <td>{col}</td>
                    <td>{st['min']:,.2f}</td>
                    <td>{st['max']:,.2f}</td>
                    <td>{st['mean']:,.4f}</td>
                    <td>{st['std']:,.4f}</td>
                    <td>{st['median']:,.2f}</td>
                    <td>{st['zeros_pct']}%</td>
                    <td>{st['negative_pct']}%</td>
                </tr>"""

        # --- String categorical stats ---
        cat_html = ""
        if "training" in splits:
            cstats = splits["training"].get("string_categorical_stats", {})
            for col, st in cstats.items():
                top5 = ", ".join(f"{k} ({v:,})" for k, v in list(st["top_10"].items())[:5])
                cat_html += f"""
                <tr>
                    <td>{col}</td>
                    <td>{st['unique_count']:,}</td>
                    <td style="font-size:0.85em">{top5}</td>
                </tr>"""

        # --- Insight cards ---
        insight_cards = ""
        for ins in insights:
            sev = ins.get("severity", "info")
            insight_cards += f"""
            <div style="border-left:4px solid {severity_colors.get(sev, '#666')};
                        background:{severity_bg.get(sev, '#f9f9f9')};
                        padding:12px 16px;margin:10px 0;border-radius:4px">
                <div style="display:flex;align-items:center;gap:8px;margin-bottom:4px">
                    <span style="background:{severity_colors.get(sev, '#666')};
                                 color:white;padding:2px 8px;border-radius:3px;
                                 font-size:0.75em;text-transform:uppercase">{sev}</span>
                    <strong>{ins['title']}</strong>
                </div>
                <p style="margin:4px 0 0 0;color:#333">{ins['detail']}</p>
            </div>"""

        # --- Recommendation rows ---
        rec_rows = ""
        for rec in sorted(recs, key=lambda r: r["priority"]):
            rec_rows += f"""
            <tr>
                <td>{rec['priority']}</td>
                <td><code>{rec['step']}</code></td>
                <td>{rec['action']}</td>
                <td style="font-size:0.9em;color:#555">{rec['reason']}</td>
            </tr>"""

        # --- Drift table ---
        drift_html = ""
        drift_data = cross.get("distribution_drift", {})
        drifted_items = [(k, v) for k, v in drift_data.items() if v.get("significant")]
        if drifted_items:
            drift_rows = ""
            for key, d in sorted(drifted_items, key=lambda x: -x[1]["z_shift"]):
                drift_rows += f"""
                <tr>
                    <td>{key}</td>
                    <td>{d['train_mean']}</td>
                    <td>{d['other_mean']}</td>
                    <td style="color:#dc3545;font-weight:bold">{d['z_shift']}</td>
                </tr>"""
            drift_html = f"""
            <h2>Distribution Drift — True Numeric Only (Train &rarr; Test/Val)</h2>
            <p>Features where the mean shifted more than 2 standard deviations.
               Only computed for truly numeric columns (not ID columns).</p>
            <table>
                <tr><th>Feature_Split</th><th>Train Mean</th><th>Other Mean</th>
                    <th>Z-Shift</th></tr>
                {drift_rows}
            </table>"""

        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>BETH Dataset — Cybersecurity Data Analysis Report {REPORT_VERSION}</title>
<style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto,
                         sans-serif; max-width: 1200px; margin: 0 auto;
                         padding: 20px; color: #1a1a1a; background: #fafafa; }}
    h1 {{ border-bottom: 3px solid #2c3e50; padding-bottom: 10px; }}
    h2 {{ color: #2c3e50; margin-top: 30px; border-bottom: 1px solid #ddd;
          padding-bottom: 6px; }}
    table {{ border-collapse: collapse; width: 100%; margin: 10px 0; }}
    th, td {{ border: 1px solid #ddd; padding: 8px 12px; text-align: left; }}
    th {{ background: #2c3e50; color: white; font-weight: 600; }}
    tr:nth-child(even) {{ background: #f2f2f2; }}
    code {{ background: #e8e8e8; padding: 2px 6px; border-radius: 3px;
            font-size: 0.9em; }}
    .banner {{ background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
               color: white; padding: 24px; border-radius: 8px; margin-bottom: 24px; }}
    .banner h1 {{ border: none; margin: 0; padding: 0; color: white; }}
    .banner p {{ margin: 8px 0 0 0; opacity: 0.85; }}
    .summary-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                     gap: 12px; margin: 16px 0; }}
    .summary-card {{ background: white; border-radius: 8px; padding: 16px;
                     box-shadow: 0 1px 3px rgba(0,0,0,0.1); text-align: center; }}
    .summary-card .number {{ font-size: 2em; font-weight: bold; color: #2c3e50; }}
    .summary-card .label {{ font-size: 0.85em; color: #777; margin-top: 4px; }}
    .callout {{ background: #fff3cd; border: 1px solid #ffc107; border-radius: 6px;
                padding: 16px; margin: 16px 0; }}
    .callout strong {{ color: #856404; }}
    h3 {{ color: #34495e; margin-top: 20px; }}
    .edu-section {{ background: white; border-radius: 8px; padding: 20px 24px;
                    box-shadow: 0 1px 4px rgba(0,0,0,0.08); margin: 20px 0; }}
    .edu-section h2 {{ margin-top: 10px; }}
    .analogy-box {{ background: #e8f5e9; border-left: 4px solid #28a745;
                    padding: 14px 18px; margin: 12px 0; border-radius: 4px;
                    white-space: pre-line; line-height: 1.6; }}
    .technical-box {{ background: #e3f2fd; border-left: 4px solid #1976d2;
                      padding: 14px 18px; margin: 12px 0; border-radius: 4px;
                      white-space: pre-line; line-height: 1.6; }}
    .thinking-box {{ background: #fce4ec; border-left: 4px solid #c62828;
                     padding: 14px 18px; margin: 12px 0; border-radius: 4px;
                     white-space: pre-line; line-height: 1.6; }}
    .metric-card {{ background: #f5f5f5; border-radius: 6px; padding: 14px 18px;
                    margin: 10px 0; border: 1px solid #e0e0e0; }}
    .metric-card .q {{ font-weight: bold; color: #1565c0; margin-bottom: 4px; }}
    .metric-card .ex {{ color: #555; margin: 4px 0; }}
    .metric-card .meaning {{ color: #2e7d32; font-style: italic; }}
    .approach-grid {{ display: grid; grid-template-columns: 1fr 1fr;
                      gap: 12px; margin: 12px 0; }}
    .approach-card {{ background: #f8f9fa; border: 1px solid #dee2e6;
                      border-radius: 6px; padding: 14px; }}
    .approach-card strong {{ color: #2c3e50; }}
    .approach-card .used {{ font-size: 0.85em; color: #6c757d;
                            margin-top: 6px; font-style: italic; }}
    .cat-grid {{ display: grid; grid-template-columns: repeat(3, 1fr);
                 gap: 12px; margin: 12px 0; }}
    .cat-card {{ border-radius: 6px; padding: 14px; }}
    .cat-card h4 {{ margin: 0 0 6px 0; }}
    .cat-card .examples {{ font-size: 0.9em; color: #555; }}
    .divider {{ border: none; border-top: 3px solid #2c3e50; margin: 40px 0; }}
    @media (max-width: 768px) {{
        .approach-grid {{ grid-template-columns: 1fr; }}
        .cat-grid {{ grid-template-columns: 1fr; }}
    }}
</style>
</head>
<body>

<div class="banner">
    <h1>BETH Dataset — Cybersecurity Data Analysis Report</h1>
    <p>Version: {REPORT_VERSION} &nbsp;|&nbsp;
       Generated: {report.get('generated_at', 'N/A')} &nbsp;|&nbsp;
       Biased Evaluation of Traces from Honeypots</p>
</div>

<div class="summary-grid">
    <div class="summary-card">
        <div class="number">{sum(s['row_count'] for s in splits.values()):,}</div>
        <div class="label">Total Events</div>
    </div>
    <div class="summary-card">
        <div class="number">{splits.get('training',{}).get('column_count',0)}</div>
        <div class="label">Columns</div>
    </div>
    <div class="summary-card">
        <div class="number">3</div>
        <div class="label">True Numeric</div>
    </div>
    <div class="summary-card">
        <div class="number" style="color:#dc3545">6</div>
        <div class="label">Categorical IDs (look numeric!)</div>
    </div>
    <div class="summary-card">
        <div class="number" style="color:#dc3545">
            {splits.get('testing',{}).get('class_distribution',{}).get('evil',{}).get(1,0):,}
        </div>
        <div class="label">Attack Events (Test Only)</div>
    </div>
</div>

<h2>Dataset Overview</h2>
<table>
    <tr>
        <th>Split</th><th>Rows</th><th>Columns</th>
        <th>Normal</th><th>Attacks</th>
        <th>Suspicious</th><th>Duplicates</th><th>Memory</th>
    </tr>
    {split_rows}
</table>

<h2>Column Classification</h2>
<div class="callout">
    <strong>Key Finding:</strong> 6 of the 16 columns are stored as integers but are
    actually <strong>categorical identifiers</strong>. Treating them as numeric values
    would teach the model false ordinal relationships (e.g., "PID 500 &gt; PID 250"
    or "userId 100 &gt; userId 0"). Only 3 columns are truly numeric.
</div>
<table>
    <tr><th>Type</th><th>Columns</th><th>Treatment</th></tr>
    {col_class_rows}
</table>

<h2>Evidence: Why ID Columns Are Categorical</h2>
<table>
    <tr><th>Column</th><th>Unique Values</th><th>Cardinality %</th>
        <th>Top 10 Coverage</th><th>Analysis</th></tr>
    {id_evidence_rows}
</table>

<h2>Analyst Insights</h2>
<p style="color:#555">Findings from the perspective of a cybersecurity data analyst.</p>
{insight_cards}

<h2>True Numeric Feature Statistics (Training Set)</h2>
<p style="color:#555">Only columns where arithmetic relationships are meaningful.</p>
<table>
    <tr><th>Column</th><th>Min</th><th>Max</th><th>Mean</th><th>Std</th>
        <th>Median</th><th>Zeros %</th><th>Negative %</th></tr>
    {numeric_html}
</table>

<h2>String Categorical Statistics (Training Set)</h2>
<table>
    <tr><th>Column</th><th>Unique Values</th><th>Top 5</th></tr>
    {cat_html}
</table>

{drift_html}

<h2>Recommended Pipeline Steps</h2>
<table>
    <tr><th>#</th><th>Step</th><th>Action</th><th>Rationale</th></tr>
    {rec_rows}
</table>

<hr class="divider">

{self._render_educational_html(thinking)}

<footer style="margin-top:40px;padding-top:16px;border-top:1px solid #ddd;
               color:#999;font-size:0.85em">
    BETH Dataset Analysis {REPORT_VERSION} &nbsp;|&nbsp;
    Cyber Attack Detection Pipeline &nbsp;|&nbsp;
    Report generated programmatically by the data_analysis task
</footer>

</body>
</html>"""
        return html

    # ------------------------------------------------------------------
    # Educational HTML sections
    # ------------------------------------------------------------------

    @staticmethod
    def _render_educational_html(thinking: dict) -> str:
        core = thinking.get("the_core_problem", {})
        approaches = thinking.get("anomaly_detection_approaches", {})
        autoenc = thinking.get("autoencoder_explained", {})
        cardinality = thinking.get("cardinality_explained", {})
        metrics = thinking.get("evaluation_metrics_explained", {})
        dl_vs_ml = thinking.get("dl_vs_ml_vs_metrics", {})
        args_analysis = thinking.get("args_column_analysis", {})
        perspective = thinking.get("analyst_perspective", {})

        approach_cards = ""
        for a in approaches.get("approaches", []):
            approach_cards += f"""
            <div class="approach-card">
                <strong>{a['name']}</strong>
                <p style="margin:6px 0 0 0">{a['how']}</p>
                <div class="used">Used in: {a['used_in']}</div>
            </div>"""

        metric_cards = ""
        for mname, minfo in metrics.get("metrics", {}).items():
            display = mname.replace("_", " ").upper()
            if mname == "f1_score":
                display = "F1 SCORE"
            elif mname == "roc_auc":
                display = "ROC-AUC"
            metric_cards += f"""
            <div class="metric-card">
                <h3 style="margin:0 0 8px 0;color:#1565c0">{display}</h3>
                <div class="q">The question it answers: "{minfo['question']}"</div>
                <div class="ex"><strong>Example:</strong> {minfo['example']}</div>
                <div class="meaning"><strong>What it means:</strong> {minfo['meaning']}</div>
            </div>"""

        cat_items = dl_vs_ml.get("categories", {})
        cat_cards = ""
        cat_colors = {"metrics": "#e8f5e9", "classical_ml": "#fff3e0",
                       "deep_learning": "#e3f2fd"}
        cat_borders = {"metrics": "#2e7d32", "classical_ml": "#e65100",
                        "deep_learning": "#1565c0"}
        cat_labels = {"metrics": "Evaluation Metrics",
                       "classical_ml": "Classical ML",
                       "deep_learning": "Deep Learning"}
        for ckey, cinfo in cat_items.items():
            bg = cat_colors.get(ckey, "#f5f5f5")
            bdr = cat_borders.get(ckey, "#999")
            lbl = cat_labels.get(ckey, ckey)
            cat_cards += f"""
            <div class="cat-card" style="background:{bg};border-left:4px solid {bdr}">
                <h4>{lbl}</h4>
                <p style="margin:4px 0"><strong>What:</strong> {cinfo['what']}</p>
                <div class="examples"><strong>Examples:</strong> {cinfo['examples']}</div>
                <p style="margin:6px 0 0 0;font-style:italic;color:#555">
                    {cinfo['analogy']}</p>
            </div>"""

        return f"""
<div class="edu-section">
    <h2 style="color:#c62828">Data Analyst's Deep Dive — What I Think About This Data</h2>
    <p style="font-size:1.05em;color:#555">
        Below is the full cognitive walkthrough a cybersecurity data analyst goes through
        when encountering this dataset — from the core problem to model selection,
        evaluation strategy, and synthesis.
    </p>
</div>

<div class="edu-section">
    <h2>{core.get('title', '')}</h2>
    <div class="thinking-box">{core.get('summary', '')}</div>
</div>

<div class="edu-section">
    <h2>{approaches.get('title', '')}</h2>
    <p>When training data has no examples of the "bad" class, industry practitioners
       reach for these 8 approaches. Each has trade-offs — there's no single winner.</p>
    <div class="approach-grid">
        {approach_cards}
    </div>
    <div class="callout">
        <strong>Why Autoencoder for this project?</strong> It's the sweet spot between
        simplicity and power. Isolation Forest is simpler but doesn't learn feature
        interactions. LSTMs are more powerful but need sequential ordering that our
        per-event data doesn't naturally have. Autoencoders work directly on tabular
        features and scale well.
    </div>
</div>

<div class="edu-section">
    <h2>{autoenc.get('title', '')}</h2>
    <h3>The Analogy</h3>
    <div class="analogy-box">{autoenc.get('analogy', '')}</div>
    <h3>The Technical Reality</h3>
    <div class="technical-box">{autoenc.get('technical', '')}</div>
</div>

<div class="edu-section">
    <h2>{cardinality.get('title', '')}</h2>
    <h3>The Analogy</h3>
    <div class="analogy-box">{cardinality.get('analogy', '')}</div>
    <h3>In Our BETH Dataset</h3>
    <div class="technical-box">{cardinality.get('in_our_data', '')}</div>
</div>

<div class="edu-section">
    <h2>{metrics.get('title', '')}</h2>
    <p><strong>The Scenario:</strong> {metrics.get('scenario', '')}</p>

    {metric_cards}

    <h3>Why Not Just Use Accuracy?</h3>
    <div class="thinking-box">
        {metrics.get('why_not_accuracy', {}).get('explanation', '')}
    </div>
</div>

<div class="edu-section">
    <h2>{dl_vs_ml.get('title', '')}</h2>
    <div class="callout">
        <strong>Key Point:</strong> {dl_vs_ml.get('summary', '')}
    </div>
    <div class="cat-grid">
        {cat_cards}
    </div>
    <div class="technical-box">
        <strong>In Our Project:</strong> {dl_vs_ml.get('our_project', '')}
    </div>
</div>

{DataAnalysis._render_args_section_html(args_analysis)}

<div class="edu-section">
    <h2>{perspective.get('title', '')}</h2>
    <div class="thinking-box" style="font-size:1.02em;line-height:1.7">
        {perspective.get('thinking', '')}
    </div>
</div>
"""


    @staticmethod
    def _render_args_section_html(args: dict) -> str:
        if not args:
            return ""

        findings = args.get("key_findings", [])
        features = args.get("features_to_extract", [])

        finding_cards = ""
        for f in findings:
            train_pct = f.get("training_pct", 0)
            attack_pct = f.get("attack_pct", 0)
            sep = f.get("separation", "?")

            try:
                sep_val = float(str(sep).replace("x", "").replace("inf", "999"))
            except (ValueError, TypeError):
                sep_val = 0
            bar_color = "#c62828" if sep_val > 5 else "#e65100"
            train_bar_w = min(train_pct * 3, 100)
            attack_bar_w = min(attack_pct * 3, 100)

            examples_html = ""
            if f.get("real_examples"):
                examples_html = (
                    '<div style="margin-top:8px;font-size:0.85em;color:#666">'
                    '<strong>Real examples found:</strong> '
                    + ", ".join(f"<code>{e}</code>" for e in f["real_examples"])
                    + '</div>'
                )

            finding_cards += f"""
            <div style="background:#fff;border:1px solid #e0e0e0;border-radius:8px;
                        padding:16px;margin-bottom:16px;box-shadow:0 1px 3px rgba(0,0,0,0.08)">
                <div style="display:flex;justify-content:space-between;align-items:center;
                            margin-bottom:8px">
                    <h4 style="margin:0;color:#1565c0;font-family:monospace;font-size:1.05em">
                        {f.get('signal', '')}
                    </h4>
                    <span style="background:{bar_color};color:white;padding:3px 10px;
                                 border-radius:12px;font-weight:bold;font-size:0.9em">
                        {sep} separation
                    </span>
                </div>
                <p style="margin:0 0 10px 0;color:#555">{f.get('description', '')}</p>

                <div style="display:flex;gap:20px;margin-bottom:10px">
                    <div style="flex:1">
                        <div style="font-size:0.82em;color:#888;margin-bottom:3px">
                            Training (normal)
                        </div>
                        <div style="background:#e8f5e9;border-radius:4px;height:20px;
                                    position:relative;overflow:hidden">
                            <div style="background:#2e7d32;height:100%;
                                        width:{train_bar_w}%;border-radius:4px"></div>
                        </div>
                        <div style="font-size:0.85em;font-weight:bold;color:#2e7d32;
                                    margin-top:2px">{train_pct}%</div>
                    </div>
                    <div style="flex:1">
                        <div style="font-size:0.82em;color:#888;margin-bottom:3px">
                            Test (attacks)
                        </div>
                        <div style="background:#ffebee;border-radius:4px;height:20px;
                                    position:relative;overflow:hidden">
                            <div style="background:#c62828;height:100%;
                                        width:{attack_bar_w}%;border-radius:4px"></div>
                        </div>
                        <div style="font-size:0.85em;font-weight:bold;color:#c62828;
                                    margin-top:2px">{attack_pct}%</div>
                    </div>
                </div>

                <div style="background:#f3e5f5;border-left:3px solid #7b1fa2;padding:10px 14px;
                            border-radius:0 6px 6px 0;font-style:italic;color:#4a148c;
                            font-size:0.92em;line-height:1.5">
                    <strong>Analyst thinking:</strong> {f.get('analyst_thinking', '')}
                </div>
                {examples_html}
            </div>"""

        feature_rows = ""
        for ft in features:
            feature_rows += f"""
            <tr>
                <td style="padding:8px 12px;font-family:monospace;font-weight:bold;
                           color:#1565c0">{ft['name']}</td>
                <td style="padding:8px 12px"><code>{ft['formula']}</code></td>
                <td style="padding:8px 12px;text-align:center">
                    <span style="background:#e8f5e9;color:#2e7d32;padding:2px 8px;
                                 border-radius:10px;font-size:0.85em">{ft['type']}</span>
                </td>
            </tr>"""

        return f"""
<div class="edu-section" style="border-top:3px solid #c62828;padding-top:20px">
    <h2 style="color:#c62828">
        {args.get('title', 'Args Column Analysis')}
    </h2>

    <div class="callout" style="background:#fff3e0;border-left:4px solid #e65100;
                                padding:16px;border-radius:0 8px 8px 0;margin-bottom:20px">
        <strong style="color:#e65100">The Story:</strong>
        <p style="margin:8px 0 0 0;line-height:1.6">{args.get('story', '')}</p>
    </div>

    <h3 style="margin-top:24px">Signal-by-Signal Breakdown</h3>
    <p style="color:#666;margin-bottom:16px">
        Each card shows what percentage of syscalls in training (normal) vs test (attacks)
        exhibit a particular behaviour. The bigger the gap, the stronger the signal.
    </p>

    {finding_cards}

    <div class="callout" style="background:#e8f5e9;border-left:4px solid #2e7d32;
                                padding:16px;border-radius:0 8px 8px 0;margin:24px 0">
        <strong style="color:#2e7d32">The Lesson:</strong>
        <p style="margin:8px 0 0 0;line-height:1.6">{args.get('lesson', '')}</p>
    </div>

    <h3>Features We Will Extract from args</h3>
    <table style="width:100%;border-collapse:collapse;margin-top:10px">
        <thead>
            <tr style="background:#1565c0;color:white">
                <th style="padding:10px 12px;text-align:left">Feature Name</th>
                <th style="padding:10px 12px;text-align:left">Formula</th>
                <th style="padding:10px 12px;text-align:center">Type</th>
            </tr>
        </thead>
        <tbody style="background:#fff">
            {feature_rows}
        </tbody>
    </table>
</div>
"""


Task = DataAnalysis
