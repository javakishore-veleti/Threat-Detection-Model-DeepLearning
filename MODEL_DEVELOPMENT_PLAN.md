# Cyber Attack Detection — Model Development Plan

> **Goal:** Build an anomaly-based cyber attack detection system using PyTorch. Train an autoencoder on normal host-level event traces (BETH dataset) to learn baseline behavior, then flag deviations as potential intrusions. The system should preprocess raw telemetry, train a neural network model, evaluate detection quality with industry-standard metrics, and lay the groundwork for real-time anomaly scoring.

---

## Current State of the Codebase

| Layer | Status | What Exists |
|---|---|---|
| **Config** | DONE | `configs/cyber_attack_detection/default.yaml` — all dataset-specific constants externalized. `core/config.py` — YAML loader with dot-path accessor, split key derivation, column helpers |
| **Framework** | DONE | `src/main.py` facade (loads config into `req.config`), `core/common/wfs/` (interfaces, DTOs with `req.df_keys`/`req.label_keys` properties), `core/logger.py`, all 5 sub-workflow facades, `package.json` commands |
| **Download** | DONE | `kaggle_beth.py` — idempotent Kaggle download, reads dataset ID and paths from config |
| **Preprocessing** | 4 of 5 done | `data_analysis.py` (v04 report, config-driven), `cleaning.py` (config-driven), `feature_engineering.py` (generic engine reads feature specs from YAML), `encoding.py` (LabelEncoder with UNKNOWN handling) |
| **Models** | Empty | Facade ready, `tasks/` empty |
| **Training** | Empty | Facade ready, `tasks/` empty |
| **Inference** | Empty | Facade ready, `tasks/` empty |

## Config-Driven Architecture

All dataset-specific constants live in `configs/<pipeline>/default.yaml`. Tasks read from
`req.config` instead of hardcoding values. To adapt this pipeline for a new dataset:

1. Copy `configs/cyber_attack_detection/default.yaml` to `configs/<new_pipeline>/default.yaml`
2. Update dataset name, source, paths, splits, column classification, feature definitions
3. Zero Python code changes required

**Key config sections:**

| Section | What It Controls |
|---|---|
| `dataset` | Name, subtitle, source (Kaggle ID, URL), paths (data_dir, report_dir) |
| `splits` | Map of split name → CSV filename. Key order determines train/val/test assignment |
| `split_prefixes` | Map of split name → short prefix (e.g., training → train) for ctx_data keys |
| `columns` | Full column classification: true_numeric, categorical_ids, string_categorical, complex_drop, parse_then_drop, labels |
| `features.structured` | Declarative feature specs: eq, lt, gt_quantile, binned — column names and thresholds |
| `features.args_parsing` | Args column parsing rules: path signals, flag signals, patterns to match |
| `cleaning` | categorical_fill value |
| `report` | Version string, whether to copy to repo root |
| `analysis` | drift_threshold, sample_size, random_state |

**Helper functions in `core/config.py`:**
- `load_config(pipeline, override_path)` — loads YAML, expands `~` in paths
- `get_cfg(cfg, "dot.path", default)` — nested dict access
- `split_keys(cfg)` — derives `(df_keys, label_keys)` from splits config
- `col_names(cfg, "category")` — get column name list
- `label_cols(cfg)` — get `[target, auxiliary]` label columns
- `build_column_classification(cfg)` — build full classification dict for reports

---

## The 13 Steps (Start to Finish)

### Phase 1: Data Acquisition — DONE

| # | Step | Status | File |
|---|---|---|---|
| 1 | Download BETH dataset from Kaggle | DONE | `download/tasks/kaggle_beth.py` |

- Idempotent download using `DOWNLOAD_COMPLETED.json` marker
- Dataset: `katehighnam/beth-dataset` → `~/python_venvs/datasets/kaggle/beth-dataset/`
- Credentials: `~/.kaggle/kaggle.json` or `KAGGLE_USERNAME`/`KAGGLE_KEY` env vars

### Phase 2: Data Understanding — DONE

| # | Step | Status | File |
|---|---|---|---|
| 2 | Programmatic data analysis (EDA) | DONE | `preprocessing/tasks/data_analysis.py` |

- Generates versioned JSON + HTML reports (currently v03)
- Column classification: 3 true numeric, 6 categorical IDs (look numeric!), 3 string categorical, 2 complex/drop, 2 labels
- Analyst insights with severity levels (critical/warning/info)
- Cross-split drift detection and unseen-category tracking
- Educational deep-dive: autoencoder explanation, cardinality, evaluation metrics, industry approaches
- Reports at `~/python_venvs/datasets/kaggle/beth-dataset/reports/`

### Phase 3: Data Preprocessing — IN PROGRESS (1 task remaining)

| # | Step | Status | File |
|---|---|---|---|
| 3 | Cleaning | DONE | `preprocessing/tasks/cleaning.py` |
| 4 | Feature Engineering | DONE | `preprocessing/tasks/feature_engineering.py` |
| 5 | Encoding | DONE | `preprocessing/tasks/encoding.py` |
| 6 | Scaling | TODO | `preprocessing/tasks/scaling.py` |

#### Step 3 — Cleaning

- Drop `stackAddresses` column (memory addresses — not useful without deep binary analysis)
- **Keep `args` column** — it contains file paths, flags, and syscall arguments with strong attack signal (parsed in feature engineering)
- Handle missing values using **training set medians/modes only** (no data leakage from val/test)
- Separate labels (`evil`, `sus`) from features before any transformation
- Store cleaned DataFrames and labels in `resp.ctx_data`

#### Step 4 — Feature Engineering

Create domain-informed features that capture attack signals. Each feature is derived from
cybersecurity domain knowledge (MITRE ATT&CK framework, HIDS best practices) and is
consistent with our column classification — no arithmetic on categorical ID columns.

| Feature | Formula | Attack Signal | Industry Basis |
|---|---|---|---|
| `is_root` | `userId == 0` | Privilege escalation — attacks need root to install malware, steal data, persist | MITRE T1068; every HIDS (OSSEC, Wazuh, Falco) monitors root activity |
| `return_negative` | `returnValue < 0` | Probing / brute force — attackers try things that fail (permission denied, file not found) | auditd rules, SIEM correlation for brute force detection |
| `return_category` | 0→success, <0→error, >0→info | Semantic grouping — raw return values span a huge range; binning captures the meaningful signal | Standard feature engineering practice for domain-specific continuous variables |
| `is_child_of_init` | `parentProcessId == 1` | Normal process tree — system services are children of init; attacker shells are spawned from compromised processes (web server, sshd) | Process tree analysis in Falco, Sysdig, CrowdStrike |
| `is_orphan` | `parentProcessId == 0` | Process injection — orphan processes (parent died or was manipulated) can indicate injection attacks or process hiding | MITRE T1055 (Process Injection) |
| `is_high_args` | `argsNum > training 95th percentile` | Unusually complex syscalls — buffer overflows and command injection stuff extra arguments | Anomaly-based IDS signature design |
| `args_touches_proc` | `/proc/` in args pathnames | Normal ops access /proc/ 34% of the time; attacks only 0.38% — **90x difference** | /proc/ enumeration is standard system monitoring |
| `args_touches_etc` | `/etc/` in args pathnames | Config file access patterns differ between normal (2.4%) and attack (0.3%) traffic | Config file monitoring in OSSEC, auditd |
| `args_has_write_flag` | `O_WRONLY\|O_RDWR\|O_CREAT` in flags | Normal has 13x more writes than attacks — attacks focus on reading/exfiltrating | File integrity monitoring (FIM) |
| `args_is_hidden_path` | `/.` in args pathnames | Hidden dirs used for malware staging (e.g., `/tmp/.X25-unix/.rsync/`) | MITRE T1564.001 (Hidden Files and Directories) |
| `args_has_pathname` | Any pathname present in args | Whether this syscall involves filesystem access at all | Basic syscall categorization |

**Why not `processId / parentProcessId` or `argsNum / eventId`?** Our data analysis proved
processId and eventId are categorical identifiers, not quantities. Dividing them produces
meaningless numbers (PID 500 / PID 250 = 2 has no semantic meaning). The features above
use only binary/categorical derivations that respect the column classification.

**Why extract from `args` instead of dropping it?** Data analysis revealed the `args` column
contains file paths and flags with massive signal separation between normal and attack traffic.
We parse it once during feature engineering into lightweight binary features, then drop the
raw string column before encoding.

- Apply identical transformations to train, val, and test
- `is_high_args` threshold must be computed from **training data only** (no data leakage)
- `args` binary features are extracted in feature engineering, then the raw `args` column is dropped

#### Step 5 — Encoding

- **Label-encode ALL 9 categorical columns:**
  - 3 string: `processName`, `hostName`, `eventName`
  - 6 numeric IDs: `processId`, `threadId`, `parentProcessId`, `userId`, `mountNamespace`, `eventId`
- `fit_transform` on training only, `transform` on val/test
- Handle unseen categories with UNKNOWN token
- Save encoders as pickle artifact (`label_encoders.pkl`)

#### Step 6 — Scaling

- `StandardScaler` on **TRUE NUMERIC columns only**: `timestamp`, `argsNum`, `returnValue` + engineered numeric features
- Do **NOT** scale label-encoded categorical columns (scaling distorts identifiers)
- `fit` on training only, `transform` on val/test
- Save scaler as pickle artifact (`scaler.pkl`)
- Convert to numpy `float32` arrays, save as `.npy` files
- Record `input_dim` (number of features) in `ctx_data` for model creation

### Phase 4: Model Architecture — TODO

| # | Step | Status | File |
|---|---|---|---|
| 7 | Autoencoder definition | TODO | `models/tasks/autoencoder.py` |

- `nn.Module` with encoder → bottleneck → decoder
- Default architecture: `input_dim → 64 → 32 → 16 → 32 → 64 → input_dim`
- `nn.Linear` + `nn.ReLU` layers, no activation on final output layer
- `nn.Sequential` for clean layer composition
- Store model instance in `resp.ctx_data["model"]`

### Phase 5: Training — TODO

| # | Step | Status | File |
|---|---|---|---|
| 8 | Training loop | TODO | `training/tasks/trainer.py` |

- `DataLoader` with `batch_size=256`, `shuffle=True`
- Loss: `nn.MSELoss()` (reconstruction error)
- Optimizer: `torch.optim.Adam(model.parameters(), lr=0.001)`
- 50 epochs with validation loss after each epoch
- Checkpointing:
  - `last.pt` — always overwritten (for resume)
  - `best.pt` — overwritten only when val_loss improves (for inference)
- Resume support: if `req.resume` is set, load checkpoint and continue from saved epoch
- Log `train_loss` and `val_loss` per epoch

### Phase 6: Evaluation & Inference — TODO

| # | Step | Status | File |
|---|---|---|---|
| 9 | Threshold calibration | TODO | `inference/tasks/predict.py` |
| 10 | Test evaluation | TODO | (same file) |
| 11 | Model & artifact saving | TODO | (same file) |

#### Step 9 — Threshold Calibration

- Compute per-sample reconstruction error on **validation set** (all normal data)
- Set threshold at **95th percentile** of validation reconstruction error
- Validate with `sus` column: do `sus=1` rows get higher error than `sus=0`? If yes, model is learning real signal

#### Step 10 — Test Evaluation

- Run test set through model → per-sample reconstruction error
- Apply threshold → binary predictions (1 = anomaly, 0 = normal)
- Compute metrics:

| Metric | What It Answers |
|---|---|
| **Precision** | Of everything flagged as attack, how many actually were? |
| **Recall** | Of all actual attacks, how many did we catch? (MOST IMPORTANT for security) |
| **F1 Score** | Balanced grade between precision and recall |
| **ROC-AUC** | How well the model separates attacks from normal, regardless of threshold (1.0 = perfect, 0.5 = random) |

- Generate evaluation report (JSON + HTML)

#### Step 11 — Save All Artifacts

- Best model weights (`best.pt`)
- Threshold value
- Scaler + encoders (already saved in preprocessing)
- Evaluation metrics
- Create `EVALUATION_REPORT.json`

### Phase 7: Real-Time Anomaly Detection — FUTURE

| # | Step | Status | File |
|---|---|---|---|
| 12 | Real-time prediction pipeline | FUTURE | TBD |
| 13 | Monitoring & retraining | FUTURE | TBD |

#### Step 12 — Real-Time Prediction

- Load saved model + scaler + encoders + threshold
- Accept streaming/new log events (e.g., from a message queue or API)
- Apply same preprocessing pipeline using saved artifacts
- Score anomaly (reconstruction error vs threshold)
- Flag and alert if above threshold

#### Step 13 — Monitoring & Retraining

- Track prediction distribution over time
- Detect concept drift (what's "normal" evolves as infrastructure changes)
- Periodic retraining on new normal data
- A/B comparison of model versions

---

## Implementation Order

Each step follows the same pattern: **create file → add task name to facade → run pipeline → check logs**.

| # | File to Create | Facade to Update | Depends On |
|---|---|---|---|
| 1 | `preprocessing/tasks/cleaning.py` | Uncomment `"cleaning"` in `preprocessing/facade.py` | data_analysis (done) |
| 2 | `preprocessing/tasks/feature_engineering.py` | Uncomment `"feature_engineering"` | cleaning |
| 3 | `preprocessing/tasks/encoding.py` | Uncomment `"encoding"` | feature_engineering |
| 4 | `preprocessing/tasks/scaling.py` | Uncomment `"scaling"` | scaling |
| 5 | `models/tasks/autoencoder.py` | Add `"autoencoder"` to `models/facade.py` | scaling (needs `input_dim`) |
| 6 | `training/tasks/trainer.py` | Add `"trainer"` to `training/facade.py` | autoencoder (needs `model`) |
| 7 | `inference/tasks/predict.py` | Add `"predict"` to `inference/facade.py` | trainer (needs trained model) |

All file paths are relative to `src/pipelines/cyber_attack_detection/`.

---

## Data Flow Through `ctx_data`

```
download
  └─► raw_data_path = "~/python_venvs/datasets/kaggle/beth-dataset/"

data_analysis
  └─► analysis_report, column_classification

cleaning
  └─► train_df, val_df, test_df           (features only, no labels)
  └─► train_labels, val_labels, test_labels (evil + sus columns)

feature_engineering
  └─► train_df, val_df, test_df           (with new engineered columns)

encoding
  └─► train_df, val_df, test_df           (all categorical → integers)
  └─► encoders                            (dict of LabelEncoders)

scaling
  └─► processed .npy files saved to disk
  └─► input_dim                           (number of features after all transforms)
  └─► scaler                              (StandardScaler object)

autoencoder
  └─► model                               (Autoencoder nn.Module instance)

trainer
  └─► model with trained weights loaded
  └─► checkpoint_dir                      (path to best.pt / last.pt)

predict
  └─► threshold                           (95th percentile of val reconstruction error)
  └─► predictions                         (binary array)
  └─► metrics                             (Precision, Recall, F1, ROC-AUC)
  └─► evaluation report paths
```

---

## Task File Template

Every task follows this pattern:

```python
from core.common.wfs.interfaces import WfTask
from core.common.wfs.dtos import WfReq, WfResp
from core.logger import get_logger

log = get_logger(__name__)


class YourClassName(WfTask):
    def execute(self, req: WfReq, resp: WfResp) -> WfResp:
        # Read from previous tasks:  resp.ctx_data["key"]
        # Write for next tasks:      resp.ctx_data["new_key"] = value
        # Log progress:              log.debug("...")
        resp.message = "task completed"
        return resp


Task = YourClassName
```

The `Task = YourClassName` alias at the bottom is required — the facade uses it to load the class.

---

## Hyperparameters to Experiment With (After End-to-End Works)

| Parameter | Default | Try | Effect |
|---|---|---|---|
| `hidden_dims` | `[64, 32]` | `[128, 64, 32]` | Deeper network, more capacity |
| `bottleneck_dim` | `16` | `8` or `32` | Tighter/looser compression |
| `lr` | `0.001` | `0.0001` or `0.01` | Slower/faster learning |
| `batch_size` | `256` | `512` or `64` | Larger = smoother, smaller = noisier |
| `num_epochs` | `50` | `100` | More training time |
| `threshold percentile` | `0.95` | `0.99` or `0.90` | Sensitivity tradeoff |

---

## Running the Pipeline

```bash
# Full pipeline (all sub-workflows)
npm run cyber-attack-detection

# Start from a specific sub-workflow
npm run cyber-attack-detection:from -- preprocessing

# Resume training from checkpoint
npm run cyber-attack-detection:resume -- data/checkpoints/last.pt
```

---

## Key Decisions Made

1. **Anomaly detection, not classification** — training data has zero attacks, so we learn "normal" and flag deviations
2. **Autoencoder** — sweet spot between simplicity (vs Isolation Forest) and power (vs LSTM). Works directly on tabular features
3. **6 columns reclassified** — processId, threadId, parentProcessId, userId, mountNamespace, eventId are identifiers, not quantities. Label-encoded, not scaled
4. **StandardScaler on 3 true numeric only** — timestamp, argsNum, returnValue. Scaling categorical encodings would teach false relationships
5. **Recall > Precision** — in cybersecurity, missing a real attack is worse than a false alarm
6. **ROC-AUC as primary metric** — threshold-independent measure of model quality
7. **Threshold from validation 95th percentile** — tunable; `sus` column provides sanity check
