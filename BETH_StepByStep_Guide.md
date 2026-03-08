# BETH Dataset — Step-by-Step PyTorch Guide

> This guide walks you through building a cyber attack detection model from scratch.
> Every concept is explained as if you've never done this before.
> You'll write all the code yourself. The guide tells you **what** to write, **why**, and **what to expect**.

---

## Table of Contents

1. [Understanding the Problem](#1-understanding-the-problem)
2. [Understanding Our Data](#2-understanding-our-data)
3. [Data Analysis (automated)](#3-data-analysis)
4. [Step-by-Step: Cleaning Task](#4-cleaning-task)
5. [Step-by-Step: Feature Engineering Task](#5-feature-engineering-task)
6. [Step-by-Step: Encoding Task](#6-encoding-task)
7. [Step-by-Step: Scaling Task](#7-scaling-task)
8. [Step-by-Step: Model Architecture](#8-model-architecture)
9. [Step-by-Step: Training Loop](#9-training-loop)
10. [Step-by-Step: Evaluation & Threshold](#10-evaluation-and-threshold)
11. [Step-by-Step: Inference](#11-inference)
12. [Putting It All Together](#12-putting-it-all-together)

---

## 1. Understanding the Problem

### What is anomaly detection?

Imagine you work at a security desk in a building. Every day, hundreds of people walk in and out. You've watched them for months — you know the patterns. Most people badge in at 9am, take the elevator, go to their floor.

One day, someone walks in at 3am, goes to the server room, and starts unplugging things. You've never seen this **pattern** before. It's an **anomaly**.

That's exactly what we're building. Our "building" is a computer network. The "people" are system processes (programs running on servers). The "patterns" are things like which process ran, who started it, what it did. We want to learn what "normal" looks like, and then flag anything that **doesn't look normal**.

### Why can't we just train a normal classifier?

A normal classifier (like "cat vs dog") needs lots of examples of **both** classes. But our training data has **zero attacks**. All 763,144 training rows are normal traffic. This is realistic — in the real world, you have months of normal logs and attacks are rare.

So we use an **autoencoder** — a neural network that learns to compress and reconstruct normal data. When it sees attack data, it can't reconstruct it well (because it's never seen anything like it). The **reconstruction error** becomes our alarm signal.

### What is BETH?

BETH = **B**iased **E**valuation of **T**races from **H**oneypots. Researchers set up honeypot servers (fake servers designed to attract attackers), recorded all system-level events (process creation, file access, network calls), and labelled them.

---

## 2. Understanding Our Data

### The files

| File | Rows | What it is |
|---|---|---|
| `labelled_training_data.csv` | 763,144 | 100% normal traffic — this is what we learn from |
| `labelled_validation_data.csv` | 188,967 | 100% normal traffic — tune our threshold here |
| `labelled_testing_data.csv` | 188,967 | Mix: 30,535 normal + 158,432 attacks — final evaluation |
| `labelled_2021may-*.csv` | varies | Per-host raw data (we won't use these directly) |
| `*-dns.csv` | small | DNS-specific logs (park for later) |

### The columns (16 total)

| Column | Type | What it means | Example |
|---|---|---|---|
| `timestamp` | float | Seconds since some start time | `1809.495787` |
| `processId` | int | ID of the process that generated this event | `381` |
| `threadId` | int | Thread within the process | `7337` |
| `parentProcessId` | int | The process that spawned this one | `1` |
| `userId` | int | Linux user ID running the process | `100` |
| `mountNamespace` | int | Linux namespace (container isolation) | `4026532231` |
| `processName` | string | Name of the program | `close` |
| `hostName` | string | Which server this happened on | `ip-10-100-1-120` |
| `eventId` | int | Type of system event (numeric code) | `157` |
| `eventName` | string | Human-readable event name | `prctl` |
| `stackAddresses` | string | Memory addresses on the call stack | `[140662...]` |
| `argsNum` | int | How many arguments the event had | `5` |
| `returnValue` | int | Return code of the system call | `0` |
| `args` | string | JSON-like string with argument details | `[{'name': ...}]` |
| `sus` | int (0/1) | Was this flagged as suspicious? | `1` |
| `evil` | int (0/1) | Was this an actual attack? **This is our target** | `0` |

### Key observations to keep in mind

1. **Training has 0 attacks, test has 84% attacks** — this is anomaly detection, not classification
2. **`sus` column exists in training** — 1,269 rows are suspicious but not evil. This is a weak signal we can use
3. **`args` and `stackAddresses` are complex strings** — we'll drop these initially
4. **`processName`, `hostName`, `eventName` are categorical** — need encoding
5. **Numeric columns have very different scales** — need normalization

---

## 3. Data Analysis

The `data_analysis` task (already created) runs automatically as the first preprocessing step. Run:

```bash
npm run cyber-attack-detection
```

It generates a JSON report at `~/python_venvs/datasets/kaggle/beth-dataset/reports/data_analysis_report.json`.

**What to look for in the report:**

- **missing_values**: Any columns with nulls? How many? This tells you what cleaning needs to handle
- **class_distribution**: Confirms the 0-attack training, mixed test split
- **numeric_stats**: Look at min/max/mean/std for each column. If `std` is huge compared to `mean`, the data has outliers. If `zeros_pct` is very high, the column might not be useful
- **categorical_stats**: How many unique process names? If there are thousands, simple one-hot encoding won't work (too many columns). We'll need label encoding or embeddings

**Read this report carefully before proceeding. It tells you exactly what your cleaning and encoding steps need to handle.**

---

## 4. Cleaning Task

**File to create:** `src/pipelines/cyber_attack_detection/preprocessing/tasks/cleaning.py`

**After creating, uncomment `"cleaning"` in `preprocessing/facade.py`'s TASKS list.**

### Step 4.1: Understand what cleaning means

Cleaning is about making the data "ready to work with." Raw data has problems: missing values, weird columns, wrong types. Think of it like washing vegetables before cooking — you need to remove the dirt.

### Step 4.2: What your cleaning task should do

Here's what the `execute` method needs to do, in order:

#### 4.2.1: Load the CSV files

```python
import pandas as pd

train_df = pd.read_csv(f"{raw_data_path}/labelled_training_data.csv")
val_df = pd.read_csv(f"{raw_data_path}/labelled_validation_data.csv")
test_df = pd.read_csv(f"{raw_data_path}/labelled_testing_data.csv")
```

**Why:** Everything starts with loading data into pandas DataFrames — the standard "table" structure in Python for data work.

**Where does `raw_data_path` come from?** It's in `resp.ctx_data["raw_data_path"]`, set by the download task.

#### 4.2.2: Drop columns we can't use

```python
drop_cols = ["stackAddresses"]
train_df = train_df.drop(columns=drop_cols)
```

**Why drop `stackAddresses`?** It's a list of raw memory addresses from the call stack.
Extracting useful features from it requires deep binary analysis expertise. Not useful for
our model.

**Why KEEP `args`?** We originally planned to drop it, but data analysis revealed it
contains file paths and flags with massive attack signal:
- Normal operations access `/proc/` 34% of the time, attacks only 0.38% (90x difference!)
- Normal has 13x more write operations than attacks
- Attack paths include hidden malware staging dirs like `/tmp/.X25-unix/.rsync/`

We'll parse `args` in the feature engineering step to extract binary features, then drop
the raw string column. Keep it for now.

**Do this for all three DataFrames** (train, val, test).

#### 4.2.3: Handle missing values

```python
print(train_df.isnull().sum())
```

Run this first to see if there ARE missing values. The data analysis report already has this info.

If there are missing values:
- For numeric columns: fill with the **median** (not mean — median is robust to outliers)
- For categorical columns: fill with a special string like `"UNKNOWN"`

```python
for col in numeric_cols:
    median_val = train_df[col].median()
    train_df[col] = train_df[col].fillna(median_val)
    val_df[col] = val_df[col].fillna(median_val)    # use TRAIN median, not val's own
    test_df[col] = test_df[col].fillna(median_val)   # use TRAIN median, not test's own
```

**Critical rule:** Always compute fill values (median, mean, etc.) from the **training set only**, then apply to val/test. If you compute from test data, you're "leaking" future information into your model.

#### 4.2.4: Separate features from labels

```python
target_col = "evil"
aux_col = "sus"

train_labels = train_df[[target_col, aux_col]].copy()
train_df = train_df.drop(columns=[target_col, aux_col])

# Same for val and test
```

**Why:** Labels are what we want to predict. They should NOT be fed to the model as input features. Separate them now so you don't accidentally include them.

#### 4.2.5: Store everything in ctx_data

After cleaning, put the DataFrames into `resp.ctx_data` so the next task can access them:

```python
resp.ctx_data["train_df"] = train_df
resp.ctx_data["val_df"] = val_df
resp.ctx_data["test_df"] = test_df
resp.ctx_data["train_labels"] = train_labels
resp.ctx_data["val_labels"] = val_labels
resp.ctx_data["test_labels"] = test_labels
```

### Step 4.3: The structure of your task class

```python
from core.common.wfs.interfaces import WfTask
from core.common.wfs.dtos import WfReq, WfResp
from core.logger import get_logger

log = get_logger(__name__)

class Cleaning(WfTask):
    def execute(self, req: WfReq, resp: WfResp) -> WfResp:
        # Step 4.2.1: Load CSVs
        # Step 4.2.2: Drop columns
        # Step 4.2.3: Handle missing values
        # Step 4.2.4: Separate features from labels
        # Step 4.2.5: Store in ctx_data
        resp.message = "cleaning completed"
        return resp

Task = Cleaning
```

### Step 4.4: Testing your task

After you write it, run:

```bash
npm run cyber-attack-detection
```

Check `runtime_logs/app.log` for your DEBUG messages. If it fails, read the error — it'll tell you exactly what went wrong.

---

## 5. Feature Engineering Task

**File to create:** `src/pipelines/cyber_attack_detection/preprocessing/tasks/feature_engineering.py`

**After creating, uncomment `"feature_engineering"` in `preprocessing/facade.py`'s TASKS list.**

### Step 5.1: What is feature engineering?

Imagine you're looking at a person's behavior over time. The raw data says "they entered at 9:00, 9:05, 9:10." But what's MORE informative is "they entered 3 times in 10 minutes." That derived number — the **rate** — is a feature you engineered.

Feature engineering means creating NEW columns from existing ones that help the model learn patterns better.

### Step 5.2: How a cybersecurity analyst decides what features to create

This isn't guesswork. Each feature comes from known attack patterns documented in the
**MITRE ATT&CK framework** (the industry encyclopedia of attack techniques) and
real-world **Host-based Intrusion Detection System (HIDS)** practices used in
tools like OSSEC, Wazuh, Falco, CrowdStrike, and Sysdig.

The golden rule: **only do arithmetic on columns where arithmetic makes sense.** We proved
that processId, eventId, etc. are categorical IDs — dividing them produces nonsense.
Every feature below uses only binary checks or operations on truly numeric columns.

### Step 5.3: Features to create

#### 5.3.1: Is the user root?

```python
df["is_root"] = (df["userId"] == 0).astype(int)
```

**The attack pattern (MITRE T1068 — Privilege Escalation):** Almost every serious attack
needs root access at some point. An attacker breaks in as a normal user (say, through a
web server vulnerability), then "escalates" to root. Once root, they can install backdoors,
steal credentials, and hide their tracks. Every security tool in the industry monitors
root activity — it's security 101.

**What the autoencoder learns:** The normal rate and timing of root processes. If attacks
cause a sudden spike in root activity from unusual processes, reconstruction error goes up.

#### 5.3.2: Is the return value negative (error)?

```python
df["return_negative"] = (df["returnValue"] < 0).astype(int)
```

**The attack pattern (probing / brute force):** When an attacker is exploring a system,
they try things that fail. They try to read `/etc/shadow` (permission denied = -13). They
try to connect to internal services (connection refused = -111). They try to execute commands
they can't (operation not permitted = -1). Each failure generates a negative return value.

**Think of it this way:** A burglar rattling every door handle on a street. Most are
locked. Normal residents rarely rattle locked doors — they have the key. A sudden burst
of "locked door" signals = someone is probing.

**Industry basis:** Failed syscall monitoring is a core feature of `auditd` (Linux audit
daemon) and every SIEM correlation rule for brute force detection.

#### 5.3.3: Return value category

```python
df["return_category"] = 0  # default: success
df.loc[df["returnValue"] < 0, "return_category"] = 1   # error
df.loc[df["returnValue"] > 0, "return_category"] = 2   # info
```

**Why bin instead of using the raw number?** The raw returnValue ranges from large
negatives to large positives. The specific number (-13 vs -111) matters less than the
category (both are errors). Binning compresses the signal into something the model
can learn more easily.

**The three groups:**
- **0 (success):** The syscall worked as expected. Normal operations are mostly success.
- **1 (error):** Something went wrong. Permission denied, file not found, connection refused.
- **2 (info):** The syscall returned data. A file descriptor number, bytes read, etc.

#### 5.3.4: Is this process a child of init?

```python
df["is_child_of_init"] = (df["parentProcessId"] == 1).astype(int)
```

**The attack pattern (MITRE T1059 — Command Scripting):** On Linux, PID 1 is
`init` (or `systemd`) — the first process that starts everything else. Legitimate system
services (sshd, nginx, cron) are children of init. But an attacker's reverse shell is
spawned from the *compromised* process (e.g., a web server spawns bash). So an attacker's
process has a parent like PID 4582 (nginx), not PID 1.

**Why not divide processId by parentProcessId?** We proved these are categorical IDs.
PID 500 / PID 250 = 2 is meaningless. But "is parent == 1?" is a clean binary check
on the *identity* of the parent, not arithmetic on IDs.

**Industry basis:** Process tree analysis is a core detection technique in CrowdStrike
Falcon, Sysdig Secure, and Falco rules.

#### 5.3.5: Is this an orphan process?

```python
df["is_orphan"] = (df["parentProcessId"] == 0).astype(int)
```

**The attack pattern (MITRE T1055 — Process Injection):** An orphan process is one whose
parent has died or been manipulated. Attackers use process injection to run malicious code
inside a legitimate process, then kill the original. The injected process becomes an orphan
(parentProcessId = 0). Orphans in normal operations are rare.

**Industry basis:** Orphan process detection is a standard HIDS rule in OSSEC and Wazuh.

#### 5.3.6: Is the argument count unusually high?

```python
threshold = df["argsNum"].quantile(0.95)  # compute from TRAINING data only
df["is_high_args"] = (df["argsNum"] > threshold).astype(int)
```

**The attack pattern (buffer overflow / command injection):** When an attacker exploits a
buffer overflow or injects commands, they often stuff extra arguments into syscalls. Normal
operations use a predictable number of arguments. An unusually high count is a red flag.

**Why not `argsNum / eventId`?** We proved eventId is an enum (157 = prctl, 2 = open).
Dividing by an enum code is meaningless. Instead, we use a statistical threshold: "is this
call's argument count in the top 5% of what we've seen in training?"

**Critical:** The 95th percentile threshold must be computed from **training data only**,
then applied to val/test. This prevents data leakage.

#### 5.3.7: Features from the `args` column (the hidden gold mine)

We almost dropped this column — but the data told us not to. The `args` column is a
JSON-like string containing syscall arguments: file paths, flags, file descriptors. We
don't feed the raw string to the model — we extract binary signals from it.

```python
import ast

def extract_args_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["args_has_pathname"] = 0
    df["args_touches_proc"] = 0
    df["args_touches_etc"] = 0
    df["args_has_write_flag"] = 0
    df["args_is_hidden_path"] = 0

    for idx, args_str in df["args"].items():
        try:
            args_list = ast.literal_eval(args_str)
        except (ValueError, SyntaxError):
            continue
        for a in args_list:
            name = a.get("name", "")
            val = str(a.get("value", ""))
            if name in ("pathname", "filename"):
                df.at[idx, "args_has_pathname"] = 1
                if "/proc/" in val:
                    df.at[idx, "args_touches_proc"] = 1
                if "/etc/" in val:
                    df.at[idx, "args_touches_etc"] = 1
                if "/." in val:
                    df.at[idx, "args_is_hidden_path"] = 1
            if name == "flags" and ("WRONLY" in val or "RDWR" in val or "CREAT" in val):
                df.at[idx, "args_has_write_flag"] = 1

    df = df.drop(columns=["args"])  # Drop raw string after extracting features
    return df
```

**The data proves these matter:**

| Feature | Training (normal) | Test attacks | Separation |
|---|---|---|---|
| `args_touches_proc` | 34.3% | 0.4% | **90x difference** — strongest signal in the dataset |
| `args_has_write_flag` | 0.7% | 0.05% | **13x difference** — attacks read/steal, normals write |
| `args_touches_etc` | 2.4% | 0.3% | **8x difference** — config file access patterns |
| `args_is_hidden_path` | 0.03% | 0.04% | Low frequency, but catches malware staging dirs |

**The attack paths tell a story:** Normal servers constantly read `/proc/` for monitoring
(CPU usage, memory, process lists). Attackers don't do this — they're busy executing
commands, exfiltrating data, and installing backdoors. The absence of `/proc/` access is
itself a signal.

**Why not feed raw `args` to the model?** The raw string is variable-length, nested JSON.
Neural networks need fixed-size numeric input. Parsing once into binary columns gives us
the signal without the complexity.

### Step 5.4: Important rules

1. **Apply identical transformations to train, val, and test.** Write a helper function:

```python
def add_features(df: pd.DataFrame, args_threshold: float) -> pd.DataFrame:
    df = df.copy()
    df["is_root"] = (df["userId"] == 0).astype(int)
    df["return_negative"] = (df["returnValue"] < 0).astype(int)
    df["return_category"] = 0
    df.loc[df["returnValue"] < 0, "return_category"] = 1
    df.loc[df["returnValue"] > 0, "return_category"] = 2
    df["is_child_of_init"] = (df["parentProcessId"] == 1).astype(int)
    df["is_orphan"] = (df["parentProcessId"] == 0).astype(int)
    df["is_high_args"] = (df["argsNum"] > args_threshold).astype(int)
    df = extract_args_features(df)  # Parse args, extract binary features, drop raw column
    return df

# Compute threshold from TRAINING only
args_threshold = resp.ctx_data["train_df"]["argsNum"].quantile(0.95)

train_df = add_features(resp.ctx_data["train_df"], args_threshold)
val_df = add_features(resp.ctx_data["val_df"], args_threshold)
test_df = add_features(resp.ctx_data["test_df"], args_threshold)
```

2. **Save `args_threshold` in ctx_data** — inference will need it for new data.

### Step 5.5: Store back

Update `resp.ctx_data["train_df"]`, `resp.ctx_data["val_df"]`, `resp.ctx_data["test_df"]` with the new DataFrames. Also store `resp.ctx_data["args_threshold"]`.

---

## 6. Encoding Task

**File to create:** `src/pipelines/cyber_attack_detection/preprocessing/tasks/encoding.py`

**After creating, uncomment `"encoding"` in `preprocessing/facade.py`'s TASKS list.**

### Step 6.1: Why encoding?

Neural networks only understand **numbers**. They can't process the string `"systemd"` or `"prctl"`. Encoding converts strings to numbers.

### Step 6.2: Which columns need encoding?

From our data: `processName`, `hostName`, `eventName`. These are categorical (string) columns.

### Step 6.3: Label Encoding (recommended for this dataset)

Label encoding assigns a unique integer to each unique string:

```
"systemd" -> 0
"close" -> 1
"prctl" -> 2
...
```

**Why label encoding instead of one-hot?**
- `processName` has potentially hundreds of unique values
- One-hot would create hundreds of new columns (one per value), making the data huge
- For an autoencoder, label encoding works well enough

### Step 6.4: How to implement

```python
from sklearn.preprocessing import LabelEncoder

categorical_cols = ["processName", "hostName", "eventName"]
encoders = {}

for col in categorical_cols:
    le = LabelEncoder()
    train_df[col] = le.fit_transform(train_df[col].astype(str))
    encoders[col] = le
```

**Critical:** `fit_transform` on **training data only**. For val/test, use `transform` only:

```python
    val_df[col] = le.transform(val_df[col].astype(str))
    test_df[col] = le.transform(test_df[col].astype(str))
```

**Problem:** What if val/test has a category that wasn't in training? `transform` will crash. Handle it:

```python
    # For val/test: replace unseen categories with a placeholder before transform
    known = set(le.classes_)
    val_df[col] = val_df[col].astype(str).apply(lambda x: x if x in known else "UNKNOWN")
```

And add `"UNKNOWN"` to the encoder's classes during fitting by appending it to training data before fitting.

### Step 6.5: Save the encoders

```python
import pickle
from pathlib import Path

artifacts_dir = Path("data/processed/cyber_attack_detection/artifacts")
artifacts_dir.mkdir(parents=True, exist_ok=True)

with open(artifacts_dir / "label_encoders.pkl", "wb") as f:
    pickle.dump(encoders, f)
```

**Why:** During inference (predicting on new data), you need the exact same encoder mapping. If you don't save it, you'd have to re-fit on training data every time.

Store the path in `resp.ctx_data["encoders_path"]` and the encoders dict in `resp.ctx_data["encoders"]`.

---

## 7. Scaling Task

**File to create:** `src/pipelines/cyber_attack_detection/preprocessing/tasks/scaling.py`

**After creating, uncomment `"scaling"` in `preprocessing/facade.py`'s TASKS list.**

### Step 7.1: Why scaling?

Look at our numeric columns:
- `timestamp` ranges from ~0 to ~86400 (seconds in a day)
- `processId` ranges from 0 to ~30000
- `argsNum` ranges from 0 to ~20
- `mountNamespace` is in the billions

If you feed these raw numbers to a neural network, the network will pay way more attention to `mountNamespace` (huge numbers) than `argsNum` (tiny numbers), simply because of scale. This is wrong — `argsNum` might be just as important.

**Scaling** puts all columns on the same scale (typically mean=0, std=1).

### Step 7.2: StandardScaler

```python
from sklearn.preprocessing import StandardScaler

numeric_cols = train_df.select_dtypes(include="number").columns.tolist()

scaler = StandardScaler()
train_df[numeric_cols] = scaler.fit_transform(train_df[numeric_cols])
```

After scaling, every column has approximately mean=0 and standard deviation=1.

**Critical:** Same rule — fit on train, transform on val/test:

```python
val_df[numeric_cols] = scaler.transform(val_df[numeric_cols])
test_df[numeric_cols] = scaler.transform(test_df[numeric_cols])
```

### Step 7.3: Save the scaler

```python
with open(artifacts_dir / "scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)
```

### Step 7.4: Convert to numpy arrays

After scaling, convert to numpy arrays (PyTorch will need these):

```python
import numpy as np

train_array = train_df.values.astype(np.float32)
val_array = val_df.values.astype(np.float32)
test_array = test_df.values.astype(np.float32)
```

**Why float32?** PyTorch uses float32 by default. float64 is slower and uses more memory with no real benefit for this use case.

### Step 7.5: Save processed data

```python
np.save(processed_dir / "train.npy", train_array)
np.save(processed_dir / "val.npy", val_array)
np.save(processed_dir / "test.npy", test_array)

# Also save labels
np.save(processed_dir / "train_labels.npy", train_labels.values)
np.save(processed_dir / "val_labels.npy", val_labels.values)
np.save(processed_dir / "test_labels.npy", test_labels.values)
```

Store paths and the **number of features** (`train_array.shape[1]`) in `resp.ctx_data` — the model needs to know how many input features there are.

---

## 8. Model Architecture

**File to create:** `src/pipelines/cyber_attack_detection/models/tasks/autoencoder.py`

**Add `"autoencoder"` to `models/facade.py`'s TASKS list.**

### Step 8.1: What is an autoencoder?

Think of it like a game of telephone:

```
Original message → Person 1 (compresses) → Person 2 (expands) → Reconstructed message
```

If the original and reconstructed messages are nearly identical, great — the message was "normal" and easy to compress. If they're very different, the message was weird.

An autoencoder has three parts:

```
Input (N features) → ENCODER → Bottleneck (small) → DECODER → Output (N features)
```

- **Encoder**: Takes N input features and compresses them down to fewer dimensions
- **Bottleneck**: The compressed representation (like a summary)
- **Decoder**: Tries to reconstruct the original N features from the summary

### Step 8.2: Define the network in PyTorch

Here's the thinking behind each piece:

```python
import torch
import torch.nn as nn
```

**`torch`** = PyTorch, the deep learning library.
**`torch.nn`** = the "neural network" module — contains all the building blocks.

#### 8.2.1: The class definition

```python
class Autoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dims, bottleneck_dim):
        super().__init__()
        ...
```

**`nn.Module`**: Every PyTorch model inherits from this. It gives you automatic parameter tracking, GPU support, save/load, etc.

**`input_dim`**: The number of features. After preprocessing, this is the number of columns in your data. You'll get this from `resp.ctx_data`.

**`hidden_dims`**: A list like `[64, 32]`. These are the sizes of the hidden layers. The encoder shrinks: input_dim → 64 → 32 → bottleneck. The decoder expands back: bottleneck → 32 → 64 → input_dim.

**`bottleneck_dim`**: The smallest layer. Like `16`. This is the "summary" size.

#### 8.2.2: Building the encoder

```python
        # Encoder: input_dim -> hidden_dims[0] -> hidden_dims[1] -> ... -> bottleneck_dim
        encoder_layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            encoder_layers.append(nn.Linear(prev_dim, h_dim))
            encoder_layers.append(nn.ReLU())
            prev_dim = h_dim
        encoder_layers.append(nn.Linear(prev_dim, bottleneck_dim))
        self.encoder = nn.Sequential(*encoder_layers)
```

**`nn.Linear(in, out)`**: A fully connected layer. It multiplies the input by a weight matrix and adds a bias. This is the fundamental building block.

**`nn.ReLU()`**: An activation function. Without it, stacking Linear layers is pointless — multiple linear transformations collapse into a single one. ReLU introduces non-linearity: `ReLU(x) = max(0, x)`. Think of it as a filter that keeps positive values and zeros out negatives.

**`nn.Sequential`**: Chains layers together. Data flows through them in order.

#### 8.2.3: Building the decoder

The decoder is the encoder in reverse:

```python
        # Decoder: bottleneck_dim -> hidden_dims[-1] -> ... -> hidden_dims[0] -> input_dim
        decoder_layers = []
        prev_dim = bottleneck_dim
        for h_dim in reversed(hidden_dims):
            decoder_layers.append(nn.Linear(prev_dim, h_dim))
            decoder_layers.append(nn.ReLU())
            prev_dim = h_dim
        decoder_layers.append(nn.Linear(prev_dim, input_dim))
        self.decoder = nn.Sequential(*decoder_layers)
```

**Notice:** The last layer has NO activation function. We want raw values as output, not ReLU-clipped values.

#### 8.2.4: The forward method

```python
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
```

**`forward`**: This is THE method PyTorch calls when you do `model(data)`. It defines how data flows through the network.

`x` goes in (N features) → gets compressed → gets expanded → comes back out (N features). We compare the output to the original input to see how good the reconstruction is.

### Step 8.3: What your models task should do

The models task **creates** the model and **stores it** in `resp.ctx_data`:

```python
    def execute(self, req: WfReq, resp: WfResp) -> WfResp:
        input_dim = resp.ctx_data["input_dim"]  # set by scaling task
        hidden_dims = [64, 32]
        bottleneck_dim = 16

        model = Autoencoder(input_dim, hidden_dims, bottleneck_dim)
        resp.ctx_data["model"] = model
        resp.ctx_data["hidden_dims"] = hidden_dims
        resp.ctx_data["bottleneck_dim"] = bottleneck_dim
        resp.message = f"Autoencoder created: {input_dim} -> {hidden_dims} -> {bottleneck_dim}"
        return resp
```

### Step 8.4: Understanding the shapes

If your data has 14 features after preprocessing:

```
Input:      [batch_size, 14]
Encoder L1: [batch_size, 64]   (14 → 64: EXPAND first, learn rich representation)
Encoder L2: [batch_size, 32]   (64 → 32: compress)
Bottleneck: [batch_size, 16]   (32 → 16: maximum compression)
Decoder L1: [batch_size, 32]   (16 → 32: start expanding)
Decoder L2: [batch_size, 64]   (32 → 64: keep expanding)
Output:     [batch_size, 14]   (64 → 14: back to original size)
```

**`batch_size`**: We don't feed all 763K rows at once (too much memory). We feed them in batches (e.g., 256 rows at a time). More on this in the training section.

---

## 9. Training Loop

**File to create:** `src/pipelines/cyber_attack_detection/training/tasks/trainer.py`

**Add `"trainer"` to `training/facade.py`'s TASKS list.**

### Step 9.1: What does training mean?

Training is an iterative process:

1. Show the model a batch of normal data
2. The model tries to reconstruct it
3. Measure how bad the reconstruction is (the **loss**)
4. Adjust the model's weights to make the loss smaller
5. Repeat thousands of times

After many rounds, the model gets good at reconstructing normal data — and bad at reconstructing attacks (because it's never seen them).

### Step 9.2: The Dataset and DataLoader

Before training, you need to wrap your numpy arrays in PyTorch's data loading utilities:

```python
import torch
from torch.utils.data import TensorDataset, DataLoader

train_array = np.load(processed_dir / "train.npy")
train_tensor = torch.from_numpy(train_array)
train_dataset = TensorDataset(train_tensor)
train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
```

**`TensorDataset`**: Wraps numpy data as a PyTorch dataset.

**`DataLoader`**: Handles batching, shuffling, and parallel loading. When you iterate over it, you get one batch at a time.

**`batch_size=256`**: Feed 256 rows at a time. Why?
- Too small (e.g., 1): Very noisy updates, slow training
- Too large (e.g., all 763K): Uses too much memory, may converge to bad solutions
- 256 is a common sweet spot

**`shuffle=True`**: Randomize the order each epoch. If the model sees data in the same order every time, it might learn the order instead of the patterns.

### Step 9.3: The loss function

```python
criterion = nn.MSELoss()
```

**MSE = Mean Squared Error**. For each input, it computes:

```
loss = average of (input - reconstruction)² across all features
```

The squared difference penalizes large errors more than small ones. If one feature is reconstructed very badly, the model is strongly pushed to fix it.

### Step 9.4: The optimizer

```python
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
```

**Optimizer**: The algorithm that adjusts the model's weights based on the loss.

**Adam**: The most commonly used optimizer. It adapts the learning rate for each parameter individually. Almost always a good default.

**`model.parameters()`**: Tells the optimizer which numbers it's allowed to change (the weights and biases of all layers).

**`lr=0.001`**: Learning rate — how big each adjustment step is. Too large = overshoots, too small = takes forever. 0.001 is a good starting point for Adam.

### Step 9.5: The training loop (epoch by epoch)

```python
num_epochs = 50

for epoch in range(num_epochs):
    model.train()                    # Put model in training mode
    epoch_loss = 0.0

    for (batch,) in train_loader:    # Loop over batches
        optimizer.zero_grad()        # Reset gradients from previous batch
        reconstruction = model(batch)       # Forward pass: input → model → reconstruction
        loss = criterion(reconstruction, batch)  # Compare reconstruction to original
        loss.backward()              # Backward pass: compute gradients
        optimizer.step()             # Update weights using gradients
        epoch_loss += loss.item()    # Accumulate loss for logging

    avg_loss = epoch_loss / len(train_loader)
    print(f"Epoch {epoch+1}/{num_epochs} — loss: {avg_loss:.6f}")
```

Let's break down the 5 critical lines inside the batch loop:

#### `optimizer.zero_grad()`
PyTorch **accumulates** gradients by default. If you don't zero them, each batch's gradients add to the previous batch's, which is wrong. Always zero before computing new gradients.

#### `reconstruction = model(batch)`
This calls `model.forward(batch)`. The batch flows through encoder → bottleneck → decoder. PyTorch automatically records every operation for backpropagation.

#### `loss = criterion(reconstruction, batch)`
Compares the reconstruction to the original input. Note: the **target is the input itself** — this is what makes it an autoencoder, not a classifier.

#### `loss.backward()`
**Backpropagation**. Starting from the loss, PyTorch walks backwards through every operation and computes how much each weight contributed to the error. These are called **gradients**.

#### `optimizer.step()`
Uses the gradients to adjust each weight. If a weight increased the loss, make it smaller. If it decreased the loss, make it bigger. This is literally how the network "learns."

### Step 9.6: Validation after each epoch

After each epoch, check reconstruction error on validation data:

```python
    model.eval()                           # Put model in evaluation mode
    with torch.no_grad():                  # Don't compute gradients (faster, less memory)
        val_reconstruction = model(val_tensor)
        val_loss = criterion(val_reconstruction, val_tensor).item()
    print(f"  val_loss: {val_loss:.6f}")
```

**`model.eval()`**: Some layers behave differently during training vs evaluation (e.g., Dropout). Always switch modes.

**`torch.no_grad()`**: Disables gradient computation. During validation we're only measuring, not learning.

### Step 9.7: Checkpointing

Save the model after every epoch (so you can resume if interrupted):

```python
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "train_loss": avg_loss,
        "val_loss": val_loss,
    }
    torch.save(checkpoint, checkpoint_dir / "last.pt")

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(checkpoint, checkpoint_dir / "best.pt")
```

**`model.state_dict()`**: A dictionary of all the model's learned weights. This is what you save and load.

**`last.pt`**: Always overwritten. Used to resume interrupted training.
**`best.pt`**: Only overwritten when validation improves. Used for final inference.

### Step 9.8: Resuming from checkpoint

If `req.resume` is set, load the checkpoint before the training loop:

```python
if req.resume:
    checkpoint = torch.load(req.resume)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    start_epoch = checkpoint["epoch"] + 1
    best_val_loss = checkpoint["val_loss"]
else:
    start_epoch = 0
    best_val_loss = float("inf")
```

---

## 10. Evaluation and Threshold

This happens at the end of training (or as a separate inference step).

### Step 10.1: Computing reconstruction error per sample

```python
model.eval()
with torch.no_grad():
    test_reconstruction = model(test_tensor)
    # Per-sample MSE (not averaged across all samples)
    errors = ((test_tensor - test_reconstruction) ** 2).mean(dim=1)
```

`errors` is a 1D tensor — one number per test sample. Higher error = more abnormal.

### Step 10.2: Finding the threshold

Use validation data (which is all normal) to set the threshold:

```python
with torch.no_grad():
    val_reconstruction = model(val_tensor)
    val_errors = ((val_tensor - val_reconstruction) ** 2).mean(dim=1)

# Set threshold at 95th percentile of normal reconstruction error
threshold = torch.quantile(val_errors, 0.95).item()
```

**Why 95th percentile?** We're saying: "95% of normal data reconstructs below this error. Anything above it is suspicious." You can tune this number. Higher = fewer false alarms but might miss attacks. Lower = catches more attacks but more false alarms.

The `sus` column in validation data can help tune this:

```python
# Compare: do suspicious rows have higher reconstruction error than normal ones?
val_sus = val_labels["sus"].values
print(f"Error for sus=0: {val_errors[val_sus == 0].mean():.6f}")
print(f"Error for sus=1: {val_errors[val_sus == 1].mean():.6f}")
```

If suspicious rows have higher errors, the model is learning something useful.

### Step 10.3: Classifying test data

```python
predictions = (errors > threshold).int()  # 1 = anomaly, 0 = normal
```

### Step 10.4: Computing metrics

```python
from sklearn.metrics import classification_report, roc_auc_score

test_evil = test_labels["evil"].values
print(classification_report(test_evil, predictions.numpy()))
print(f"ROC-AUC: {roc_auc_score(test_evil, errors.numpy()):.4f}")
```

**What the metrics mean:**

| Metric | What it tells you |
|---|---|
| **Precision** | Of everything you flagged as attack, how many actually were? |
| **Recall** | Of all actual attacks, how many did you catch? |
| **F1** | Harmonic mean of precision and recall (balanced score) |
| **ROC-AUC** | How well the model separates attacks from normal, regardless of threshold. 1.0 = perfect, 0.5 = random guessing |

For security, **recall** matters most — you'd rather have some false alarms than miss real attacks.

---

## 11. Inference

**File to create:** `src/pipelines/cyber_attack_detection/inference/tasks/predict.py`

**Add `"predict"` to `inference/facade.py`'s TASKS list.**

Inference is about using the trained model on new, unseen data:

1. Load the saved model (`best.pt`), scaler, and encoders
2. Load new data
3. Apply the same preprocessing (clean → encode → scale) using the saved artifacts
4. Run through the model
5. Compare reconstruction error to the saved threshold
6. Output predictions

The key insight: **every preprocessing step must be identical to training**. Use the saved scaler and encoders, not new ones.

---

## 12. Putting It All Together

### The full pipeline flow

```
npm run cyber-attack-detection
│
├── download/          → Fetch BETH from Kaggle (or skip if marker exists)
├── preprocessing/     → data_analysis → cleaning → feature_engineering → encoding → scaling
├── models/            → Create autoencoder (input_dim from scaling task)
├── training/          → Train on normal data, save checkpoints
└── inference/         → Load best model, evaluate on test set, report metrics
```

### Order to implement (one task at a time)

1. **Run `npm run cyber-attack-detection` now** — data_analysis task will generate the report
2. Read the report at `~/python_venvs/datasets/kaggle/beth-dataset/reports/`
3. Write `cleaning.py` → uncomment in facade → run pipeline → check logs
4. Write `feature_engineering.py` → uncomment in facade → run pipeline → check logs
5. Write `encoding.py` → uncomment in facade → run pipeline → check logs
6. Write `scaling.py` → uncomment in facade → run pipeline → check logs
7. Write `autoencoder.py` in models → add to facade → run pipeline → check logs
8. Write `trainer.py` in training → add to facade → run pipeline → **watch it train**
9. Write `predict.py` in inference → add to facade → run full pipeline → **see your results**

**After each step, run the pipeline and check `runtime_logs/app.log`.** If something breaks, the error message will tell you exactly where.

### Key PyTorch concepts you'll learn along the way

| Step | New PyTorch concept |
|---|---|
| Model | `nn.Module`, `nn.Linear`, `nn.ReLU`, `nn.Sequential`, `forward()` |
| Training | `DataLoader`, `loss.backward()`, `optimizer.step()`, `zero_grad()` |
| Evaluation | `model.eval()`, `torch.no_grad()`, `state_dict()` |
| Checkpointing | `torch.save()`, `torch.load()`, `load_state_dict()` |
| Tensors | `torch.from_numpy()`, `.float()`, `.mean(dim=1)`, `.item()` |

### Hyperparameters to experiment with

Once it works end-to-end, try changing these and see what happens:

| Parameter | Default | Try | Effect |
|---|---|---|---|
| `hidden_dims` | `[64, 32]` | `[128, 64, 32]` | Deeper network, more capacity |
| `bottleneck_dim` | `16` | `8` or `32` | Tighter/looser compression |
| `lr` | `0.001` | `0.0001` or `0.01` | Slower/faster learning |
| `batch_size` | `256` | `512` or `64` | Larger = smoother, smaller = noisier |
| `num_epochs` | `50` | `100` | More training time |
| `threshold percentile` | `0.95` | `0.99` or `0.90` | Sensitivity tradeoff |

---

## Quick Reference: Where Each File Goes

| File | Location |
|---|---|
| `cleaning.py` | `src/pipelines/cyber_attack_detection/preprocessing/tasks/cleaning.py` |
| `feature_engineering.py` | `src/pipelines/cyber_attack_detection/preprocessing/tasks/feature_engineering.py` |
| `encoding.py` | `src/pipelines/cyber_attack_detection/preprocessing/tasks/encoding.py` |
| `scaling.py` | `src/pipelines/cyber_attack_detection/preprocessing/tasks/scaling.py` |
| `autoencoder.py` | `src/pipelines/cyber_attack_detection/models/tasks/autoencoder.py` |
| `trainer.py` | `src/pipelines/cyber_attack_detection/training/tasks/trainer.py` |
| `predict.py` | `src/pipelines/cyber_attack_detection/inference/tasks/predict.py` |

Every file follows the same pattern:

```python
from core.common.wfs.interfaces import WfTask
from core.common.wfs.dtos import WfReq, WfResp
from core.logger import get_logger

log = get_logger(__name__)

class YourClassName(WfTask):
    def execute(self, req: WfReq, resp: WfResp) -> WfResp:
        # Your code here
        # Read from: resp.ctx_data["key"]
        # Write to: resp.ctx_data["new_key"] = value
        # Log: log.debug("...")
        resp.message = "task completed"
        return resp

Task = YourClassName
```

**Don't forget the `Task = YourClassName` alias at the bottom — the facade uses it to load your class.**
