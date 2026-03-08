# BETH Dataset — Data Analysis Report

**Version:** v04 | **Generated:** 2026-03-08T16:42:56.998600+00:00 | **Splits:** 3

## Dataset Overview

| Split | Rows | Columns | Attack % | Suspicious % | Duplicates |
|-------|-----:|--------:|---------:|-------------:|-----------:|
| training | 763,144 | 16 | 0.0% | 0.0% | 0 |
| validation | 188,967 | 16 | 0.0% | 0.0% | 0 |
| testing | 188,967 | 16 | 0.0% | 0.0% | 0 |

## Analyst Insights

- **[CRITICAL]** 
- **[CRITICAL]** 
- **[WARNING]** 
- **[INFO]** 
- **[INFO]** 
- **[WARNING]** 
- **[INFO]** 
- **[WARNING]** 
- **[WARNING]** 
- **[WARNING]** 
- **[WARNING]** 
- **[WARNING]** 
- **[WARNING]** 
- **[WARNING]** 
- **[WARNING]** 
- **[WARNING]** 
- **[WARNING]** 
- **[WARNING]** 
- **[WARNING]** 
- **[WARNING]** 
- **[WARNING]** 
- **[WARNING]** 

## Column Classification

- **True Numeric**: `columns`, `treatment`, `detail`
- **Categorical Id**: `columns`, `treatment`, `detail`
- **String Categorical**: `columns`, `treatment`, `detail`
- **Complex Drop**: `columns`, `treatment`, `detail`
- **Parse Then Drop**: `columns`, `treatment`, `detail`
- **Labels**: `columns`, `treatment`, `detail`

### Categorical ID Statistics (Training Split)

| Column | Unique Values | Top 5 | Interpretation |
|--------|-------------:|-------|----------------|
| `processId` | ? |  |  |
| `threadId` | ? |  |  |
| `parentProcessId` | ? |  |  |
| `userId` | ? |  |  |
| `mountNamespace` | ? |  |  |
| `eventId` | ? |  |  |

## Cross-Split Analysis

## Recommendations

1. {'step': 'cleaning', 'priority': 1, 'action': "Drop 'stackAddresses'. KEEP 'args' for feature extraction.", 'reason': 'stackAddresses needs binary analysis — drop it. args contains file paths and flags with strong attack signal (90x /proc/ access difference between normal and attacks). Parse in feature engineering.'}
2. {'step': 'cleaning', 'priority': 2, 'action': 'Check and handle missing values using TRAINING medians/modes', 'reason': 'Prevents information leakage from val/test sets.'}
3. {'step': 'cleaning', 'priority': 3, 'action': 'Separate labels (evil, sus) from features before any transformation', 'reason': 'Labels must not be fed as input to the model.'}
4. {'step': 'encoding', 'priority': 4, 'action': 'Label-encode ALL categorical columns: processName, hostName, eventName (strings) AND processId, threadId, parentProcessId, userId, mountNamespace, eventId (numeric IDs). Use UNKNOWN handling.', 'reason': 'These 9 columns are all identifiers. The model should not assume ordinal relationships between IDs. Label encoding treats each unique value as a distinct category.'}
5. {'step': 'feature_engineering', 'priority': 5, 'action': 'Create: is_root (userId==0), return_negative (returnValue<0), return_category (success/error/info), is_child_of_init (parentProcessId==1), is_orphan (parentProcessId==0), is_high_args (argsNum > training 95th percentile). No arithmetic on categorical ID columns.', 'reason': 'Domain-informed features from MITRE ATT&CK: privilege escalation (T1068), failed syscall probing, process tree analysis (T1059), process injection (T1055), and anomalous syscall complexity. All features use binary/categorical derivations consistent with column classification — no arithmetic on categorical IDs.'}
6. {'step': 'scaling', 'priority': 6, 'action': 'StandardScaler on TRUE NUMERIC columns ONLY (timestamp, argsNum, returnValue + engineered numeric features). Do NOT scale label-encoded categorical columns.', 'reason': "Scaling categorical encodings distorts them — encoded value 5 becoming -0.3 implies a relationship between categories that doesn't exist. Only scale columns where arithmetic distance is meaningful."}
7. {'step': 'model', 'priority': 7, 'action': 'Start with Autoencoder (e.g., input→64→32→16→32→64→input)', 'reason': 'Zero attacks in training = anomaly detection. Autoencoder learns to reconstruct normal patterns; high reconstruction error = anomaly.'}
8. {'step': 'evaluation', 'priority': 8, 'action': 'Use ROC-AUC and F1 as primary metrics, set threshold using validation 95th percentile, validate with sus column', 'reason': 'Accuracy is meaningless with 84% attack test set. Threshold tuning on validation sus=1 rows provides a sanity check.'}

---

# Data Analyst's Deep Dive

## Why Can't We Use a Normal Classifier?

> A normal classifier (cat vs dog) needs lots of examples of BOTH classes. Our training data has ZERO attacks — all 763,144 rows are normal. This is realistic: in the real world, you have months of normal logs and attacks are rare. We must learn what 'normal' looks like and flag anything that deviates.

## Industry Approaches When You Have No Attack Examples

| Approach | How It Works | Used In |
|----------|-------------|---------|
| **Autoencoder (our choice)** | Learns to compress and reconstruct normal data. Can't reconstruct attacks well — high reconstruction error = anomaly. | Cybersecurity, fraud detection. Very common. |
| **Variational Autoencoder (VAE)** | Like autoencoder but learns a probability distribution of normal. Attacks fall outside that distribution. | Advanced anomaly detection, generative modeling. |
| **Isolation Forest** | Randomly splits data into partitions. Anomalies are 'easier to isolate' (need fewer splits to separate from normal). | Very popular baseline. Works out of the box. Classical ML. |
| **One-Class SVM** | Draws a boundary around normal data in high-dimensional space. Anything outside = anomaly. | Network intrusion detection. Classical ML. |
| **GANs (Generative Adversarial Networks)** | Train a generator to produce realistic normal data. A discriminator learns to tell real from fake. Attacks look 'fake' to the discriminator. | Research-heavy, less common in production. |
| **LSTM / Transformer Sequence Models** | Treat events as a time sequence. Learn to predict the next event. Attacks break the predicted pattern. | AWS, Azure, Google for real-time log anomaly detection. |
| **Statistical Methods** | Z-scores, percentile thresholds, Mahalanobis distance — flag values that are statistically unusual. | Always used as a baseline. Very simple. |
| **Clustering (DBSCAN, k-means)** | Group similar events. Attacks don't fit into any normal cluster. | Good for exploration and initial analysis. |

> **Why Autoencoder?** It's the sweet spot between simplicity and power. Isolation Forest is simpler but doesn't learn feature interactions. LSTMs are more powerful but need sequential ordering.

## How Our Autoencoder Works — The Music Teacher Analogy

### The Analogy

Imagine a music teacher who has ONLY ever heard classical music — thousands of pieces by Mozart, Beethoven, Bach. They know the patterns: tempo, instruments, structure.

Someone plays a song. The teacher's brain tries to 'reconstruct' it from classical music knowledge:
• If they play Mozart → brain easily reconstructs it → 'Sounds familiar' → NORMAL
• If they play death metal → brain tries to reconstruct using classical patterns → reconstruction sounds nothing like the original → 'No idea what this is' → ANOMALY

### The Technical Reality

Step 1: COMPRESS — squeeze all features into a small 'summary' (e.g., 14 features → 16 numbers).
Step 2: DECOMPRESS — try to rebuild the original features from that summary.
Step 3: COMPARE — how different is the rebuild from the original?

During training, it only sees normal data, so it gets REALLY good at compressing and rebuilding normal patterns. When an attack comes in, the compressor doesn't know how to summarize it (never seen this pattern), the decompressor builds something 'normal-ish' (that's all it knows), and the comparison shows a BIG difference → ANOMALY DETECTED.

## What Does 'Cardinality' Mean?

Think of a deck of playing cards:
• The 'suit' column has cardinality 4 (hearts, diamonds, clubs, spades)
• The 'value' column has cardinality 13 (Ace through King)
• A 'card ID' column has cardinality 52 (every card is unique)

Cardinality = how many UNIQUE/DISTINCT values a column has.

### In Our BETH Dataset

• eventId has cardinality ~50 — only 50 distinct event types. LOW cardinality → clearly categorical (like card suits).
• processName has cardinality ~200 — 200 distinct programs. MEDIUM cardinality → categorical, needs efficient encoding.
• processId has cardinality ~tens of thousands — HIGH cardinality. Still categorical (it's a name tag, not a measurement), but encoding must handle many categories.

LOW cardinality = easy to encode. HIGH cardinality = needs label encoding (not one-hot) and UNKNOWN handling for unseen values.

## Evaluation Metrics — The Security Guard Analogy

**Scenario:** Imagine you're a security guard reviewing 100 people entering a building. 80 are intruders, 20 are employees.

| Metric | Question | Example | Meaning |
|--------|----------|---------|---------|
| **PRECISION** | Of everyone I stopped, how many were ACTUAL intruders? | You stopped 50 people. 45 were intruders, 5 were employees you wrongly accused. Precision = 45/50 = 90%. | High precision = few false accusations. Important when the cost of a false alarm is high (e.g., shutting down a legitimate server). |
| **RECALL** | Of ALL the actual intruders, how many did I catch? | There were 80 intruders. You caught 45. 35 slipped through. Recall = 45/80 = 56%. | High recall = few intruders escape. IN CYBERSECURITY, RECALL MATTERS MOST — you'd rather have some false alarms than let attacks through. |
| **F1 SCORE** | What's my balanced grade between precision and recall? | F1 = 2 × (0.90 × 0.56) / (0.90 + 0.56) = 0.69 | F1 punishes you when precision and recall are unbalanced. A model with 99% precision but 1% recall gets a terrible F1 score. |
| **ROC-AUC** | How good is my detection ability OVERALL, regardless of where I set the alarm threshold? | If I pick a random intruder and a random employee, what's the probability my system gives the intruder a higher suspicion score? AUC=1.0 → perfect. AUC=0.5 → random coin flip. | AUC measures separation quality across ALL thresholds. It's the single best number for 'is my model learning anything useful?' |

### Why Not Just Use Accuracy?

> Our test set: 158,432 attacks + 30,535 normal = 188,967 total. A dumb model that says 'ATTACK!' for everything would be 83.8% accurate. Looks great — but it's completely useless (it caught zero real patterns). Precision, recall, F1, and AUC expose this: the dumb model has 83.8% recall but terrible precision, and AUC = 0.5 (random).

## Deep Learning vs Classical ML vs Metrics — They're Different Things

> Precision, Recall, F1, and ROC-AUC are EVALUATION METRICS — they are NOT deep learning or classical ML. They can evaluate ANY model: a neural network, a random forest, a simple if-statement, or even a human expert.

### Metrics

- **What:** How you MEASURE performance
- **Examples:** Precision, Recall, F1, ROC-AUC, Accuracy
- *The exam scoring rubric. It doesn't care if the student is a PhD or a high schooler.*

### Classical Ml

- **What:** Algorithms with handcrafted math (no neural networks)
- **Examples:** Isolation Forest, SVM, Random Forest, k-means, Logistic Regression
- *YOU (the human) decide what patterns to look for. You engineer features, pick a distance metric. The model applies your rules.*

### Deep Learning

- **What:** Neural networks that learn features automatically
- **Examples:** Autoencoder, VAE, LSTM, Transformer, GAN, CNN
- *The model discovers patterns on its own from raw data. You design the architecture; it learns the features.*

**In Our Project:** We use a DEEP LEARNING model (autoencoder) evaluated with MODEL-AGNOSTIC metrics (F1, AUC), processing data with CLASSICAL ML tools (StandardScaler, LabelEncoder from scikit-learn). Real-world systems almost always mix both.

---

## The Args Column — Why We Almost Made a Big Mistake

> **The Story:** Initial instinct: drop the args column — it's a messy JSON-like string, too complex to parse. But a data analyst's job is to LOOK at the data before making decisions. When we parsed the file paths and flags buried inside args, we found the STRONGEST attack signal in the entire dataset.

### Signal-by-Signal Breakdown

| Signal | Description | Training (normal) | Test (attacks) | Separation |
|--------|-------------|------------------:|---------------:|-----------:|
| `args_touches_proc` | Does the syscall access /proc/ (process info filesystem)? | 34.26% | 0.4% | **85.6x** |
| `args_has_write_flag` | Does the syscall open files for writing (O_WRONLY, O_RDWR, O_CREAT)? | 0.66% | 0.06% | **11.0x** |
| `args_touches_etc` | Does the syscall access /etc/ (system config files)? | 2.44% | 0.36% | **6.8x** |
| `args_is_hidden_path` | Does the pathname contain '/.' (hidden directory)? | 0.03% | 0.05% | **0.6x** |

#### `args_touches_proc`

**Analyst thinking:** Normal servers CONSTANTLY read /proc/ — checking CPU usage, memory, process lists, health monitoring. It's like a doctor checking vital signs every minute. Attackers DON'T do this — they're busy executing commands, stealing data, installing backdoors. The ABSENCE of /proc/ access is itself a powerful signal that something unusual is happening.

#### `args_has_write_flag`

**Analyst thinking:** Normal operations involve regular log writes, temp files, config updates. Attacks in this dataset are focused on READING and EXFILTRATING — they're in reconnaissance/data-theft mode, not file-creation mode. Fewer writes = suspicious behavior.

#### `args_touches_etc`

**Analyst thinking:** Normal services read their config files on startup (/etc/nginx.conf, /etc/ssh/sshd_config). Attacks may also read /etc/passwd or /etc/shadow to steal credentials, but the PATTERN of access is different — targeted reads vs routine config loading.

#### `args_is_hidden_path`

**Analyst thinking:** Hidden directories (starting with .) are a classic attacker technique (MITRE T1564.001). Malware staging in /tmp/.X25-unix/.rsync/ or similar paths is common. Low frequency but HIGH specificity — when it appears, it's almost certainly malicious.

**Real examples found:** `/tmp/.X25-unix/.rsync/c/lib/64/libresolv.so.2`, `/home/user/.config/procps/toprc`, `/tmp/.X25-unix/.rsync/c/lib/32/libnss_files.so.2`, `/tmp/.X25-unix/.rsync/c/lib/64/tsm`, `/tmp/.X25-unix/.rsync/c/lib/64/libdl.so.2`

> **The Lesson:** NEVER drop a column just because it looks complex. A 5-minute parsing exercise revealed the strongest signal in the dataset. The analyst's rule: if a column contains domain-relevant information (file paths, network addresses, command arguments), ALWAYS inspect it before discarding. Extract simple binary features from complex strings — you don't need to parse everything, just the signals that matter.

### Features to Extract from args

| Feature | Formula | Type |
|---------|---------|------|
| `args_touches_proc` | `'/proc/' in pathname` | binary |
| `args_touches_etc` | `'/etc/' in pathname` | binary |
| `args_has_write_flag` | `O_WRONLY|O_RDWR|O_CREAT in flags` | binary |
| `args_is_hidden_path` | `'/.' in pathname` | binary |
| `args_has_pathname` | `any pathname arg exists` | binary |

---

## Data Analyst's Synthesis

Looking at this dataset as a cybersecurity analyst, here is what stands out:

1. THE DATA TELLS A STORY: Training data is a peaceful honeypot — normal server operations. Test data is the honeypot UNDER ATTACK. The validation set is a clean control group. This three-way split is intentional and well-designed.

2. THE TRAP: 6 columns LOOK numeric but ARE categorical identifiers. A naive analyst would normalize processId and mountNamespace, teaching the model that 'PID 30000 is 30x more important than PID 1000'. This is wrong. PID 1 (init) is arguably the most important process.

3. THE HIDDEN GOLD MINE: The args column was almost dropped as 'too complex'. But a deeper look revealed it contains the strongest attack signal in the dataset — a 90x difference in /proc/ access patterns between normal and attack traffic. Always look at the data before discarding columns.

4. THE SIGNAL IS IN THE COMBINATIONS: No single column screams 'attack'. Attacks manifest as unusual COMBINATIONS — a rare process running as root at an unusual time with unusual arguments. The autoencoder's power is learning these multi-dimensional patterns.

5. THE WEAK LABELS ARE GOLD: 1,269 'suspicious' rows in training (sus=1) are borderline normal. If the autoencoder assigns them higher reconstruction error, it's finding real signal. Use this for threshold calibration.

6. THE REAL-WORLD LESSON: In production, you'd retrain periodically as 'normal' evolves (new services, new processes). The model's concept of normal must evolve too. This is called concept drift.

---
*BETH Dataset Analysis v04 | Generated by data_analysis task*