# BETH Dataset — Model Evaluation Report

> Generated: 2026-03-09T00:10:53.486835+00:00

## Architecture

`23` -> `[64, 32]` -> `16` -> `[32, 64]` -> `23`

---

## Key Metrics at a Glance

| Metric | Value | Grade |
|--------|-------|-------|
| **Precision** | 0.0116 | Very poor |
| **Recall** | 0.0005 | Very poor |
| **F1 Score** | 0.0010 | Very poor |
| **ROC-AUC** | 0.4022 | Poor |

---

## Before We Dive In — Key Terms (No Jargon)

Our model looked at 188,967 network events and for each one said either **"this looks normal"** or **"this looks like an attack!"**. After it was done, we compared its guesses to the real answers and counted four things:

| Term | What It Means | Our Count |
|------|--------------|-----------|
| **True Positive (TP)** — Correctly caught | The model said *"Attack!"* and it really WAS an attack. **This is what we want most.** | **81** |
| **False Positive (FP)** — False alarm | The model said *"Attack!"* but it was actually normal traffic. Like a fire alarm going off when someone burns toast — annoying but not dangerous. | **6,901** |
| **False Negative (FN)** — Missed attack | The model said *"Looks normal"* but it was actually an attack. **This is the most dangerous mistake** — like a security guard letting a burglar walk right past. | **158,351** |
| **True Negative (TN)** — Correctly ignored | The model said *"Looks normal"* and it really WAS normal. Good — no unnecessary alarm. | **23,634** |

---

## What Each Metric Means — and How Our Model Did

### 1. Precision — "When the alarm goes off, is it real?"

**Formula:** Correctly caught attacks / All alarms raised = TP / (TP + FP)

Think of it like this: if a fire alarm rings 100 times, how many times was there actually a fire? If 90 out of 100 were real fires, precision = 0.90 (great!). If only 2 out of 100 were real, precision = 0.02 (terrible — 98% false alarms).

**Scale:** 0.0 (every alarm is fake) -> 1.0 (every alarm is a real attack)

> **Our result: 0.0116 (Very poor)** — Our model flagged 6,982 rows as attacks. Out of those, only 81 were real attacks and 6,901 were false alarms. That means 1.2% of the alarms were correct — almost every alarm is a false alarm. The security team would be overwhelmed with noise. This needs significant improvement.

### 2. Recall — "Of all real attacks, how many did we catch?"

**Formula:** Correctly caught attacks / All real attacks = TP / (TP + FN)

This is like asking: if 100 burglars tried to enter a building, how many did security actually stop? If security caught 95 out of 100, recall = 0.95. If they only caught 1, recall = 0.01 — 99 burglars got through.

**Scale:** 0.0 (caught nothing) -> 1.0 (caught every single attack)

> **Our result: 0.0005 (Very poor)** — There were 158,432 real attacks in the test set. Our model caught 81 of them and missed 158,351. That means it detected only 0.05% of the attacks — almost every attack slipped through undetected. This is the most critical problem — in cybersecurity, a missed attack can mean a data breach, ransomware infection, or full network compromise.

> **Why Recall matters more than Precision in cybersecurity:** A false alarm (FP) is annoying — a security analyst spends 5 minutes checking and says "never mind". A missed attack (FN) can mean ransomware encrypting your servers, customer data stolen, or attackers moving deeper into the network. In security, **we'd rather have 1,000 false alarms than miss 1 real attack**.

### 3. F1 Score — "Overall report card"

**Formula:** 2 x (Precision x Recall) / (Precision + Recall)

If a student gets 100% in math but 0% in English, their average (50%) doesn't tell the full story. F1 is similar — it's a special average that stays low unless *both* precision and recall are good. It punishes imbalance.

**Scale:** 0.0 (useless) -> 1.0 (perfect at both catching attacks AND avoiding false alarms)

> **Our result: 0.0010 (Very poor)** — F1 = 0.0010 (very poor). Both precision and recall are very low, so the combined score is near zero. The model is not yet useful as a detector — it needs more training or architectural changes.

### 4. ROC-AUC — "Does the model understand the difference at all?"

**Full name:** Area Under the Receiver Operating Characteristic Curve

All the metrics above depend on our chosen threshold (the cutoff line that decides "normal vs attack"). ROC-AUC ignores the threshold entirely and asks a deeper question: *if I pick one random attack and one random normal event, how often does the model give the attack a higher suspicion score?*

**Scale:** 0.5 (coin flip — model learned nothing) -> 1.0 (perfect — attack scores are always higher than normal scores)

> **Our result: 0.4022 (Poor)** — ROC-AUC = 0.4022. This is below 0.5, which means the model is doing WORSE than random guessing. A coin flip would perform better. The model may have learned inverted patterns (giving low error to attacks and high error to normal data). This is a strong sign that the model needs more training epochs, different architecture, or better features.

---

## Confusion Matrix

| | Model said "Normal" | Model said "Attack!" |
|---|---|---|
| **Actually Normal** | 23,634 (correct) | 6,901 (false alarms) |
| **Actually an Attack** | 158,351 (missed!) | 81 (caught!) |

Total: 188,967 rows (158,432 real attacks, 30,535 normal events)

---

## Threshold Calibration

The threshold is the cutoff line: any event with a reconstruction error above this number is flagged as an attack. We set it at the 95th percentile of validation errors — meaning only 5% of normal traffic would trigger a false alarm.

| | Value |
|---|---|
| **Threshold** | 363.057648 (p95) |
| **Val Error Range** | 0.014050 — 3215.280518 (median: 0.075999) |
| **Test Error Range** | 0.125771 — 3183.831787 (median: 46.708324) |

**Suspicious Row Sanity Check:**
- sus=1 mean error: 30.138968
- sus=0 mean error: 38.46986
- Ratio: 0.7834x — **No signal**

---

## Training History

- **Epochs:** 2
- **Learning Rate:** 0.001
- **Batch Size:** 256
- **Best Val Loss:** 38.435210

<details>
<summary>Epoch-by-epoch losses</summary>

| Epoch | Train Loss | Val Loss |
|-------|-----------|----------|
| 1 | 67.457176 | 39.1953 |
| 2 | 0.221892 | 38.4352 |

</details>

---

*BETH Dataset Evaluation Report — Generated programmatically by the predict task*
