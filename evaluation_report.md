# BETH Dataset — Model Evaluation Report

> Generated: 2026-03-09T00:02:52.877004+00:00

## Architecture

`23` → `[64, 32]` → `16` → `[32, 64]` → `23`

---

## Key Metrics

| Metric | Value |
|--------|-------|
| **Precision** | 0.0055 |
| **Recall** | 0.0002 |
| **F1 Score** | 0.0004 |
| **ROC-AUC** | 0.3839 |

---

## What Do These Metrics Mean?

| Metric | Formula | In Plain English (BETH context) | Ideal |
|--------|---------|--------------------------------|-------|
| **Precision** | TP / (TP + FP) | Of everything the model flagged as an attack, what fraction actually was? Low precision = your security team wastes time chasing false alarms. | 1.0 |
| **Recall** | TP / (TP + FN) | Of all real attacks in the data, what fraction did the model catch? **Most important for security** — a missed attack can mean a breach. Low recall = attacks slip through undetected. | 1.0 |
| **F1 Score** | 2 x (P x R) / (P + R) | Harmonic mean — a balanced "grade" combining both precision and recall. Only scores well when *both* are good. Think of it as a single number that says "how useful is this model overall?" | 1.0 |
| **ROC-AUC** | Area Under ROC Curve | Unlike the other three metrics (which depend on the chosen threshold p95), ROC-AUC measures the model's raw ability to rank attack rows higher than normal rows. 0.5 = random coin flip (useless), 1.0 = perfect separation. Tells you whether the model learned anything meaningful, regardless of threshold. | 1.0 |

> **Security tradeoff:** In cyber attack detection, **Recall > Precision**. Missing a real attack (false negative) is far more dangerous than a false alarm (false positive). A SOC analyst can dismiss a false alarm in seconds — but a missed intrusion can lead to data breach, ransomware, or lateral movement. When tuning the threshold, prioritise recall.

---

## Confusion Matrix

| | Predicted Normal | Predicted Attack |
|---|---|---|
| **Actual Normal** | 23,847 | 6,688 (false alarms) |
| **Actual Attack** | 158,395 (missed) | 37 (caught) |

Total: 188,967 rows (158,432 attacks, 30,535 normal)

---

## Threshold Calibration

| | Value |
|---|---|
| **Threshold** | 597.193176 (p95) |
| **Val Error Range** | 0.006648 — 3213.692139 (median: 0.045826) |
| **Test Error Range** | 0.195655 — 2111.081299 (median: 62.549549) |

**Suspicious Row Sanity Check:**
- sus=1 mean error: 31.370281
- sus=0 mean error: 46.565701
- Ratio: 0.6737x — **No signal**

---

## Training History

- **Epochs:** 2
- **Learning Rate:** 0.001
- **Batch Size:** 256
- **Best Val Loss:** 46.502502

<details>
<summary>Epoch-by-epoch losses</summary>

| Epoch | Train Loss | Val Loss |
|-------|-----------|----------|
| 1 | 96.625159 | 47.7224 |
| 2 | 0.212395 | 46.5025 |

</details>

---

*BETH Dataset Evaluation Report — Generated programmatically by the predict task*
