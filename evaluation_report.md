# BETH Dataset — Model Evaluation Report

> Generated: 2026-03-08T23:58:04.453795+00:00

## Architecture

`23` → `[64, 32]` → `16` → `[32, 64]` → `23`

---

## Key Metrics

| Metric | Value |
|--------|-------|
| **Precision** | 0.0692 |
| **Recall** | 0.0032 |
| **F1 Score** | 0.0062 |
| **ROC-AUC** | 0.4020 |

---

## Confusion Matrix

| | Predicted Normal | Predicted Attack |
|---|---|---|
| **Actual Normal** | 23,643 | 6,892 (false alarms) |
| **Actual Attack** | 157,920 (missed) | 512 (caught) |

Total: 188,967 rows (158,432 attacks, 30,535 normal)

---

## Threshold Calibration

| | Value |
|---|---|
| **Threshold** | 399.998169 (p95) |
| **Val Error Range** | 0.043357 — 3509.480225 (median: 0.192716) |
| **Test Error Range** | 0.401279 — 2420.031738 (median: 77.249260) |

**Suspicious Row Sanity Check:**
- sus=1 mean error: 28.273914
- sus=0 mean error: 40.75581
- Ratio: 0.6937x — **No signal**

---

## Training History

- **Epochs:** 2
- **Learning Rate:** 0.001
- **Batch Size:** 256
- **Best Val Loss:** 40.703892

<details>
<summary>Epoch-by-epoch losses</summary>

| Epoch | Train Loss | Val Loss |
|-------|-----------|----------|
| 1 | 84.849133 | 40.7039 |
| 2 | 0.211539 | 42.8592 |

</details>

---

*BETH Dataset Evaluation Report — Generated programmatically by the predict task*
