"""Threshold calibration, test evaluation, and artifact saving.

Combines Steps 9, 10, and 11 from the development plan into a single task.
All parameters (threshold percentile, report paths) come from YAML config.

For the BETH dataset — what happens during evaluation:
-------------------------------------------------------

Step 9 — Threshold Calibration:
  The trained autoencoder was trained on 763K NORMAL rows.  Now we need a
  cutoff: "how much reconstruction error is too much?"

  We feed the VALIDATION set (188K rows, all normal, but 1,269 are sus=1)
  through the model and compute per-sample MSE.  The 95th percentile of
  these errors becomes our threshold — meaning 5% of NORMAL data would be
  flagged as anomalous.  This is a tunable tradeoff:
    - Higher percentile (99%) → fewer false alarms, might miss subtle attacks
    - Lower percentile (90%) → catches more attacks, more false alarms

  Sanity check: if sus=1 rows have higher mean error than sus=0 rows, the
  model is learning meaningful patterns even within normal data.

Step 10 — Test Evaluation:
  Feed the TEST set (188K rows, ~84% attacks) through the model.  For each
  row, compute MSE → compare to threshold → predict 0 (normal) or 1 (attack).

  Metrics computed:
    Precision — of rows flagged as attack, how many actually are? (false alarm rate)
    Recall    — of actual attacks, how many did we catch? (MOST IMPORTANT for security)
    F1 Score  — harmonic mean of precision and recall (balanced grade)
    ROC-AUC   — how well errors separate attacks from normal (1.0=perfect, 0.5=random)

Step 11 — Artifact Saving:
  Save threshold, metrics, and evaluation report (JSON + HTML) alongside the
  model checkpoints.  Everything needed to deploy the model in production.

Reads from ctx_data: model, model_device, val_X, test_X,
                     val_labels, test_labels
Writes to ctx_data:  threshold, predictions, metrics, evaluation_report_path
"""

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from core.common.wfs.dtos import WfReq, WfResp
from core.common.wfs.interfaces import WfTask
from core.config import get_cfg
from core.logger import get_logger

log = get_logger(__name__)


class Predict(WfTask):

    def execute(self, req: WfReq, resp: WfResp) -> WfResp:
        cfg = req.config

        # ---------- Retrieve model and data from ctx_data ----------------------
        model: nn.Module | None = resp.ctx_data.get("model")
        if model is None:
            resp.success = False
            resp.message = "predict requires model in ctx_data — run trainer first"
            return resp

        device_str = resp.ctx_data.get("model_device", "cpu")
        device = torch.device(device_str)

        # For BETH: val_X (188967, 23), test_X (188967, 23)
        val_X: np.ndarray | None = resp.ctx_data.get("val_X")
        test_X: np.ndarray | None = resp.ctx_data.get("test_X")
        val_labels: np.ndarray | None = resp.ctx_data.get("val_labels")
        test_labels: np.ndarray | None = resp.ctx_data.get("test_labels")

        if val_X is None or test_X is None:
            resp.success = False
            resp.message = "predict requires val_X and test_X in ctx_data"
            return resp

        # Threshold percentile from YAML config (default 95th)
        threshold_pct = get_cfg(cfg, "evaluation.threshold_percentile", 95)

        # Where to save the evaluation report
        artifact_dir = Path(get_cfg(
            cfg, "dataset.paths.artifact_dir",
            str(Path(get_cfg(cfg, "dataset.paths.data_dir", ".")) / "artifacts"),
        ))
        report_dir = Path(get_cfg(cfg, "dataset.paths.report_dir", str(artifact_dir)))

        # ======================================================================
        # STEP 9: THRESHOLD CALIBRATION
        # ======================================================================
        log.debug("Step 9: Threshold calibration on validation set")

        # Compute per-sample reconstruction error on validation set.
        # Each error = mean squared difference between input and output
        # across all 23 features for that row.
        val_errors = _compute_reconstruction_errors(model, val_X, device)

        # The threshold is the N-th percentile of validation errors.
        # For BETH: 95th percentile means "an error higher than 95% of normal
        # validation rows is considered anomalous."
        threshold = float(np.percentile(val_errors, threshold_pct))
        log.debug("Threshold (p%d): %.6f", threshold_pct, threshold)
        log.debug("Val errors: min=%.6f, median=%.6f, mean=%.6f, max=%.6f",
                  val_errors.min(), np.median(val_errors),
                  val_errors.mean(), val_errors.max())

        # --- Sanity check: do sus=1 rows have higher error than sus=0? ---
        # The validation set has 1,269 sus=1 rows (suspicious but not evil).
        # If the model learned real patterns, these borderline rows should
        # reconstruct slightly worse than fully normal (sus=0) rows.
        sus_analysis = {}
        if val_labels is not None and val_labels.shape[1] >= 2:
            # val_labels columns: [evil, sus]
            sus_col = val_labels[:, 1]
            sus_mask = sus_col == 1
            normal_mask = sus_col == 0

            if sus_mask.any() and normal_mask.any():
                sus_mean = float(val_errors[sus_mask].mean())
                normal_mean = float(val_errors[normal_mask].mean())
                ratio = sus_mean / normal_mean if normal_mean > 0 else 0
                sus_analysis = {
                    "sus_mean_error": round(sus_mean, 6),
                    "normal_mean_error": round(normal_mean, 6),
                    "ratio": round(ratio, 4),
                    "signal_detected": ratio > 1.0,
                }
                marker = "YES — model detects signal" if ratio > 1.0 else "NO — no signal"
                log.debug("Sus check: sus_mean=%.6f, normal_mean=%.6f, ratio=%.4f → %s",
                          sus_mean, normal_mean, ratio, marker)

        # ======================================================================
        # STEP 10: TEST EVALUATION
        # ======================================================================
        log.debug("Step 10: Test evaluation")

        # Compute reconstruction errors for the test set (~84% attacks).
        test_errors = _compute_reconstruction_errors(model, test_X, device)

        # Apply threshold: error > threshold → predict attack (1), else normal (0)
        predictions = (test_errors > threshold).astype(int)

        # Extract true labels (evil column = first column of test_labels)
        if test_labels is not None and test_labels.shape[1] >= 1:
            y_true = test_labels[:, 0].astype(int)
        else:
            resp.success = False
            resp.message = "test_labels missing or has no evil column"
            return resp

        # --- Compute metrics ---
        precision = float(precision_score(y_true, predictions, zero_division=0))
        recall = float(recall_score(y_true, predictions, zero_division=0))
        f1 = float(f1_score(y_true, predictions, zero_division=0))
        roc_auc = float(roc_auc_score(y_true, test_errors))

        # Classification report for detailed breakdown
        cls_report = classification_report(y_true, predictions,
                                           target_names=["normal", "attack"],
                                           output_dict=True)

        log.debug("Test metrics:")
        log.debug("  Precision: %.4f", precision)
        log.debug("  Recall:    %.4f", recall)
        log.debug("  F1 Score:  %.4f", f1)
        log.debug("  ROC-AUC:   %.4f", roc_auc)

        total_attacks = int(y_true.sum())
        total_normal = int((y_true == 0).sum())
        predicted_attacks = int(predictions.sum())
        true_positives = int(((predictions == 1) & (y_true == 1)).sum())
        false_positives = int(((predictions == 1) & (y_true == 0)).sum())
        false_negatives = int(((predictions == 0) & (y_true == 1)).sum())
        true_negatives = int(((predictions == 0) & (y_true == 0)).sum())

        log.debug("  Confusion: TP=%d, FP=%d, FN=%d, TN=%d",
                  true_positives, false_positives, false_negatives, true_negatives)

        metrics = {
            "precision": round(precision, 6),
            "recall": round(recall, 6),
            "f1_score": round(f1, 6),
            "roc_auc": round(roc_auc, 6),
            "threshold": round(threshold, 6),
            "threshold_percentile": threshold_pct,
            "confusion_matrix": {
                "true_positives": true_positives,
                "false_positives": false_positives,
                "false_negatives": false_negatives,
                "true_negatives": true_negatives,
            },
            "test_set": {
                "total_rows": len(y_true),
                "actual_attacks": total_attacks,
                "actual_normal": total_normal,
                "predicted_attacks": predicted_attacks,
            },
        }

        # ======================================================================
        # STEP 11: SAVE ARTIFACTS AND REPORT
        # ======================================================================
        log.debug("Step 11: Saving evaluation artifacts")

        dataset_name = get_cfg(cfg, "dataset.name", "Dataset")
        report = {
            "dataset_name": dataset_name,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "model_architecture": {
                "hidden_dims": get_cfg(cfg, "model.hidden_dims", []),
                "bottleneck_dim": get_cfg(cfg, "model.bottleneck_dim", 0),
                "input_dim": resp.ctx_data.get("input_dim"),
            },
            "training": {
                "epochs": get_cfg(cfg, "training.num_epochs", 0),
                "lr": get_cfg(cfg, "training.lr", 0),
                "batch_size": get_cfg(cfg, "training.batch_size", 0),
                "best_val_loss": resp.ctx_data.get("best_val_loss"),
                "train_losses": resp.ctx_data.get("train_losses", []),
                "val_losses": resp.ctx_data.get("val_losses", []),
            },
            "threshold_calibration": {
                "threshold": round(threshold, 6),
                "percentile": threshold_pct,
                "val_error_stats": {
                    "min": round(float(val_errors.min()), 6),
                    "median": round(float(np.median(val_errors)), 6),
                    "mean": round(float(val_errors.mean()), 6),
                    "p95": round(float(np.percentile(val_errors, 95)), 6),
                    "p99": round(float(np.percentile(val_errors, 99)), 6),
                    "max": round(float(val_errors.max()), 6),
                },
                "sus_analysis": sus_analysis,
            },
            "test_evaluation": {
                "metrics": metrics,
                "classification_report": cls_report,
                "test_error_stats": {
                    "min": round(float(test_errors.min()), 6),
                    "median": round(float(np.median(test_errors)), 6),
                    "mean": round(float(test_errors.mean()), 6),
                    "p95": round(float(np.percentile(test_errors, 95)), 6),
                    "max": round(float(test_errors.max()), 6),
                },
            },
        }

        # Save JSON report
        report_dir.mkdir(parents=True, exist_ok=True)
        json_path = report_dir / "evaluation_report.json"
        json_path.write_text(json.dumps(report, indent=2, default=str))
        log.debug("Evaluation report saved to %s", json_path)

        # Save HTML report
        html = _render_evaluation_html(report)
        html_path = report_dir / "evaluation_report.html"
        html_path.write_text(html)
        log.debug("Evaluation HTML saved to %s", html_path)

        # Save Markdown report (viewable directly on GitHub)
        md = _render_evaluation_md(report)
        md_path = report_dir / "evaluation_report.md"
        md_path.write_text(md)
        log.debug("Evaluation MD saved to %s", md_path)

        # Copy to repo root for easy GitHub viewing
        if get_cfg(cfg, "report.copy_to_repo_root", False):
            repo_root = Path(__file__).resolve()
            for parent in repo_root.parents:
                if (parent / ".git").exists():
                    repo_root = parent
                    break
            (repo_root / "evaluation_report.html").write_text(html)
            (repo_root / "evaluation_report.md").write_text(md)
            log.debug("Evaluation reports copied to repo root")

        # Save threshold as a standalone artifact
        threshold_path = artifact_dir / "threshold.json"
        threshold_path.write_text(json.dumps({
            "threshold": threshold,
            "percentile": threshold_pct,
            "generated_at": datetime.now(timezone.utc).isoformat(),
        }, indent=2))

        # ---------- Publish to ctx_data ----------------------------------------
        resp.ctx_data["threshold"] = threshold
        resp.ctx_data["predictions"] = predictions
        resp.ctx_data["metrics"] = metrics
        resp.ctx_data["evaluation_report_path"] = str(json_path)

        resp.message = (
            f"Evaluation complete — Precision={precision:.4f}, Recall={recall:.4f}, "
            f"F1={f1:.4f}, ROC-AUC={roc_auc:.4f} (threshold={threshold:.4f} at p{threshold_pct})"
        )
        log.debug(resp.message)
        return resp


def _compute_reconstruction_errors(model: nn.Module, X: np.ndarray,
                                   device: torch.device,
                                   batch_size: int = 1024) -> np.ndarray:
    """Compute per-sample MSE between input and autoencoder reconstruction.

    For BETH: feeds data through the trained autoencoder in batches.
    For each row, computes mean((input - reconstruction)^2) across all
    23 features.  Returns an array of shape (n_samples,) where each
    value is the reconstruction error for that row.

    High error → the model couldn't reconstruct this row well → anomaly.
    """
    model.eval()
    errors_list: list[np.ndarray] = []

    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            batch = torch.from_numpy(X[i:i + batch_size]).to(device)
            reconstruction = model(batch)
            # Per-sample MSE: mean squared error across features for each row
            mse = ((batch - reconstruction) ** 2).mean(dim=1)
            errors_list.append(mse.cpu().numpy())

    return np.concatenate(errors_list)


def _render_evaluation_html(report: dict) -> str:
    """Render a self-contained HTML evaluation report."""
    metrics = report["test_evaluation"]["metrics"]
    cm = metrics["confusion_matrix"]
    thresh = report["threshold_calibration"]
    training = report["training"]
    sus = thresh.get("sus_analysis", {})
    test_stats = report["test_evaluation"]["test_error_stats"]
    val_stats = thresh["val_error_stats"]
    ds_name = report.get("dataset_name", "Dataset")
    arch = report.get("model_architecture", {})

    # Determine quality color for each metric
    def metric_color(val: float, good: float = 0.8, mid: float = 0.6) -> str:
        if val >= good:
            return "#28a745"
        if val >= mid:
            return "#ffc107"
        return "#dc3545"

    sus_html = ""
    if sus:
        signal_color = "#28a745" if sus.get("signal_detected") else "#dc3545"
        signal_text = "Signal detected" if sus.get("signal_detected") else "No signal"
        sus_html = f"""
        <div style="background:{signal_color}10;border-left:4px solid {signal_color};
                    padding:12px;margin:16px 0;border-radius:4px">
            <strong>Suspicious Row Sanity Check:</strong>
            sus=1 mean error: {sus.get('sus_mean_error', '?')},
            sus=0 mean error: {sus.get('normal_mean_error', '?')},
            ratio: {sus.get('ratio', '?')}x →
            <strong style="color:{signal_color}">{signal_text}</strong>
        </div>"""

    # Loss chart data
    train_losses = training.get("train_losses", [])
    val_losses = training.get("val_losses", [])
    loss_rows = ""
    for i, (tl, vl) in enumerate(zip(train_losses, val_losses)):
        loss_rows += f"<tr><td>{i+1}</td><td>{tl:.6f}</td><td>{vl:.4f}</td></tr>\n"

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>{ds_name} — Evaluation Report</title>
<style>
  body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
         max-width: 960px; margin: 40px auto; padding: 0 20px; color: #333; line-height:1.6; }}
  h1 {{ border-bottom: 3px solid #007bff; padding-bottom: 8px; }}
  h2 {{ color: #007bff; margin-top: 32px; }}
  table {{ border-collapse: collapse; width: 100%; margin: 16px 0; }}
  th, td {{ border: 1px solid #ddd; padding: 8px 12px; text-align: left; }}
  th {{ background: #f8f9fa; }}
  .metric-card {{ display: inline-block; background: #f8f9fa; border-radius: 8px;
                  padding: 16px 24px; margin: 8px; text-align: center; min-width: 140px; }}
  .metric-val {{ font-size: 2em; font-weight: bold; }}
  .metric-label {{ font-size: 0.85em; color: #666; }}
</style>
</head>
<body>

<h1>{ds_name} — Model Evaluation Report</h1>
<p>Generated: {report.get('generated_at', '')}</p>
<p>Architecture: {arch.get('input_dim', '?')} → {arch.get('hidden_dims', [])} →
   {arch.get('bottleneck_dim', '?')} → {list(reversed(arch.get('hidden_dims', [])))} →
   {arch.get('input_dim', '?')}</p>

<h2>Key Metrics</h2>
<div>
  <div class="metric-card">
    <div class="metric-val" style="color:{metric_color(metrics['precision'])}">{metrics['precision']:.4f}</div>
    <div class="metric-label">Precision</div>
  </div>
  <div class="metric-card">
    <div class="metric-val" style="color:{metric_color(metrics['recall'])}">{metrics['recall']:.4f}</div>
    <div class="metric-label">Recall</div>
  </div>
  <div class="metric-card">
    <div class="metric-val" style="color:{metric_color(metrics['f1_score'])}">{metrics['f1_score']:.4f}</div>
    <div class="metric-label">F1 Score</div>
  </div>
  <div class="metric-card">
    <div class="metric-val" style="color:{metric_color(metrics['roc_auc'])}">{metrics['roc_auc']:.4f}</div>
    <div class="metric-label">ROC-AUC</div>
  </div>
</div>

<h2>What Do These Metrics Mean?</h2>
<table>
  <tr>
    <th style="width:130px">Metric</th>
    <th>Definition</th>
    <th>In Plain English (BETH context)</th>
    <th style="width:80px">Ideal</th>
  </tr>
  <tr>
    <td><strong>Precision</strong></td>
    <td>TP / (TP + FP)<br><em>Of everything the model flagged as an attack, what fraction actually was?</em></td>
    <td>If the model raises 100 alarms, precision tells you how many are real attacks
        vs false alarms. Low precision = your security team wastes time chasing ghosts.</td>
    <td>1.0</td>
  </tr>
  <tr>
    <td><strong>Recall</strong></td>
    <td>TP / (TP + FN)<br><em>Of all real attacks in the data, what fraction did the model catch?</em></td>
    <td>If there are 1,000 real cyber attacks, recall tells you how many the model
        detected. <strong>This is the most important metric for security</strong> —
        a missed attack can mean a breach. Low recall = attacks slip through.</td>
    <td>1.0</td>
  </tr>
  <tr>
    <td><strong>F1 Score</strong></td>
    <td>2 &times; (Precision &times; Recall) / (Precision + Recall)<br>
        <em>Harmonic mean — a balanced "grade" combining both.</em></td>
    <td>If precision is high but recall is low (or vice versa), F1 will be low too.
        It only scores well when <em>both</em> precision and recall are good.
        Think of it as a single number that says "how useful is this model overall?"</td>
    <td>1.0</td>
  </tr>
  <tr>
    <td><strong>ROC-AUC</strong></td>
    <td>Area Under the Receiver Operating Characteristic curve<br>
        <em>How well can the model separate attacks from normal traffic at ANY threshold?</em></td>
    <td>Unlike the other three metrics which depend on our chosen threshold (p{thresh['percentile']}),
        ROC-AUC measures the model's raw ability to rank attack rows higher than normal rows.
        0.5 = random coin flip (useless), 1.0 = perfect separation.
        This tells you whether the model learned anything meaningful, regardless of
        where you set the alarm cutoff.</td>
    <td>1.0</td>
  </tr>
</table>
<div style="background:#e7f3fe;border-left:4px solid #2196F3;padding:12px;margin:16px 0;border-radius:4px">
  <strong>Security tradeoff:</strong> In cyber attack detection, <strong>Recall &gt; Precision</strong>.
  Missing a real attack (false negative) is far more dangerous than a false alarm (false positive).
  A SOC analyst can dismiss a false alarm in seconds — but a missed intrusion can lead to data
  breach, ransomware, or lateral movement. When tuning the threshold, prioritise recall.
</div>

<h2>Confusion Matrix</h2>
<table>
  <tr><th></th><th>Predicted Normal</th><th>Predicted Attack</th></tr>
  <tr><th>Actual Normal</th><td>{cm['true_negatives']:,}</td>
      <td style="color:#dc3545">{cm['false_positives']:,} (false alarms)</td></tr>
  <tr><th>Actual Attack</th>
      <td style="color:#dc3545">{cm['false_negatives']:,} (missed)</td>
      <td style="color:#28a745">{cm['true_positives']:,} (caught)</td></tr>
</table>
<p>Total: {metrics['test_set']['total_rows']:,} rows
   ({metrics['test_set']['actual_attacks']:,} attacks,
    {metrics['test_set']['actual_normal']:,} normal)</p>

<h2>Threshold Calibration</h2>
<table>
  <tr><th>Threshold</th><td>{thresh['threshold']:.6f} (p{thresh['percentile']})</td></tr>
  <tr><th>Val Error Range</th>
      <td>{val_stats['min']:.6f} — {val_stats['max']:.6f} (median: {val_stats['median']:.6f})</td></tr>
  <tr><th>Test Error Range</th>
      <td>{test_stats['min']:.6f} — {test_stats['max']:.6f} (median: {test_stats['median']:.6f})</td></tr>
</table>
{sus_html}

<h2>Training History</h2>
<p>{training.get('epochs', 0)} epochs, lr={training.get('lr', '?')},
   batch_size={training.get('batch_size', '?')},
   best_val_loss={training.get('best_val_loss', '?'):.6f}</p>
<details>
<summary>Show epoch-by-epoch losses</summary>
<table>
  <tr><th>Epoch</th><th>Train Loss</th><th>Val Loss</th></tr>
  {loss_rows}
</table>
</details>

<footer style="margin-top:40px;padding-top:16px;border-top:1px solid #ddd;
               color:#999;font-size:0.85em">
    {ds_name} Evaluation Report &nbsp;|&nbsp; Generated programmatically by the predict task
</footer>

</body>
</html>"""


def _render_evaluation_md(report: dict) -> str:
    """Render a Markdown evaluation report (viewable on GitHub)."""
    metrics = report["test_evaluation"]["metrics"]
    cm = metrics["confusion_matrix"]
    thresh = report["threshold_calibration"]
    training = report["training"]
    sus = thresh.get("sus_analysis", {})
    test_stats = report["test_evaluation"]["test_error_stats"]
    val_stats = thresh["val_error_stats"]
    ds_name = report.get("dataset_name", "Dataset")
    arch = report.get("model_architecture", {})

    hidden = arch.get("hidden_dims", [])
    reversed_hidden = list(reversed(hidden))
    input_dim = arch.get("input_dim", "?")
    bottleneck = arch.get("bottleneck_dim", "?")

    sus_md = ""
    if sus:
        signal = "Signal detected" if sus.get("signal_detected") else "No signal"
        sus_md = (
            f"\n**Suspicious Row Sanity Check:**\n"
            f"- sus=1 mean error: {sus.get('sus_mean_error', '?')}\n"
            f"- sus=0 mean error: {sus.get('normal_mean_error', '?')}\n"
            f"- Ratio: {sus.get('ratio', '?')}x — **{signal}**\n"
        )

    loss_rows = ""
    train_losses = training.get("train_losses", [])
    val_losses = training.get("val_losses", [])
    for i, (tl, vl) in enumerate(zip(train_losses, val_losses)):
        loss_rows += f"| {i+1} | {tl:.6f} | {vl:.4f} |\n"

    best_val = training.get("best_val_loss")
    best_val_str = f"{best_val:.6f}" if best_val is not None else "?"

    return f"""# {ds_name} — Model Evaluation Report

> Generated: {report.get('generated_at', '')}

## Architecture

`{input_dim}` → `{hidden}` → `{bottleneck}` → `{reversed_hidden}` → `{input_dim}`

---

## Key Metrics

| Metric | Value |
|--------|-------|
| **Precision** | {metrics['precision']:.4f} |
| **Recall** | {metrics['recall']:.4f} |
| **F1 Score** | {metrics['f1_score']:.4f} |
| **ROC-AUC** | {metrics['roc_auc']:.4f} |

---

## What Do These Metrics Mean?

| Metric | Formula | In Plain English (BETH context) | Ideal |
|--------|---------|--------------------------------|-------|
| **Precision** | TP / (TP + FP) | Of everything the model flagged as an attack, what fraction actually was? Low precision = your security team wastes time chasing false alarms. | 1.0 |
| **Recall** | TP / (TP + FN) | Of all real attacks in the data, what fraction did the model catch? **Most important for security** — a missed attack can mean a breach. Low recall = attacks slip through undetected. | 1.0 |
| **F1 Score** | 2 x (P x R) / (P + R) | Harmonic mean — a balanced "grade" combining both precision and recall. Only scores well when *both* are good. Think of it as a single number that says "how useful is this model overall?" | 1.0 |
| **ROC-AUC** | Area Under ROC Curve | Unlike the other three metrics (which depend on the chosen threshold p{thresh['percentile']}), ROC-AUC measures the model's raw ability to rank attack rows higher than normal rows. 0.5 = random coin flip (useless), 1.0 = perfect separation. Tells you whether the model learned anything meaningful, regardless of threshold. | 1.0 |

> **Security tradeoff:** In cyber attack detection, **Recall > Precision**. Missing a real attack (false negative) is far more dangerous than a false alarm (false positive). A SOC analyst can dismiss a false alarm in seconds — but a missed intrusion can lead to data breach, ransomware, or lateral movement. When tuning the threshold, prioritise recall.

---

## Confusion Matrix

| | Predicted Normal | Predicted Attack |
|---|---|---|
| **Actual Normal** | {cm['true_negatives']:,} | {cm['false_positives']:,} (false alarms) |
| **Actual Attack** | {cm['false_negatives']:,} (missed) | {cm['true_positives']:,} (caught) |

Total: {metrics['test_set']['total_rows']:,} rows ({metrics['test_set']['actual_attacks']:,} attacks, {metrics['test_set']['actual_normal']:,} normal)

---

## Threshold Calibration

| | Value |
|---|---|
| **Threshold** | {thresh['threshold']:.6f} (p{thresh['percentile']}) |
| **Val Error Range** | {val_stats['min']:.6f} — {val_stats['max']:.6f} (median: {val_stats['median']:.6f}) |
| **Test Error Range** | {test_stats['min']:.6f} — {test_stats['max']:.6f} (median: {test_stats['median']:.6f}) |
{sus_md}
---

## Training History

- **Epochs:** {training.get('epochs', 0)}
- **Learning Rate:** {training.get('lr', '?')}
- **Batch Size:** {training.get('batch_size', '?')}
- **Best Val Loss:** {best_val_str}

<details>
<summary>Epoch-by-epoch losses</summary>

| Epoch | Train Loss | Val Loss |
|-------|-----------|----------|
{loss_rows}
</details>

---

*{ds_name} Evaluation Report — Generated programmatically by the predict task*
"""


Task = Predict
