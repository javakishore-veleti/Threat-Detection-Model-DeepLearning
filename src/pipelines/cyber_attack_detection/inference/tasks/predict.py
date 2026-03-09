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


def _metric_grade(val: float) -> str:
    """Return a human-readable grade word for a 0-1 metric value."""
    if val >= 0.9:
        return "Excellent"
    if val >= 0.8:
        return "Good"
    if val >= 0.6:
        return "Moderate"
    if val >= 0.3:
        return "Poor"
    return "Very poor"


def _metric_color(val: float, good: float = 0.8, mid: float = 0.6) -> str:
    if val >= good:
        return "#28a745"
    if val >= mid:
        return "#ffc107"
    return "#dc3545"


def _build_dynamic_commentary(metrics: dict, cm: dict) -> dict:
    """Build dynamic, plain-English commentary for each metric based on actual values."""
    p = metrics["precision"]
    r = metrics["recall"]
    f1 = metrics["f1_score"]
    auc = metrics["roc_auc"]
    tp = cm["true_positives"]
    fp = cm["false_positives"]
    fn = cm["false_negatives"]
    total_attacks = metrics["test_set"]["actual_attacks"]
    predicted_attacks = metrics["test_set"]["predicted_attacks"]

    p_pct = p * 100
    r_pct = r * 100

    precision_commentary = (
        f"Our model flagged {predicted_attacks:,} rows as attacks. "
        f"Out of those, only {tp:,} were real attacks and {fp:,} were false alarms. "
        f"That means {p_pct:.1f}% of the alarms were correct — "
    )
    if p < 0.1:
        precision_commentary += (
            "almost every alarm is a false alarm. The security team would be "
            "overwhelmed with noise. This needs significant improvement."
        )
    elif p < 0.5:
        precision_commentary += "less than half of the alarms are real. Still too noisy for production."
    elif p < 0.8:
        precision_commentary += "more than half are real. Getting useful but still noisy."
    else:
        precision_commentary += "most alarms are real attacks. Very useful for a security team."

    recall_commentary = (
        f"There were {total_attacks:,} real attacks in the test set. "
        f"Our model caught {tp:,} of them and missed {fn:,}. "
        f"That means it detected only {r_pct:.2f}% of the attacks — "
    )
    if r < 0.01:
        recall_commentary += (
            "almost every attack slipped through undetected. "
            "This is the most critical problem — in cybersecurity, a missed attack "
            "can mean a data breach, ransomware infection, or full network compromise."
        )
    elif r < 0.3:
        recall_commentary += "most attacks are being missed. Not safe for production."
    elif r < 0.7:
        recall_commentary += "catching some attacks but still missing too many."
    elif r < 0.9:
        recall_commentary += "catching most attacks. Approaching production-ready."
    else:
        recall_commentary += "catching nearly all attacks. Excellent for security use."

    f1_commentary = (
        f"F1 = {f1:.4f} ({_metric_grade(f1).lower()}). "
    )
    if f1 < 0.1:
        f1_commentary += (
            "Both precision and recall are very low, so the combined score is near zero. "
            "The model is not yet useful as a detector — it needs more training or architectural changes."
        )
    elif f1 < 0.5:
        f1_commentary += "Either precision or recall (or both) are dragging this down."
    else:
        f1_commentary += "Both precision and recall are contributing meaningfully."

    auc_commentary = f"ROC-AUC = {auc:.4f}. "
    if auc < 0.5:
        auc_commentary += (
            "This is below 0.5, which means the model is doing WORSE than random guessing. "
            "A coin flip would perform better. The model may have learned inverted patterns "
            "(giving low error to attacks and high error to normal data). "
            "This is a strong sign that the model needs more training epochs, "
            "different architecture, or better features."
        )
    elif auc < 0.6:
        auc_commentary += (
            "Barely above random guessing (0.5). The model has learned very little about "
            "what separates attacks from normal traffic."
        )
    elif auc < 0.8:
        auc_commentary += "The model has learned some patterns but isn't reliable yet."
    elif auc < 0.95:
        auc_commentary += "Good separation between attacks and normal traffic."
    else:
        auc_commentary += "Near-perfect separation. The model is very effective."

    return {
        "precision": precision_commentary,
        "recall": recall_commentary,
        "f1": f1_commentary,
        "auc": auc_commentary,
    }


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

    commentary = _build_dynamic_commentary(metrics, cm)

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
  h3 {{ color: #555; margin-top: 24px; }}
  table {{ border-collapse: collapse; width: 100%; margin: 16px 0; }}
  th, td {{ border: 1px solid #ddd; padding: 8px 12px; text-align: left; }}
  th {{ background: #f8f9fa; }}
  .metric-card {{ display: inline-block; background: #f8f9fa; border-radius: 8px;
                  padding: 16px 24px; margin: 8px; text-align: center; min-width: 140px; }}
  .metric-val {{ font-size: 2em; font-weight: bold; }}
  .metric-label {{ font-size: 0.85em; color: #666; }}
  .metric-grade {{ font-size: 0.75em; color: #999; }}
  .callout {{ padding: 12px 16px; margin: 16px 0; border-radius: 4px; }}
  .callout-blue {{ background: #e7f3fe; border-left: 4px solid #2196F3; }}
  .callout-red {{ background: #fde8e8; border-left: 4px solid #dc3545; }}
  .callout-green {{ background: #e8f5e9; border-left: 4px solid #28a745; }}
  .commentary {{ background: #fffbea; border-left: 4px solid #ffc107; padding: 12px 16px;
                 margin: 8px 0; border-radius: 4px; font-size: 0.95em; }}
</style>
</head>
<body>

<h1>{ds_name} — Model Evaluation Report</h1>
<p>Generated: {report.get('generated_at', '')}</p>
<p>Architecture: {arch.get('input_dim', '?')} &rarr; {arch.get('hidden_dims', [])} &rarr;
   {arch.get('bottleneck_dim', '?')} &rarr; {list(reversed(arch.get('hidden_dims', [])))} &rarr;
   {arch.get('input_dim', '?')}</p>

<h2>Key Metrics at a Glance</h2>
<div>
  <div class="metric-card">
    <div class="metric-val" style="color:{_metric_color(metrics['precision'])}">{metrics['precision']:.4f}</div>
    <div class="metric-label">Precision</div>
    <div class="metric-grade">{_metric_grade(metrics['precision'])}</div>
  </div>
  <div class="metric-card">
    <div class="metric-val" style="color:{_metric_color(metrics['recall'])}">{metrics['recall']:.4f}</div>
    <div class="metric-label">Recall</div>
    <div class="metric-grade">{_metric_grade(metrics['recall'])}</div>
  </div>
  <div class="metric-card">
    <div class="metric-val" style="color:{_metric_color(metrics['f1_score'])}">{metrics['f1_score']:.4f}</div>
    <div class="metric-label">F1 Score</div>
    <div class="metric-grade">{_metric_grade(metrics['f1_score'])}</div>
  </div>
  <div class="metric-card">
    <div class="metric-val" style="color:{_metric_color(metrics['roc_auc'])}">{metrics['roc_auc']:.4f}</div>
    <div class="metric-label">ROC-AUC</div>
    <div class="metric-grade">{_metric_grade(metrics['roc_auc'])}</div>
  </div>
</div>

<h2>Before We Dive In — Key Terms (No Jargon)</h2>

<p>Imagine our model looks at {metrics['test_set']['total_rows']:,} network events and
   for each one says either <strong>"this looks normal"</strong> or
   <strong>"this looks like an attack!"</strong>. After it's done, we compare
   its guesses to the real answers and count four things:</p>

<table>
  <tr>
    <th style="width:200px">Term</th>
    <th>What It Means</th>
    <th>Our Numbers</th>
  </tr>
  <tr style="background:#e8f5e9">
    <td><strong>True Positive (TP)</strong><br>Correctly caught</td>
    <td>The model said <em>"Attack!"</em> and it really WAS an attack. <strong>This is what we want most.</strong></td>
    <td style="font-size:1.2em"><strong>{cm['true_positives']:,}</strong></td>
  </tr>
  <tr style="background:#fde8e8">
    <td><strong>False Positive (FP)</strong><br>False alarm</td>
    <td>The model said <em>"Attack!"</em> but it was actually normal traffic.
        Like a fire alarm going off when someone burns toast — annoying, wastes time,
        but not dangerous.</td>
    <td style="font-size:1.2em"><strong>{cm['false_positives']:,}</strong></td>
  </tr>
  <tr style="background:#fde8e8">
    <td><strong>False Negative (FN)</strong><br>Missed attack</td>
    <td>The model said <em>"Looks normal"</em> but it was actually an attack.
        <strong>This is the most dangerous mistake</strong> — like a security guard
        letting a burglar walk right past. The attack goes undetected.</td>
    <td style="font-size:1.2em"><strong>{cm['false_negatives']:,}</strong></td>
  </tr>
  <tr style="background:#e8f5e9">
    <td><strong>True Negative (TN)</strong><br>Correctly ignored</td>
    <td>The model said <em>"Looks normal"</em> and it really WAS normal. Good — no
        unnecessary alarm.</td>
    <td style="font-size:1.2em"><strong>{cm['true_negatives']:,}</strong></td>
  </tr>
</table>

<h2>What Each Metric Means — and How Our Model Did</h2>

<h3>1. Precision — "When the alarm goes off, is it real?"</h3>
<p><strong>Formula:</strong> Correctly caught attacks / All alarms raised = TP / (TP + FP)</p>
<p>Think of it like this: if a fire alarm rings 100 times, how many times was there
   actually a fire? If 90 out of 100 were real fires, precision = 0.90 (great!).
   If only 2 out of 100 were real, precision = 0.02 (terrible — 98% false alarms).</p>
<p><strong>Scale:</strong> 0.0 (every alarm is fake) → 1.0 (every alarm is a real attack)</p>
<div class="commentary">
  <strong>Our result: {metrics['precision']:.4f} ({_metric_grade(metrics['precision'])})</strong><br>
  {commentary['precision']}
</div>

<h3>2. Recall — "Of all real attacks, how many did we catch?"</h3>
<p><strong>Formula:</strong> Correctly caught attacks / All real attacks = TP / (TP + FN)</p>
<p>This is like asking: if 100 burglars tried to enter a building, how many did
   security actually stop? If security caught 95 out of 100, recall = 0.95. If they
   only caught 1, recall = 0.01 — 99 burglars got through.</p>
<p><strong>Scale:</strong> 0.0 (caught nothing) → 1.0 (caught every single attack)</p>
<div class="commentary">
  <strong>Our result: {metrics['recall']:.4f} ({_metric_grade(metrics['recall'])})</strong><br>
  {commentary['recall']}
</div>

<div class="callout callout-blue">
  <strong>Why Recall matters more than Precision in cybersecurity:</strong>
  A false alarm (FP) is annoying — a security analyst spends 5 minutes checking and
  says "never mind". A missed attack (FN) can mean ransomware encrypting your servers,
  customer data stolen, or attackers moving deeper into the network. In security,
  <strong>we'd rather have 1,000 false alarms than miss 1 real attack</strong>.
</div>

<h3>3. F1 Score — "Overall report card"</h3>
<p><strong>Formula:</strong> 2 &times; (Precision &times; Recall) / (Precision + Recall)</p>
<p>If a student gets 100% in math but 0% in English, their average (50%) doesn't
   tell the full story. F1 is similar — it's a special average that stays low unless
   <em>both</em> precision and recall are good. It punishes imbalance.</p>
<p><strong>Scale:</strong> 0.0 (useless) → 1.0 (perfect at both catching attacks AND avoiding false alarms)</p>
<div class="commentary">
  <strong>Our result: {metrics['f1_score']:.4f} ({_metric_grade(metrics['f1_score'])})</strong><br>
  {commentary['f1']}
</div>

<h3>4. ROC-AUC — "Does the model understand the difference at all?"</h3>
<p><strong>Full name:</strong> Area Under the Receiver Operating Characteristic Curve</p>
<p>All the metrics above depend on our chosen threshold (the cutoff line that decides
   "normal vs attack"). ROC-AUC ignores the threshold entirely and asks a deeper
   question: <em>if I pick one random attack and one random normal event, how often
   does the model give the attack a higher suspicion score?</em></p>
<p><strong>Scale:</strong> 0.5 (coin flip — model learned nothing) → 1.0 (perfect — attack scores are always higher than normal scores)</p>
<div class="commentary">
  <strong>Our result: {metrics['roc_auc']:.4f} ({_metric_grade(metrics['roc_auc'])})</strong><br>
  {commentary['auc']}
</div>

<h2>Confusion Matrix</h2>
<table>
  <tr><th></th><th>Model said "Normal"</th><th>Model said "Attack!"</th></tr>
  <tr><th>Actually Normal</th>
      <td style="background:#e8f5e9">{cm['true_negatives']:,} &#10004; correct</td>
      <td style="background:#fde8e8">{cm['false_positives']:,} &#9888; false alarms</td></tr>
  <tr><th>Actually an Attack</th>
      <td style="background:#fde8e8">{cm['false_negatives']:,} &#9888; missed!</td>
      <td style="background:#e8f5e9">{cm['true_positives']:,} &#10004; caught!</td></tr>
</table>
<p>Total: {metrics['test_set']['total_rows']:,} rows
   ({metrics['test_set']['actual_attacks']:,} real attacks,
    {metrics['test_set']['actual_normal']:,} normal events)</p>

<h2>Threshold Calibration</h2>
<p>The threshold is the cutoff line: any event with a reconstruction error above this
   number is flagged as an attack. We set it at the {thresh['percentile']}th percentile
   of validation errors — meaning only 5% of normal traffic would trigger a false alarm.</p>
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
    commentary = _build_dynamic_commentary(metrics, cm)

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

`{input_dim}` -> `{hidden}` -> `{bottleneck}` -> `{reversed_hidden}` -> `{input_dim}`

---

## Key Metrics at a Glance

| Metric | Value | Grade |
|--------|-------|-------|
| **Precision** | {metrics['precision']:.4f} | {_metric_grade(metrics['precision'])} |
| **Recall** | {metrics['recall']:.4f} | {_metric_grade(metrics['recall'])} |
| **F1 Score** | {metrics['f1_score']:.4f} | {_metric_grade(metrics['f1_score'])} |
| **ROC-AUC** | {metrics['roc_auc']:.4f} | {_metric_grade(metrics['roc_auc'])} |

---

## Before We Dive In — Key Terms (No Jargon)

Our model looked at {metrics['test_set']['total_rows']:,} network events and for each one said either **"this looks normal"** or **"this looks like an attack!"**. After it was done, we compared its guesses to the real answers and counted four things:

| Term | What It Means | Our Count |
|------|--------------|-----------|
| **True Positive (TP)** — Correctly caught | The model said *"Attack!"* and it really WAS an attack. **This is what we want most.** | **{cm['true_positives']:,}** |
| **False Positive (FP)** — False alarm | The model said *"Attack!"* but it was actually normal traffic. Like a fire alarm going off when someone burns toast — annoying but not dangerous. | **{cm['false_positives']:,}** |
| **False Negative (FN)** — Missed attack | The model said *"Looks normal"* but it was actually an attack. **This is the most dangerous mistake** — like a security guard letting a burglar walk right past. | **{cm['false_negatives']:,}** |
| **True Negative (TN)** — Correctly ignored | The model said *"Looks normal"* and it really WAS normal. Good — no unnecessary alarm. | **{cm['true_negatives']:,}** |

---

## What Each Metric Means — and How Our Model Did

### 1. Precision — "When the alarm goes off, is it real?"

**Formula:** Correctly caught attacks / All alarms raised = TP / (TP + FP)

Think of it like this: if a fire alarm rings 100 times, how many times was there actually a fire? If 90 out of 100 were real fires, precision = 0.90 (great!). If only 2 out of 100 were real, precision = 0.02 (terrible — 98% false alarms).

**Scale:** 0.0 (every alarm is fake) -> 1.0 (every alarm is a real attack)

> **Our result: {metrics['precision']:.4f} ({_metric_grade(metrics['precision'])})** — {commentary['precision']}

### 2. Recall — "Of all real attacks, how many did we catch?"

**Formula:** Correctly caught attacks / All real attacks = TP / (TP + FN)

This is like asking: if 100 burglars tried to enter a building, how many did security actually stop? If security caught 95 out of 100, recall = 0.95. If they only caught 1, recall = 0.01 — 99 burglars got through.

**Scale:** 0.0 (caught nothing) -> 1.0 (caught every single attack)

> **Our result: {metrics['recall']:.4f} ({_metric_grade(metrics['recall'])})** — {commentary['recall']}

> **Why Recall matters more than Precision in cybersecurity:** A false alarm (FP) is annoying — a security analyst spends 5 minutes checking and says "never mind". A missed attack (FN) can mean ransomware encrypting your servers, customer data stolen, or attackers moving deeper into the network. In security, **we'd rather have 1,000 false alarms than miss 1 real attack**.

### 3. F1 Score — "Overall report card"

**Formula:** 2 x (Precision x Recall) / (Precision + Recall)

If a student gets 100% in math but 0% in English, their average (50%) doesn't tell the full story. F1 is similar — it's a special average that stays low unless *both* precision and recall are good. It punishes imbalance.

**Scale:** 0.0 (useless) -> 1.0 (perfect at both catching attacks AND avoiding false alarms)

> **Our result: {metrics['f1_score']:.4f} ({_metric_grade(metrics['f1_score'])})** — {commentary['f1']}

### 4. ROC-AUC — "Does the model understand the difference at all?"

**Full name:** Area Under the Receiver Operating Characteristic Curve

All the metrics above depend on our chosen threshold (the cutoff line that decides "normal vs attack"). ROC-AUC ignores the threshold entirely and asks a deeper question: *if I pick one random attack and one random normal event, how often does the model give the attack a higher suspicion score?*

**Scale:** 0.5 (coin flip — model learned nothing) -> 1.0 (perfect — attack scores are always higher than normal scores)

> **Our result: {metrics['roc_auc']:.4f} ({_metric_grade(metrics['roc_auc'])})** — {commentary['auc']}

---

## Confusion Matrix

| | Model said "Normal" | Model said "Attack!" |
|---|---|---|
| **Actually Normal** | {cm['true_negatives']:,} (correct) | {cm['false_positives']:,} (false alarms) |
| **Actually an Attack** | {cm['false_negatives']:,} (missed!) | {cm['true_positives']:,} (caught!) |

Total: {metrics['test_set']['total_rows']:,} rows ({metrics['test_set']['actual_attacks']:,} real attacks, {metrics['test_set']['actual_normal']:,} normal events)

---

## Threshold Calibration

The threshold is the cutoff line: any event with a reconstruction error above this number is flagged as an attack. We set it at the {thresh['percentile']}th percentile of validation errors — meaning only 5% of normal traffic would trigger a false alarm.

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
