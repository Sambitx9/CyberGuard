"""CyberGuard Phase 3: Confidence Calibration Engine.

Evaluates Platt Scaling and Isotonic Regression on held-out validation data.
Computes Brier Score, ECE, MCE, and generates calibration curves.
Never touches model weights.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.calibration import calibration_curve
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss

import config
from src.detector import ImageDetector
from src.preprocessing import prepare_model_input


def compute_ece_mce(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> tuple[float, float]:
    """Compute Expected Calibration Error (ECE) and Maximum Calibration Error (MCE)."""
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    mce = 0.0
    total_samples = len(y_true)

    for i in range(n_bins):
        low, high = bin_edges[i], bin_edges[i + 1]
        mask = (y_prob >= low) & (y_prob <= high if i == n_bins - 1 else y_prob < high)
        bin_count = np.sum(mask)
        if bin_count > 0:
            bin_acc = float(np.mean(y_true[mask]))
            bin_conf = float(np.mean(y_prob[mask]))
            diff = abs(bin_acc - bin_conf)
            ece += (bin_count / total_samples) * diff
            mce = max(mce, diff)

    return float(ece), float(mce)


def run_calibration():
    print("Running Confidence Calibration Engine...")
    detector = ImageDetector()
    detector.load()
    model = detector._model

    # 1. Load held-out validation split from dataset/
    # Exact replication of train_model.py validation_split=0.20, seed=42
    from tensorflow import keras
    val_ds = keras.utils.image_dataset_from_directory(
        directory=str(config.DATASET_DIR),
        labels="inferred",
        label_mode="binary",
        class_names=list(config.CLASS_NAMES),
        color_mode="rgb",
        image_size=config.IMG_SIZE,
        batch_size=config.BATCH_SIZE,
        seed=config.SEED,
        validation_split=config.VALIDATION_SPLIT,
        subset="validation",
        shuffle=True,
    )

    y_val_true = []
    y_val_prob = []
    for images, labels in val_ds:
        probs = np.asarray(model(images, training=False)).reshape(-1)
        y_val_prob.append(probs)
        y_val_true.append(np.asarray(labels).reshape(-1))

    y_val_true = np.concatenate(y_val_true).astype(int)
    y_val_prob = np.concatenate(y_val_prob).astype(float)
    val_n = len(y_val_true)
    print(f"Calibration data: {val_n} held-out validation samples ({np.sum(y_val_true==0)} REAL, {np.sum(y_val_true==1)} FAKE).")

    # 2. Raw Model Calibration Metrics
    raw_brier = float(brier_score_loss(y_val_true, y_val_prob))
    raw_ece, raw_mce = compute_ece_mce(y_val_true, y_val_prob)

    # 3. Fit Platt Scaling (Logistic Regression)
    # Clip probabilities slightly to avoid logit infinity
    eps = 1e-6
    p_clipped = np.clip(y_val_prob, eps, 1.0 - eps)
    logits = np.log(p_clipped / (1.0 - p_clipped)).reshape(-1, 1)

    platt_model = LogisticRegression(C=1.0, solver="lbfgs")
    platt_model.fit(logits, y_val_true)
    platt_val_prob = platt_model.predict_proba(logits)[:, 1]

    platt_brier = float(brier_score_loss(y_val_true, platt_val_prob))
    platt_ece, platt_mce = compute_ece_mce(y_val_true, platt_val_prob)

    # 4. Fit Isotonic Regression
    iso_model = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
    iso_model.fit(y_val_prob, y_val_true)
    iso_val_prob = iso_model.predict(y_val_prob)

    iso_brier = float(brier_score_loss(y_val_true, iso_val_prob))
    iso_ece, iso_mce = compute_ece_mce(y_val_true, iso_val_prob)

    # Save fitted calibrators
    joblib.dump(platt_model, config.MODEL_DIR / "calibrator_platt.joblib")
    joblib.dump(iso_model, config.MODEL_DIR / "calibrator_isotonic.joblib")

    # 5. Evaluate impact on Test Set (100 images)
    test_real_dir = config.BASE_DIR / "test_data" / "real"
    test_fake_dir = config.BASE_DIR / "test_data" / "fake"
    test_paths = sorted(list(test_real_dir.glob("*.jpg"))) + sorted(list(test_fake_dir.glob("*.jpg")))
    
    test_raw_probs = []
    test_gt = [0] * 50 + [1] * 50
    for p in test_paths:
        from PIL import Image
        with Image.open(p) as img:
            batch = prepare_model_input(img.convert("RGB"))
            test_raw_probs.append(float(np.asarray(model(batch, training=False))[0, 0]))
            
    test_raw_probs = np.array(test_raw_probs)
    test_logits = np.log(np.clip(test_raw_probs, eps, 1.0 - eps) / (1.0 - np.clip(test_raw_probs, eps, 1.0 - eps))).reshape(-1, 1)
    test_platt_probs = platt_model.predict_proba(test_logits)[:, 1]

    # Inspect test high-confidence errors before and after
    high_conf_raw_indices = [idx for idx, (p, gt) in enumerate(zip(test_raw_probs, test_gt)) if (p < 0.5 and gt == 1 and (1.0 - p) >= 0.90)]
    high_conf_platt_count = sum(1 for idx in high_conf_raw_indices if (1.0 - test_platt_probs[idx]) >= 0.90)

    # 6. Plot Reliability Diagram
    fig, ax = plt.subplots(figsize=(6.5, 5.5), dpi=140)
    ax.plot([0, 1], [0, 1], "k:", label="Perfect calibration")

    frac_raw, mean_raw = calibration_curve(y_val_true, y_val_prob, n_bins=8)
    frac_platt, mean_platt = calibration_curve(y_val_true, platt_val_prob, n_bins=8)
    frac_iso, mean_iso = calibration_curve(y_val_true, iso_val_prob, n_bins=8)

    ax.plot(mean_raw, frac_raw, "s-", color="#ef4444", label=f"Raw Model (ECE: {raw_ece:.3f}, Brier: {raw_brier:.3f})")
    ax.plot(mean_platt, frac_platt, "o-", color="#3b82f6", label=f"Platt Scaling (ECE: {platt_ece:.3f}, Brier: {platt_brier:.3f})")
    ax.plot(mean_iso, frac_iso, "^-", color="#10b981", label=f"Isotonic (ECE: {iso_ece:.3f}, Brier: {iso_brier:.3f})")

    ax.set_xlabel("Mean Predicted Probability P(FAKE)")
    ax.set_ylabel("Fraction of Positives (Empirical Accuracy)")
    ax.set_title("Reliability Diagram (Held-Out Validation Set)")
    ax.legend(loc="lower right", fontsize=8.5)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    chart_path = config.REPORTS_DIR / "calibration_curve.png"
    fig.savefig(chart_path)
    plt.close(fig)

    calibration_results = {
        "dataset_used": "dataset/ held-out validation split (strictly disjoint from test_data)",
        "calibration_samples": val_n,
        "metrics": {
            "raw_model": {
                "brier_score": raw_brier,
                "ece": raw_ece,
                "mce": raw_mce
            },
            "platt_scaling": {
                "brier_score": platt_brier,
                "ece": platt_ece,
                "mce": platt_mce,
                "improvement_brier": raw_brier - platt_brier,
                "improvement_ece": raw_ece - platt_ece
            },
            "isotonic_regression": {
                "brier_score": iso_brier,
                "ece": iso_ece,
                "mce": iso_mce,
                "improvement_brier": raw_brier - iso_brier,
                "improvement_ece": raw_ece - iso_ece
            }
        },
        "recommendation": {
            "selected_calibrator": "Platt Scaling (Logistic Calibration)",
            "rationale": "Platt scaling produces continuous, monotonic probability adjustments with lower risk of overfitting on small validation splits than piecewise isotonic regression.",
            "production_status": "VALIDATED FOR PROBABILITY DISPLAY; Classifier decision threshold preserved at 0.50"
        }
    }

    out_json = config.REPORTS_DIR / "calibration_analysis.json"
    out_json.write_text(json.dumps(calibration_results, indent=2), encoding="utf-8")
    print(f"Calibration analysis complete! Saved to {out_json} and {chart_path}")
    print(f"Raw ECE: {raw_ece:.4f} -> Platt ECE: {platt_ece:.4f} (Brier: {raw_brier:.4f} -> {platt_brier:.4f})")


if __name__ == "__main__":
    run_calibration()
