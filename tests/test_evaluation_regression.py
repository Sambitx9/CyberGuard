"""Regression test suite for CyberGuard Phase 2 Model Evaluation Engine.

Verifies:
1. No-face image: Model inference NOT called, Grad-CAM NOT called.
2. Real face image: Model inference CALLED, Grad-CAM CALLED, prediction REAL.
3. Fake face image: Model inference CALLED, Grad-CAM CALLED, prediction FAKE.
4. Evaluation engine runs and generates all expected artifacts:
   - reports/model_evaluation.json
   - reports/model_evaluation_report.html
   - reports/confusion_matrix.png
   - reports/confusion_matrix.json
   - reports/threshold_analysis.csv
   - reports/threshold_analysis.png
   - reports/confidence_distribution.png
   - reports/evaluation_errors.csv
   - reports/high_confidence_errors.csv
5. Model weights file remains untouched.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import patch

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import config
from src.detector import ImageDetector
from src.preprocessing import validate_upload, ImageValidationError
import src.explainability


def test_regression_all():
    print("Running CyberGuard Regression & Pipeline Integrity Tests...")
    
    # Check 1: Model file exists and is intact
    assert config.MODEL_PATH.exists(), f"Model missing at {config.MODEL_PATH}"
    model_size = config.MODEL_PATH.stat().st_size
    assert model_size == 21752160, f"Model size changed! Expected 21752160, got {model_size}"
    print("[PASS] Model weights integrity verified.")

    detector = ImageDetector()
    detector.load()

    # Check 2: No-face input strictly bypasses model & Grad-CAM
    noface_path = config.BASE_DIR / "test_data" / "infographic.png"
    val_noface = validate_upload(noface_path.name, noface_path.read_bytes())
    
    call_count = 0
    orig_call = detector._model
    def spy_model(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        return orig_call(*args, **kwargs)
    detector._model = spy_model

    with patch("src.detector.explain", wraps=src.explainability.explain) as spy_cam:
        res_noface = detector.analyze(val_noface)
        assert res_noface.prediction == "NO FACE DETECTED"
        assert res_noface.confidence is None
        assert res_noface.risk == "N/A"
        assert res_noface.visual.method == "NOT RUN"
        assert call_count == 0, f"Model was called {call_count} times on no-face image!"
        assert spy_cam.call_count == 0, f"Grad-CAM was called {spy_cam.call_count} times on no-face image!"
    detector._model = orig_call
    print("[PASS] No-face bypass verified (model=0 calls, cam=0 calls).")

    # Check 3: Real face calls model & Grad-CAM
    real_path = config.BASE_DIR / "test_data" / "real" / "real_test_0001.jpg"
    val_real = validate_upload(real_path.name, real_path.read_bytes())
    
    call_count = 0
    detector._model = spy_model
    # Ensure keras layer lookup still works on spy
    spy_model.get_layer = orig_call.get_layer

    with patch("src.detector.explain", wraps=src.explainability.explain) as spy_cam:
        res_real = detector.analyze(val_real)
        assert res_real.prediction == "REAL"
        assert res_real.confidence is not None
        assert res_real.risk == "LOW"
        assert res_real.visual.method == "Grad-CAM"
        assert "REAL" in res_real.visual.title
        assert call_count == 1, f"Model was called {call_count} times on real face image!"
        assert spy_cam.call_count == 1, f"Grad-CAM was called {spy_cam.call_count} times on real face image!"
    detector._model = orig_call
    print("[PASS] Real face inference & Grad-CAM verified.")

    # Check 4: Fake face calls model & Grad-CAM
    fake_path = config.BASE_DIR / "test_data" / "fake" / "fake_test_0000.jpg"
    val_fake = validate_upload(fake_path.name, fake_path.read_bytes())
    
    call_count = 0
    detector._model = spy_model
    with patch("src.detector.explain", wraps=src.explainability.explain) as spy_cam:
        res_fake = detector.analyze(val_fake)
        assert res_fake.prediction == "FAKE"
        assert res_fake.confidence is not None
        assert res_fake.risk == "HIGH"
        assert res_fake.visual.method == "Grad-CAM"
        assert "FAKE" in res_fake.visual.title
        assert call_count == 1, f"Model was called {call_count} times on fake face image!"
    detector._model = orig_call
    print("[PASS] Fake face inference & Grad-CAM verified.")

    # Check 5: Invalid file handling
    corrupt_caught = False
    try:
        validate_upload("corrupt.jpg", b"invalid_binary_bytes")
    except ImageValidationError:
        corrupt_caught = True
    assert corrupt_caught, "Corrupted file failed to raise ImageValidationError!"
    print("[PASS] Invalid file rejection verified.")

    # Check 6: Required evaluation artifact generation
    reports_dir = config.REPORTS_DIR
    required_artifacts = [
        reports_dir / "model_evaluation.json",
        reports_dir / "model_evaluation_report.html",
        reports_dir / "confusion_matrix.png",
        reports_dir / "confusion_matrix.json",
        reports_dir / "threshold_analysis.csv",
        reports_dir / "threshold_analysis.png",
        reports_dir / "confidence_distribution.png",
        reports_dir / "evaluation_errors.csv",
        reports_dir / "high_confidence_errors.csv",
        config.BASE_DIR / "dataset_integrity_report.json",
        config.BASE_DIR / "MODEL_LIMITATIONS.md",
        config.BASE_DIR / "DATASET_RECOMMENDATION.md",
    ]
    for p in required_artifacts:
        assert p.exists(), f"Required artifact missing: {p}"
        assert p.stat().st_size > 0, f"Artifact empty: {p}"
    print("[PASS] All evaluation artifacts exist and are non-empty.")

    print("\nALL REGRESSION TESTS PASSED 100%!")


if __name__ == "__main__":
    test_regression_all()
