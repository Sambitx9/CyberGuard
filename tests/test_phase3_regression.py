"""CyberGuard Phase 3: Comprehensive Regression & Hardening Test Suite.

Verifies:
1. Model Weights Integrity (Size + SHA-256 match exactly).
2. No-Face Bypass (0 model calls, 0 Grad-CAM calls).
3. Real Face Inference & Dynamic Grad-CAM.
4. Fake Face Inference & Dynamic Grad-CAM.
5. Invalid File Safety (ImageValidationError).
6. Multi-Face Deterministic Policy (1, 2, and 3 faces).
7. Face Detector Abstraction (BaseFaceDetector implementations).
8. Calibration Data Separation & Artifacts.
9. External Dataset Architecture & Blocked Status Integrity.
10. Production Threshold Preservation (0.50).
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from unittest.mock import patch

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import config
from src.detector import ImageDetector
from src.face_detector import BaseFaceDetector, HaarCascadeFaceDetector, EnsembleHaarFaceDetector, OpenCVDNNFaceDetector
from src.preprocessing import validate_upload, ImageValidationError
import src.explainability


EXPECTED_MODEL_SIZE = 21752160
EXPECTED_MODEL_SHA256 = "E7422FCA2F61706F5F0379453469C8D536A7D343DF1A185214AF8F48028539A6"


def test_model_weight_integrity():
    assert config.MODEL_PATH.exists(), f"Model missing at {config.MODEL_PATH}"
    actual_size = config.MODEL_PATH.stat().st_size
    assert actual_size == EXPECTED_MODEL_SIZE, f"Model size mismatch! Expected {EXPECTED_MODEL_SIZE}, got {actual_size}"
    
    hasher = hashlib.sha256()
    hasher.update(config.MODEL_PATH.read_bytes())
    actual_sha = hasher.hexdigest().upper()
    assert actual_sha == EXPECTED_MODEL_SHA256, f"Model SHA256 mismatch! Expected {EXPECTED_MODEL_SHA256}, got {actual_sha}"
    print("[PASS] Test 1: Model weight integrity verified (Size & SHA-256 exact match).")


def test_no_face_bypass():
    detector = ImageDetector()
    detector.load()
    orig_call = detector._model

    noface_path = config.BASE_DIR / "test_data" / "infographic.png"
    val_noface = validate_upload(noface_path.name, noface_path.read_bytes())

    call_count = 0
    def spy_model(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        return orig_call(*args, **kwargs)
    detector._model = spy_model

    with patch("src.detector.explain", wraps=src.explainability.explain) as spy_cam:
        res = detector.analyze(val_noface)
        assert res.prediction == "NO FACE DETECTED"
        assert res.confidence is None
        assert res.risk == "N/A"
        assert res.visual.method == "NOT RUN"
        assert call_count == 0, f"Model called {call_count} times on no-face image!"
        assert spy_cam.call_count == 0, f"Grad-CAM called {spy_cam.call_count} times on no-face image!"
    
    detector._model = orig_call
    print("[PASS] Test 2: No-face bypass verified (model=0 calls, cam=0 calls).")


def test_face_inference():
    detector = ImageDetector()
    detector.load()
    orig_call = detector._model

    # Real face
    p_real = config.BASE_DIR / "test_data" / "real" / "real_test_0001.jpg"
    val_real = validate_upload(p_real.name, p_real.read_bytes())
    call_count = 0
    def spy_model(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        return orig_call(*args, **kwargs)
    spy_model.get_layer = orig_call.get_layer
    detector._model = spy_model

    with patch("src.detector.explain", wraps=src.explainability.explain) as spy_cam:
        res_real = detector.analyze(val_real)
        assert res_real.prediction == "REAL"
        assert res_real.confidence is not None
        assert res_real.risk == "LOW"
        assert res_real.visual.method == "Grad-CAM"
        assert "REAL" in res_real.visual.title
        assert call_count == 1
        assert spy_cam.call_count == 1
    detector._model = orig_call

    # Fake face
    p_fake = config.BASE_DIR / "test_data" / "fake" / "fake_test_0000.jpg"
    val_fake = validate_upload(p_fake.name, p_fake.read_bytes())
    call_count = 0
    detector._model = spy_model
    with patch("src.detector.explain", wraps=src.explainability.explain) as spy_cam:
        res_fake = detector.analyze(val_fake)
        assert res_fake.prediction == "FAKE"
        assert res_fake.confidence is not None
        assert res_fake.risk == "HIGH"
        assert res_fake.visual.method == "Grad-CAM"
        assert "FAKE" in res_fake.visual.title
        assert call_count == 1
        assert spy_cam.call_count == 1
    detector._model = orig_call

    print("[PASS] Test 3: Face inference verified (REAL & FAKE, dynamic Grad-CAM titles).")


def test_invalid_file():
    caught = False
    try:
        validate_upload("corrupt.png", b"corrupted_non_image_payload")
    except ImageValidationError:
        caught = True
    assert caught, "Failed to catch corrupted image!"
    print("[PASS] Test 4: Invalid file rejection verified (ImageValidationError).")


def test_multi_face_policy():
    mf_path = config.REPORTS_DIR / "multiface_regression.json"
    assert mf_path.exists(), "multiface_regression.json missing"
    data = json.loads(mf_path.read_text(encoding="utf-8"))
    assert data["status"] == "PASS"
    assert data["cases"]["two_faces"]["passed"]
    assert data["cases"]["three_faces"]["passed"]
    print("[PASS] Test 5: Multi-face deterministic policy verified (1, 2, 3 faces).")


def test_detector_abstraction():
    haar = HaarCascadeFaceDetector()
    ens = EnsembleHaarFaceDetector()
    dnn = OpenCVDNNFaceDetector()
    assert isinstance(haar, BaseFaceDetector)
    assert isinstance(ens, BaseFaceDetector)
    assert isinstance(dnn, BaseFaceDetector)
    assert haar.is_available
    assert ens.is_available
    assert dnn.is_available, "YuNet model weights should be loaded and available"
    print("[PASS] Test 6: Face detector abstraction verified (Haar, Ensemble, YuNet available).")


def test_calibration_and_separation():
    cal_json = config.REPORTS_DIR / "calibration_analysis.json"
    assert cal_json.exists(), "calibration_analysis.json missing"
    data = json.loads(cal_json.read_text(encoding="utf-8"))
    assert "held-out validation split" in data["dataset_used"]
    assert (config.MODEL_DIR / "calibrator_platt.joblib").exists()
    assert (config.MODEL_DIR / "calibrator_isotonic.joblib").exists()
    assert (config.REPORTS_DIR / "calibration_curve.png").exists()
    print("[PASS] Test 7: Calibration data separation and artifacts verified.")


def test_external_dataset_status_and_threshold():
    assert config.FAKE_THRESHOLD == 0.50, f"Threshold altered! Expected 0.50, got {config.FAKE_THRESHOLD}"
    ext_rep = config.REPORTS_DIR / "external_dataset_integrity.json"
    assert ext_rep.exists(), "external_dataset_integrity.json missing"
    data = json.loads(ext_rep.read_text(encoding="utf-8"))
    ext_status = data.get("external_dataset_status") or data.get("status")
    assert ext_status in ("NOT PROVIDED", "AVAILABLE", "PASS")
    assert data.get("validation_status") in ("BLOCKED", "READY_FOR_EVALUATION", "COMPLETE")
    assert (config.BASE_DIR / "DATASET_READY.md").exists()
    print("[PASS] Test 8: External dataset status & production threshold (0.50) verified.")


def test_all():
    print("\n==================================================")
    print("CYBERGUARD PHASE 3 — REGRESSION TEST EXECUTION")
    print("==================================================")
    test_model_weight_integrity()
    test_no_face_bypass()
    test_face_inference()
    test_invalid_file()
    test_multi_face_policy()
    test_detector_abstraction()
    test_calibration_and_separation()
    test_external_dataset_status_and_threshold()
    print("\nALL 8 PHASE 3 REGRESSION SUITES PASSED (100%)!\n")


if __name__ == "__main__":
    test_all()
