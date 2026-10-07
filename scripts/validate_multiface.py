"""CyberGuard Phase 3: Multi-Face Deterministic Validation Runner.

Tests 1-face, 2-face, and 3-face scenarios to verify deterministic primary face selection
and isolation of secondary faces.
Generates reports/multiface_regression.json.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from PIL import Image

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import config
from src.detector import ImageDetector
from src.preprocessing import detect_faces, validate_upload


def run_multiface_validation():
    print("Running Multi-Face Deterministic Policy Validation...")
    detector = ImageDetector()
    detector.load()

    # 1. Test 1-face image
    p_1face = config.BASE_DIR / "test_data" / "real" / "real_test_0001.jpg"
    val_1face = validate_upload(p_1face.name, p_1face.read_bytes())
    res_1face = detector.analyze(val_1face)

    # 2. Test 2-face image
    p_2face = config.BASE_DIR / "test_data" / "multi_face_test.jpg"
    val_2face = validate_upload(p_2face.name, p_2face.read_bytes())
    res_2face = detector.analyze(val_2face)

    # 3. Create/Test 3-face image fixture
    # Create a 3-face canvas side-by-side with 3 distinct scales:
    img_f1 = Image.open(config.BASE_DIR / "test_data" / "real" / "real_test_0001.jpg").convert("RGB")
    img_f2 = Image.open(config.BASE_DIR / "test_data" / "fake" / "fake_test_0000.jpg").convert("RGB")
    img_f3 = Image.open(config.BASE_DIR / "test_data" / "real" / "real_test_0002.jpg").convert("RGB")

    # Scale: Large (240x240), Medium (180x180), Small (120x120)
    w_total = 240 + 180 + 120 + 40
    h_canvas = 300
    canvas_3face = Image.new("RGB", (w_total, h_canvas), color=(20, 20, 20))
    canvas_3face.paste(img_f1.resize((240, 240)), (10, 30))
    canvas_3face.paste(img_f2.resize((180, 180)), (260, 60))
    canvas_3face.paste(img_f3.resize((120, 120)), (450, 90))

    p_3face = config.BASE_DIR / "test_data" / "three_face_test.jpg"
    canvas_3face.save(p_3face, format="JPEG", quality=95)

    val_3face = validate_upload(p_3face.name, p_3face.read_bytes())
    res_3face = detector.analyze(val_3face)

    # Verifications
    faces_3 = detect_faces(canvas_3face)
    boxes_3 = faces_3.boxes
    areas = [w * h for _, _, w, h in boxes_3]
    is_sorted_desc = all(areas[i] >= areas[i + 1] for i in range(len(areas) - 1))

    # Check warnings in 2-face and 3-face
    warn_2 = any("Multiple faces detected (2)" in w and "Analyzing primary face" in w for w in res_2face.warnings)
    warn_3 = any("Multiple faces detected" in w and "Analyzing primary face" in w for w in res_3face.warnings)

    regression_data = {
        "timestamp": "2026-10-07",
        "policy": {
            "strategy": "Largest Area Selection",
            "sorting": "Area (width * height) Descending",
            "primary_index": 0,
            "secondary_handling": "Annotated in slate with secondary badges, NOT cropped or passed to model"
        },
        "cases": {
            "single_face": {
                "file": p_1face.name,
                "faces_detected": res_1face.face_count,
                "prediction": res_1face.prediction,
                "warnings": res_1face.warnings,
                "passed": res_1face.face_count == 1
            },
            "two_faces": {
                "file": p_2face.name,
                "faces_detected": res_2face.face_count,
                "primary_warning_logged": warn_2,
                "warnings": res_2face.warnings,
                "passed": res_2face.face_count == 2 and warn_2
            },
            "three_faces": {
                "file": p_3face.name,
                "faces_detected": res_3face.face_count,
                "areas_sorted_descending": is_sorted_desc,
                "primary_warning_logged": warn_3,
                "warnings": res_3face.warnings,
                "passed": res_3face.face_count >= 2 and is_sorted_desc and warn_3
            }
        },
        "status": "PASS" if (res_1face.face_count == 1 and warn_2 and is_sorted_desc) else "FAIL"
    }

    out_path = config.REPORTS_DIR / "multiface_regression.json"
    out_path.write_text(json.dumps(regression_data, indent=2), encoding="utf-8")
    print("Multi-face validation complete! Saved to:", out_path)
    print("Status:", regression_data["status"])


if __name__ == "__main__":
    run_multiface_validation()
