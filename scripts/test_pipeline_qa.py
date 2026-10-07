"""CyberGuard Phase 5: Comprehensive Pipeline QA Test Suite.

Executes all 10 mandated pipeline QA tests:
TEST 1: Valid single-face REAL image
TEST 2: Valid single-face FAKE image
TEST 3: No-face image
TEST 4: Multiple faces (2 faces & 3 faces)
TEST 5: Corrupted image
TEST 6: Unsupported file type
TEST 7: Extremely small image (<32px)
TEST 8: Very large image (>20MB)
TEST 9: CLI prediction (predict.py)
TEST 10: HTML forensic report validation
"""
from __future__ import annotations

import io
import json
import os
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import numpy as np
from PIL import Image

import config
from src import utils
from src.detector import ImageDetector
from src.preprocessing import ImageValidationError, validate_upload


def run_pipeline_qa():
    print("=" * 60)
    print("CYBERGUARD PHASE 5 — PIPELINE QA AUDIT")
    print("=" * 60)
    results = {}

    detector = ImageDetector()
    detector.load()

    # ----------------------------------------------------
    # TEST 1: Valid single-face REAL image
    # ----------------------------------------------------
    print("\n--- TEST 1: Valid single-face REAL image ---")
    real_path = config.BASE_DIR / "test_data" / "real" / "real_test_0001.jpg"
    val_real = validate_upload(real_path.name, real_path.read_bytes())
    res_real = detector.analyze(val_real)

    t1_pass = (
        res_real.face_count >= 1
        and res_real.prediction == "REAL"
        and res_real.confidence is not None
        and res_real.confidence > 50.0
        and res_real.risk == "LOW"
        and "Grad-CAM" in res_real.visual.method
        and "REAL" in res_real.visual.title
    )
    print(f"Face count: {res_real.face_count}, Prediction: {res_real.prediction}, "
          f"Conf: {res_real.confidence:.1f}%, Risk: {res_real.risk}, Title: '{res_real.visual.title}'")
    print(f"TEST 1 Result: {'PASS' if t1_pass else 'FAIL'}")
    results["test_1_real"] = "PASS" if t1_pass else "FAIL"

    # ----------------------------------------------------
    # TEST 2: Valid single-face FAKE image
    # ----------------------------------------------------
    print("\n--- TEST 2: Valid single-face FAKE image ---")
    fake_path = config.BASE_DIR / "test_data" / "fake" / "fake_test_0000.jpg"
    val_fake = validate_upload(fake_path.name, fake_path.read_bytes())
    res_fake = detector.analyze(val_fake)

    t2_pass = (
        res_fake.face_count >= 1
        and res_fake.prediction == "FAKE"
        and res_fake.confidence is not None
        and res_fake.confidence > 50.0
        and res_fake.risk in ("MEDIUM", "HIGH")
        and "Grad-CAM" in res_fake.visual.method
        and "FAKE" in res_fake.visual.title
    )
    print(f"Face count: {res_fake.face_count}, Prediction: {res_fake.prediction}, "
          f"Conf: {res_fake.confidence:.1f}%, Risk: {res_fake.risk}, Title: '{res_fake.visual.title}'")
    print(f"TEST 2 Result: {'PASS' if t2_pass else 'FAIL'}")
    results["test_2_fake"] = "PASS" if t2_pass else "FAIL"

    # ----------------------------------------------------
    # TEST 3: No-face image
    # ----------------------------------------------------
    print("\n--- TEST 3: No-face image ---")
    noface_path = config.BASE_DIR / "test_data" / "infographic.png"
    val_noface = validate_upload(noface_path.name, noface_path.read_bytes())

    call_count = 0
    orig_model = detector._model
    def spy_model(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        return orig_model(*args, **kwargs)

    detector._model = spy_model
    res_noface = detector.analyze(val_noface)
    detector._model = orig_model

    t3_pass = (
        res_noface.prediction == "NO FACE DETECTED"
        and res_noface.confidence is None
        and res_noface.fake_probability is None
        and res_noface.risk == "N/A"
        and res_noface.face_count == 0
        and call_count == 0
        and res_noface.visual.is_gradcam is False
        and res_noface.visual.method == "NOT RUN"
    )
    print(f"Prediction: {res_noface.prediction}, Conf: {res_noface.confidence}, Risk: {res_noface.risk}, "
          f"Model calls: {call_count}, Visual: {res_noface.visual.method}")
    print(f"TEST 3 Result: {'PASS' if t3_pass else 'FAIL'}")
    results["test_3_noface"] = "PASS" if t3_pass else "FAIL"

    # ----------------------------------------------------
    # TEST 4: Multiple faces (2 faces & 3 faces)
    # ----------------------------------------------------
    print("\n--- TEST 4: Multiple faces (2 faces & 3 faces) ---")
    # Generate deterministic multi-face composite images using real face crops
    from src.preprocessing import detect_faces, crop_face
    real_img = Image.open(real_path).convert("RGB")
    f_detect = detect_faces(real_img)
    base_crop = crop_face(real_img, f_detect.boxes[0]) if f_detect.boxes else real_img.resize((150, 150))

    # 2 Faces composite: Face 1 (220x220), Face 2 (140x140)
    comp_2 = Image.new("RGB", (700, 400), (230, 230, 230))
    face1_2 = base_crop.resize((220, 220))
    face2_2 = base_crop.resize((140, 140))
    comp_2.paste(face1_2, (40, 90))
    comp_2.paste(face2_2, (380, 130))

    buf2 = io.BytesIO()
    comp_2.save(buf2, format="JPEG")
    val_2 = validate_upload("two_faces.jpg", buf2.getvalue())
    res_2 = detector.analyze(val_2)

    has_multi_warn_2 = any("Multiple faces detected" in w for w in res_2.warnings)
    print(f"2-Face test -> Count: {res_2.face_count}, Warning logged: {has_multi_warn_2}")

    # 3 Faces composite: Face 1 (240x240), Face 2 (160x160), Face 3 (100x100)
    comp_3 = Image.new("RGB", (900, 400), (230, 230, 230))
    face1_3 = base_crop.resize((240, 240))
    face2_3 = base_crop.resize((160, 160))
    face3_3 = base_crop.resize((100, 100))
    comp_3.paste(face1_3, (30, 80))
    comp_3.paste(face2_3, (330, 120))
    comp_3.paste(face3_3, (560, 150))

    buf3 = io.BytesIO()
    comp_3.save(buf3, format="JPEG")
    val_3 = validate_upload("three_faces.jpg", buf3.getvalue())
    res_3 = detector.analyze(val_3)

    has_multi_warn_3 = any("Multiple faces detected" in w for w in res_3.warnings)
    print(f"3-Face test -> Count: {res_3.face_count}, Warning logged: {has_multi_warn_3}")

    t4_pass = has_multi_warn_2 and has_multi_warn_3 and (res_2.prediction in ("REAL", "FAKE")) and (res_3.prediction in ("REAL", "FAKE"))
    print(f"TEST 4 Result: {'PASS' if t4_pass else 'FAIL'}")
    results["test_4_multiface"] = "PASS" if t4_pass else "FAIL"

    # ----------------------------------------------------
    # TEST 5: Corrupted image
    # ----------------------------------------------------
    print("\n--- TEST 5: Corrupted image ---")
    corrupt_data = b"\xff\xd8\xff\xe0" + b"garbagecorrupteddata" * 50
    t5_rejected = False
    try:
        validate_upload("corrupt.jpg", corrupt_data)
    except ImageValidationError as exc:
        t5_rejected = True
        print(f"Safely rejected: {exc}")
    print(f"TEST 5 Result: {'PASS' if t5_rejected else 'FAIL'}")
    results["test_5_corrupt"] = "PASS" if t5_rejected else "FAIL"

    # ----------------------------------------------------
    # TEST 6: Unsupported file type
    # ----------------------------------------------------
    print("\n--- TEST 6: Unsupported file type ---")
    t6_rejected = False
    try:
        validate_upload("payload.exe", b"MZ\x90\x00\x03\x00\x00\x00")
    except ImageValidationError as exc:
        t6_rejected = True
        print(f"Safely rejected: {exc}")
    print(f"TEST 6 Result: {'PASS' if t6_rejected else 'FAIL'}")
    results["test_6_unsupported"] = "PASS" if t6_rejected else "FAIL"

    # ----------------------------------------------------
    # TEST 7: Extremely small image (<32px)
    # ----------------------------------------------------
    print("\n--- TEST 7: Extremely small image (<32px) ---")
    small_img = Image.new("RGB", (16, 16), (255, 0, 0))
    sbuf = io.BytesIO()
    small_img.save(sbuf, format="PNG")
    t7_rejected = False
    try:
        validate_upload("tiny.png", sbuf.getvalue())
    except ImageValidationError as exc:
        t7_rejected = True
        print(f"Safely rejected: {exc}")
    print(f"TEST 7 Result: {'PASS' if t7_rejected else 'FAIL'}")
    results["test_7_small"] = "PASS" if t7_rejected else "FAIL"

    # ----------------------------------------------------
    # TEST 8: Very large image (>20MB)
    # ----------------------------------------------------
    print("\n--- TEST 8: Very large image (>20MB) ---")
    large_data = b"0" * (25 * 1024 * 1024)
    t8_rejected = False
    try:
        validate_upload("huge.jpg", large_data)
    except ImageValidationError as exc:
        t8_rejected = True
        print(f"Safely rejected: {exc}")
    print(f"TEST 8 Result: {'PASS' if t8_rejected else 'FAIL'}")
    results["test_8_large"] = "PASS" if t8_rejected else "FAIL"

    # ----------------------------------------------------
    # TEST 9: CLI prediction
    # ----------------------------------------------------
    print("\n--- TEST 9: CLI prediction ---")
    cli_real_report = config.REPORTS_DIR / "cli_test_real.html"
    cli_noface_report = config.REPORTS_DIR / "cli_test_noface.html"

    p_real = subprocess.run(
        [sys.executable, str(config.BASE_DIR / "predict.py"), str(real_path), "--report", str(cli_real_report)],
        capture_output=True, text=True,
    )
    p_noface = subprocess.run(
        [sys.executable, str(config.BASE_DIR / "predict.py"), str(noface_path), "--report", str(cli_noface_report)],
        capture_output=True, text=True,
    )

    t9_pass = (
        p_real.returncode == 0
        and "CYBERGUARD ANALYSIS" in p_real.stdout
        and "REAL" in p_real.stdout
        and p_noface.returncode == 0
        and "UNSUPPORTED — NO FACE DETECTED" in p_noface.stdout
        and cli_real_report.exists()
        and cli_noface_report.exists()
    )
    print(f"CLI Real Exit code: {p_real.returncode}, CLI NoFace Exit code: {p_noface.returncode}")
    print(f"TEST 9 Result: {'PASS' if t9_pass else 'FAIL'}")
    results["test_9_cli"] = "PASS" if t9_pass else "FAIL"

    # ----------------------------------------------------
    # TEST 10: HTML forensic report
    # ----------------------------------------------------
    print("\n--- TEST 10: HTML forensic report ---")
    html_real = utils.build_report_html(res_real, utils.load_training_metadata())
    html_noface = utils.build_report_html(res_noface, utils.load_training_metadata())
    html_fake = utils.build_report_html(res_fake, utils.load_training_metadata())

    # Verify no stale hardcoded FAKE in REAL report and check escaping
    import html as html_lib
    note_real_escaped = html_lib.escape("Warm colours (red/yellow) mark regions that most influenced the model's REAL prediction.")
    note_fake_escaped = html_lib.escape("Warm colours (red/yellow) mark regions that most influenced the model's FAKE prediction.")

    t10_checks = [
        "Prediction</th><td>REAL" in html_real,
        "Grad-CAM heatmap (REAL)" in html_real,
        note_real_escaped in html_real,
        "UNSUPPORTED &mdash; NO FACE DETECTED" in html_noface,
        "Model status</th><td>NOT RUN" in html_noface,
        "Grad-CAM status</th><td>NOT RUN" in html_noface,
        "Prediction</th><td>FAKE" in html_fake,
        "Grad-CAM heatmap (FAKE)" in html_fake,
        note_fake_escaped in html_fake,
    ]
    t10_pass = all(t10_checks)
    print(f"HTML Checks: {sum(t10_checks)} / {len(t10_checks)} passed")
    print(f"TEST 10 Result: {'PASS' if t10_pass else 'FAIL'}")
    results["test_10_html_report"] = "PASS" if t10_pass else "FAIL"

    print("\n" + "=" * 60)
    print(f"SUMMARY: {sum(1 for v in results.values() if v == 'PASS')} / {len(results)} PIPELINE TESTS PASSED")
    print("=" * 60)
    return results


if __name__ == "__main__":
    res = run_pipeline_qa()
    all_ok = all(v == "PASS" for v in res.values())
    sys.exit(0 if all_ok else 1)
