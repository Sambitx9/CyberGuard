# CyberGuard Phase 5 Final QA & Release Audit

**Audit Date:** 2026-10-07  
**Auditor Roles:** Principal ML Engineer, Computer Vision Engineer, QA Lead, Security Engineer, Release Engineer  
**Target Checkpoint:** `model/cyberguard_model.keras`  
**Current Release Candidate Status:** **RELEASE CANDIDATE — EXTERNAL VALIDATION REQUIRED**  

---

## 1. Executive Verdict

CyberGuard has successfully satisfied all core software engineering, pipeline safety, security, explainability, and regression criteria. The core inference pipeline deterministically intercepts non-face images, selects primary faces in multi-face scenes, generates class-congruent Grad-CAM heatmaps, and cleanly rejects malicious, oversized, and corrupted inputs.

However, **independent external validation cannot be declared at this time because no external dataset has been deposited into `external_test_data/`**. In accordance with CyberGuard's strict scientific integrity and zero-fabrication standards:
* No external metrics have been fabricated or extrapolated.
* The system is classified as **RELEASE CANDIDATE — EXTERNAL VALIDATION REQUIRED**.
* Full production deployment is conditioned upon evaluating independent multi-generator deepfake datasets.

---

## 2. Dataset / Validation Status

* **External Dataset Path:** `external_test_data/` (`real/`, `fake/`)
* **Real Images Present:** 0
* **Fake Images Present:** 0
* **External Validation State:** **BLOCKED — DATASET NOT PROVIDED**
* **Integrity Audit:** Verified that directories exist and are empty. Ingestion guidelines and verification scripts are preserved in [`DATASET_READY.md`](file:///c:/Users/HP/Desktop/CyberGuard/DATASET_READY.md) and [`reports/external_dataset_integrity.json`](file:///c:/Users/HP/Desktop/CyberGuard/reports/external_dataset_integrity.json).

---

## 3. Model Integrity

The production model file was inspected and audited before and after testing:

* **File Path:** [`model/cyberguard_model.keras`](file:///c:/Users/HP/Desktop/CyberGuard/model/cyberguard_model.keras)
* **File Existence:** Verified
* **File Size:** `21,752,160` bytes (Exact match with expected baseline)
* **SHA-256 Checksum:** `E7422FCA2F61706F5F0379453469C8D536A7D343DF1A185214AF8F48028539A6` (Exact match with expected baseline)
* **Architecture Load Test:** Model loaded cleanly via Keras 3.x with MobileNetV2 backbone.
* **Inference Test:** Forward pass on dummy tensor $(1, 224, 224, 3)$ returned output shape $(1, 1)$ without errors.
* **Status:** **PASS — UNCHANGED**

---

## 4. Internal Benchmark

The internal labeled test benchmark consists of 100 images (50 REAL, 50 FAKE) evaluated at the fixed production threshold $P(\text{FAKE}) \ge 0.50$:

| Metric | Classifier Performance (on 100 faces) |
| :--- | :---: |
| **Accuracy** | 88.00% |
| **Precision** | 100.00% |
| **Recall (FAKE)** | 76.00% |
| **Specificity (REAL)** | 100.00% |
| **F1-Score** | 86.36% |
| **False Positive Rate (FPR)** | 0.00% (0 / 50) |
| **False Negative Rate (FNR)** | 24.00% (12 / 50) |
| **Balanced Accuracy** | 88.00% |
| **ROC-AUC** | 0.9588 |
| **PR-AUC** | 0.9686 |

* **Known Dataset Leakage:** Audit revealed 2 exact train/test duplicate pairs and 16 perceptual near-duplicates within the internal dataset source (140K Real and Fake Faces).
* **Engineering Constraint:** These internal metrics **must not** be presented as generalized production accuracy due to single-generator bias and internal duplicate leakage.

---

## 5. External Benchmark

* **External Dataset Status:** **NOT PROVIDED (BLOCKED)**
* **External Accuracy:** `N/A`
* **External Precision:** `N/A`
* **External Recall:** `N/A`
* **External F1-Score:** `N/A`
* **External ROC-AUC:** `N/A`
* **External PR-AUC:** `N/A`

In compliance with the Phase 5 protocol, no synthetic metrics have been generated.

---

## 6. Face Detector Comparison

Face detection performance was benchmarked across all 100 images on identical test inputs:

| Detector Backend | Face Detection Rate | Missed Faces | End-to-End Pipeline Accuracy | Classifier Acc on Localized Faces | Precision (FAKE) | Recall (FAKE) | F1-Score | Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **OpenCV Haar Cascade** | 64.0% (64/100) | 36 / 100 | **56.0%** (56/100) | 87.5% (56/64) | 96.3% | 52.0% | 0.675 | Active Production Default |
| **Ensemble Haar (Default+Alt2)** | 71.0% (71/100) | 29 / 100 | **63.0%** (63/100) | 88.7% (63/71) | 96.7% | 58.0% | 0.725 | Benchmarked |
| **OpenCV YuNet (DNN)** | **94.0%** (94/100) | **6 / 100** | **82.0%** (82/100) | 87.2% (82/94) | **100.0%** | **72.0%** | **0.837** | Benchmarked & Verified |

### Critical Clarifications:
1. **Pipeline Accuracy vs. Classifier Accuracy:**
   * **82.0%** represents the **end-to-end pipeline accuracy** (82 correct classifications out of all 100 submitted test images, with 6 unlocalized images intercepted as unsupported).
   * **87.2%** represents **classifier accuracy on successfully localized faces** (82 correct classifications out of the 94 localized face crops). These metrics use different denominators and are never merged.
2. **YuNet Status & Terminology:**
   * YuNet delivers **+26 percentage points in measured end-to-end accuracy on the internal 100-image benchmark** over baseline Haar Cascade.
   * YuNet Status is formally designated as: **PROMISING BUT NOT EXTERNALLY VALIDATED**.
   * **Recommendation:** Retain Haar Cascade as the verified zero-weight baseline for the initial release candidate; plan a seamless transition to YuNet as the default face detector in the first update once external validation is completed.

---

## 7. Calibration Audit & Discrepancy Resolution

### Identified Discrepancy:
* **Phase 3 Report (`phase3_validation_report.html`):** Isotonic Brier = 0.0242, ECE = 0.00%.
* **Phase 4 Summary Narrative:** Isotonic Brier = 0.0371, ECE = 4.92%; Platt Brier = 0.0332, ECE = 4.38%.

### Root Cause Audit:
```
CALIBRATION DISCREPANCY:
CAUSE = Reporting transcription error in Phase 4 conversational narrative vs canonical on-disk artifact.
```
1. **Canonical Source of Truth:** On disk, [`reports/calibration_analysis.json`](file:///c:/Users/HP/Desktop/CyberGuard/reports/calibration_analysis.json) generated by `scripts/calibrate_model.py` on the 100-sample validation split records:
   * **Raw Model:** Brier = `0.0578`, ECE = `10.76%`, MCE = `75.85%`
   * **Platt Scaling:** Brier = `0.0332`, ECE = `4.38%`, MCE = `44.07%`
   * **Isotonic Regression:** Brier = `0.0242`, ECE = `0.00%`, MCE = `0.00%`
2. **Mathematical Mechanics:** In-sample piecewise constant Isotonic regression fitted directly on the 100 validation samples achieves an empirical ECE of 0.00% because bin-averaged predictions match sample empirical probabilities by construction.
3. **Out-of-Sample Empirical Evaluation:** When tested on the independent 100 test images:
   * **Raw Model:** Brier = `0.0902`, ECE = `12.33%`
   * **Platt Scaling:** Brier = `0.0847`, ECE = `9.33%`
   * **Isotonic Regression:** Brier = `0.1095`, ECE = `12.48%`
4. **Conclusion:** Isotonic regression overfits on small calibration sets, causing out-of-sample degradation. Platt scaling generalizes reliably. The numbers "0.0371" and "4.92%" appeared solely in narrative commentary and do not exist in any JSON or codebase artifacts.

---

## 8. Error Analysis

Audit of misclassified test images on localized crops:
* **False Positives:** **0** (No authentic face misclassified as deepfake at threshold 0.50).
* **False Negatives:** **12** (12 deepfakes classified as REAL).
* **High-Confidence False Negatives ($\ge 90\%$ Real Confidence):**
  1. `fake_test_0040.jpg` — $P(\text{FAKE}) = 1.37\%$ ($98.63\%$ Real confidence)
  2. `fake_test_0033.jpg` — $P(\text{FAKE}) = 6.63\%$ ($93.37\%$ Real confidence)
  3. `fake_test_0041.jpg` — $P(\text{FAKE}) = 7.89\%$ ($92.11\%$ Real confidence)

*Ledger saved to [`reports/phase4_high_confidence_errors.csv`](file:///c:/Users/HP/Desktop/CyberGuard/reports/phase4_high_confidence_errors.csv).* These failure modes involve subtle generative synthesis artifacts that evade boundary convolutions.

---

## 9. Pipeline QA

All 10 critical execution paths were tested via [`scripts/test_pipeline_qa.py`](file:///c:/Users/HP/Desktop/CyberGuard/scripts/test_pipeline_qa.py):

| Test ID | Path Tested | Expected Behavior | Measured Result | Verdict |
| :---: | :--- | :--- | :--- | :---: |
| **TEST 1** | Valid Single-Face REAL | Face detected, REAL prediction, LOW risk, Grad-CAM (REAL) | Face=1, REAL, Conf=79.5%, Risk=LOW, Grad-CAM (REAL) | **PASS** |
| **TEST 2** | Valid Single-Face FAKE | Face detected, FAKE prediction, HIGH risk, Grad-CAM (FAKE) | Face=1, FAKE, Conf=99.0%, Risk=HIGH, Grad-CAM (FAKE) | **PASS** |
| **TEST 3** | No-Face Infographic | Model=0, Cam=0, status=UNSUPPORTED, prediction=NO FACE DETECTED | Model=0 calls, Cam=0 calls, NO FACE DETECTED, Risk=N/A | **PASS** |
| **TEST 4** | Multiple Faces (2 & 3) | Deterministic primary face selection, largest face analyzed, warning logged | Largest bounding box analyzed, secondary faces boxed, warnings logged | **PASS** |
| **TEST 5** | Corrupted Image | Graceful rejection via `ImageValidationError`, no crash, no model call | `ImageValidationError` raised, safe rejection | **PASS** |
| **TEST 6** | Unsupported File Type | Rejection of `.exe` / non-image formats | `ImageValidationError: Unsupported file type '.exe'` | **PASS** |
| **TEST 7** | Extremely Small (<32px) | Safe handling, dimension guard triggered | `ImageValidationError: Image is too small (16x16)` | **PASS** |
| **TEST 8** | Very Large Image (>15MB)| Safe size rejection before decode | `ImageValidationError: File is larger than the 15 MB limit` | **PASS** |
| **TEST 9** | CLI Prediction | Equivalent behavior to Web UI (`predict.py`) | Exit code 0, correct console output and report files | **PASS** |
| **TEST 10**| HTML Forensic Report | Correct fields, dynamic Grad-CAM wording, HTML escaping verified | 9/9 HTML sanity and security assertions verified | **PASS** |

---

## 10. Grad-CAM QA

* **Dynamic Class Logit Targeting:** In [`src/explainability.py`](file:///c:/Users/HP/Desktop/CyberGuard/src/explainability.py), Grad-CAM calculates gradients with respect to the predicted class logit:
  * For `prediction == "REAL"`: Target logit is $-fake\_logit$ (attributing evidence supporting authenticity).
  * For `prediction == "FAKE"`: Target logit is $+fake\_logit$ (attributing evidence supporting manipulation).
* **Caption & Text Congruence:**
  * REAL results generate caption: `Grad-CAM heatmap (REAL)`
  * FAKE results generate caption: `Grad-CAM heatmap (FAKE)`
* **Disclaimer Integrity:** Verified that all explanations state:
  > *"Warm colours (red/yellow) mark regions that most influenced the model's prediction. This shows where the model looked, not proof that those regions were edited."*

---

## 11. Security QA

Application-level security audit across all attack surfaces:
* **Upload Ingestion:** Handled strictly in-memory (`io.BytesIO`); no unauthenticated disk writes.
* **Path Traversal:** Filenames sanitized via `safe_stem()` regex (`[^a-zA-Z0-9_-]`); directory traversal attempts (`../../`) neutered.
* **Arbitrary Code Execution:** Extension whitelist enforcement (`.jpg`, `.jpeg`, `.png`, `.webp`) plus Pillow format verification (`verify()` + `load()`). Executables, scripts, and polyglots are rejected.
* **HTML Injection:** All report variables (`prediction`, `risk`, `warnings`, `visual.title`, `visual.note`) pass through `html.escape()`.
* **Memory & DoS Protection:**
  * File size clamped at `15 MB` before read.
  * Image dimensions restricted by `Image.MAX_IMAGE_PIXELS = 100_000_000` to prevent decompression bomb attacks.
* **Secrets & Credentials:** Zero committed API keys, tokens, or plaintext credentials detected.

---

## 12. Regression Tests

Executed test suites in [`tests/`](file:///c:/Users/HP/Desktop/CyberGuard/tests):

### Suite A: [`test_evaluation_regression.py`](file:///c:/Users/HP/Desktop/CyberGuard/tests/test_evaluation_regression.py)
* `[PASS]` Model weights integrity verified
* `[PASS]` No-face bypass verified (model=0 calls, cam=0 calls)
* `[PASS]` Real face inference & Grad-CAM verified
* `[PASS]` Fake face inference & Grad-CAM verified
* `[PASS]` Invalid file rejection verified
* `[PASS]` All evaluation artifacts exist and non-empty

### Suite B: [`test_phase3_regression.py`](file:///c:/Users/HP/Desktop/CyberGuard/tests/test_phase3_regression.py)
* `[PASS]` Test 1: Model weight integrity verified (Size & SHA-256 exact match)
* `[PASS]` Test 2: No-face bypass verified (model=0 calls, cam=0 calls)
* `[PASS]` Test 3: Face inference verified (REAL & FAKE, dynamic Grad-CAM titles)
* `[PASS]` Test 4: Invalid file rejection verified (`ImageValidationError`)
* `[PASS]` Test 5: Multi-face deterministic policy verified (1, 2, 3 faces)
* `[PASS]` Test 6: Face detector abstraction verified (Haar, Ensemble, YuNet available)
* `[PASS]` Test 7: Calibration data separation and artifacts verified
* `[PASS]` Test 8: External dataset status & production threshold (0.50) verified

**Regression Summary:**
* **Total Tests:** 14
* **Passed:** 14 (100%)
* **Failed:** 0
* **Skipped:** 0

---

## 13. Reproducibility

* **Model Checkpoint:** `model/cyberguard_model.keras`
* **Test Dataset:** `test_data/` (50 Real, 50 Fake)
* **Threshold:** Fixed at 0.50
* **Evaluation Runner:** `evaluate_model.py` re-executed independently.
* **Result:** Yielded identical results (Accuracy: 88.00%, Precision: 100.00%, Recall: 76.00%, F1: 86.36%, ROC-AUC: 0.9588).
* **Reproducibility Status:** **PASS**

---

## 14. Release Gates

| Gate ID | Release Gate Name | Required Condition | Actual Finding | Gate Status |
| :---: | :--- | :--- | :--- | :---: |
| **GATE 1** | Model Integrity | Size & SHA-256 match baseline exactly | 21,752,160 bytes, SHA-256 exact match | **PASS** |
| **GATE 2** | Pipeline Functionality | 10/10 pipeline QA tests pass | 10/10 passed without regressions | **PASS** |
| **GATE 3** | No-Face Guard | 0 model calls, 0 Grad-CAM calls on no-face | Intercepted with 0 model calls | **PASS** |
| **GATE 4** | Multi-Face Handling | Deterministic primary selection & warning | Verified on 1, 2, and 3-face images | **PASS** |
| **GATE 5** | Grad-CAM Correctness | Target logit matches predicted class | Logit flipped for REAL; captions congruent | **PASS** |
| **GATE 6** | Report Correctness | HTML escaping, accurate metadata | 9/9 security & rendering assertions pass | **PASS** |
| **GATE 7** | Security Baseline | Memory limits, decompression bomb guard | Clean validation errors on bad inputs | **PASS** |
| **GATE 8** | Regression Suite | 100% pass on all existing test suites | 14/14 tests passed | **PASS** |
| **GATE 9** | Internal Reproducibility | Baseline metrics reproduce deterministically | Exactly matches 88.00% benchmark | **PASS** |
| **GATE 10**| Calibration Consistency| Root cause of discrepancy documented | Discrepancy resolved; Platt validated | **PASS** |
| **GATE 11**| Independent Validation | Independent external dataset evaluated | `external_test_data/` is unpopulated | **BLOCKED** |

---

## 15. Known Limitations

1. **Unpopulated External Benchmark:** Performance against modern unseen generators (e.g. Midjourney v6, StyleGAN3, Diffusion-based face swappers) has not yet been externally validated.
2. **Haar Cascade Crop Sensitivity:** Haar Cascade misses 36% of tight crops that lack full forehead/chin margins (mitigated when upgraded to YuNet).
3. **High-Confidence False Negatives:** Three subtle deepfakes bypassed the classifier with $>90\%$ confidence (`fake_test_0040.jpg`, `fake_test_0033.jpg`, `fake_test_0041.jpg`).
4. **Single-Face Focus:** In multi-face images, only the largest primary face is classified; secondary faces are not analyzed.

---

## 16. Release Decision

$$\mathbf{RELEASE\ CANDIDATE\ —\ EXTERNAL\ VALIDATION\ REQUIRED}$$

### Formal Justification:
The CyberGuard codebase has met all technical, architectural, explainability, safety, and security requirements. However, production readiness cannot be declared without empirical validation on an independent external dataset. Adhering to the zero-fabrication charter, the system is frozen as a **Release Candidate**.

---

## 17. Exact Next Action

Deposit labeled test images into `external_test_data/real/` and `external_test_data/fake/`, then run `python scripts/evaluate_external.py` to complete the final external validation gate.
