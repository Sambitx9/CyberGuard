# CyberGuard &mdash; External Benchmark Dataset Instructions

This document explains how to supply an independent external validation dataset for CyberGuard Phase 3 evaluation.

---

## 1. Directory Structure

Place your labeled evaluation images into the dedicated `external_test_data/` directory:

```text
CyberGuard/
├── external_test_data/
│   ├── real/
│   │   ├── authentic_face_001.jpg
│   │   ├── authentic_face_002.png
│   │   └── ...
│   └── fake/
│       ├── deepfake_face_001.jpg
│       ├── deepfake_face_002.png
│       └── ...
```

---

## 2. Supported Formats & Constraints

- **Supported file types**: `JPG`, `JPEG`, `PNG`, `WEBP`
- **Integrity requirement**: Must be independent from `dataset/` (training) and `test_data/` (smoke/initial test).
- **Separation**: Do NOT copy external images into `dataset/`. The evaluation framework evaluates `external_test_data/` strictly out-of-sample.

---

## 3. Running External Evaluation

Once images are deposited in `external_test_data/real` and `external_test_data/fake`, execute:

```bash
python scripts/evaluate_external.py
```

The system will automatically:
1. Audit image integrity, SHA-256 collisions, and train/test leakage (`reports/external_dataset_integrity.json`).
2. Run production preprocessing and deterministic face localization.
3. Compute Accuracy, Precision, Recall, Specificity, F1, FPR, FNR, ROC-AUC, and PR-AUC.
4. Generate `reports/external_model_evaluation.html`, `reports/external_confusion_matrix.png`, and `reports/external_errors.csv`.
5. Update the Model Decision Gate status to `EXTERNALLY VALIDATED` (or flag if improvements are required).
