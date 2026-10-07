# CyberGuard &mdash; Evaluation Dataset Recommendation

This document specifies the required structure and methodology for acquiring an independent, production-grade validation benchmark for CyberGuard.

---

## 1. Directory Structure

To evaluate an external benchmark dataset, organize files into the following standardized layout:

```text
dataset/ (or test_data/)
├── real/
│   ├── real_0001.jpg
│   ├── real_0002.png
│   └── ...
└── fake/
    ├── fake_0001.jpg
    ├── fake_0002.png
    └── ...
```

The evaluation script (`scripts/evaluate_model.py`) automatically discovers, validates, and evaluates all `.jpg`, `.jpeg`, `.png`, and `.webp` images in these folders.

---

## 2. Recommended Benchmark Sources

To obtain unbiased, statistically rigorous validation:

1. **FaceForensics++ (FF++)**:
   - Industry standard benchmark containing Deepfakes, Face2Face, FaceSwap, and NeuralTextures.
   - High-, medium-, and low-compression versions (C0, C23, C40).
2. **Celeb-DF (v2)**:
   - High-quality deepfake video frames featuring subtle blending and diverse lighting.
3. **DiffusionFace / FakeAVCeleb**:
   - Modern diffusion-based (Latent Diffusion, Midjourney) synthesized portraits to test cross-generator generalization.
4. **Flickr-Faces-HQ (FFHQ) / LFW**:
   - Clean, diverse authentic human faces across demographic groups, age brackets, and lighting conditions.

---

## 3. Recommended Dataset Size & Balance

| Metric | Minimum Recommended | Production Target |
|--------|---------------------|-------------------|
| Authentic Faces (`real/`) | 500 images | 2,500+ images |
| Deepfake Faces (`fake/`) | 500 images | 2,500+ images |
| Balance Ratio | Exactly 1:1 | Exactly 1:1 |
| Generator Diversity | &ge; 3 generator architectures | &ge; 5 generator architectures |
| Quality Variations | Uncompressed + Compressed (JPEG Q50, Q75) | Diverse resolutions |

---

## 4. Integrity Requirements

- **Disjoint Partitions**: No subject (individual person) appearing in the training split should appear in the evaluation split.
- **Deduplication**: Run `dataset_integrity_report.json` checks to verify 0% byte or perceptual overlap before benchmarking.
- **Full-Scene Diversity**: Include natural scenes with natural head margins (not only 224x224 tight crops) to evaluate both face detection localization and deepfake classification end-to-end.

---

## 5. Automated Execution

Once the new images are placed in `test_data/real` and `test_data/fake`, simply execute:

```bash
python scripts/evaluate_model.py
```

The validation engine will automatically:
1. Validate image format integrity
2. Run inference & face localization
3. Recalculate Confusion Matrix, ROC-AUC, PR-AUC, and F1-score
4. Regenerate CSV sweeps, PNG charts, and the forensic HTML report
