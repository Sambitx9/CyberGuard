# CyberGuard &mdash; Model Limitations

This document outlines the known technical and operational limitations of CyberGuard's deepfake detection model (`model/cyberguard_model.keras`), based strictly on quantitative evaluation findings from Phase 2.

---

## 1. Evaluation Sample Size & Statistical Status

- **Status**: **PRELIMINARY**
- **Evaluation Size**: 100 test images (50 REAL, 50 FAKE).
- **Caution**: An evaluation set of 100 images provides directional performance indications but lacks statistical power to claim production reliability across diverse global populations, generative architectures, and adversarial scenarios.

---

## 2. Asymmetric Class Sensitivity (High False Negative Rate)

- **Authentic Face Retention (Specificity)**: **100.0%** (50/50 REAL classified correctly, 0 False Positives).
- **Deepfake Detection Recall**: **76.0%** (38/50 FAKE classified correctly, 12 False Negatives).
- **Impact**: The model is conservative and strongly biased toward predicting REAL. Consequently, approximately **24% of deepfakes slip through unnoticed** as authentic faces at the default 0.50 threshold.

---

## 3. High-Confidence Misclassifications & Calibration

- **Observation**: Out of 12 false negatives (deepfakes misclassified as authentic), several received confidence scores exceeding **90%**.
- **Calibration Risk**: Model softmax/sigmoid output probabilities are uncalibrated. A prediction of `REAL (94.2% confidence)` does not guarantee 94% true empirical certainty.
- **Recommendation**: End-users must adhere to CyberGuard's operational guidance: *"Confidence reflects model certainty, not proof that an image is authentic or manipulated."* Probability calibration (e.g. Platt scaling) should be implemented in future versions.

---

## 4. Upstream Dataset Overlap & Leakage

- **Exact Duplicates**: 2 test images (`fake_test_0004.jpg`, `fake_test_0009.jpg`) are bit-for-bit identical to training images (`fake_train_0001.jpg`, `fake_train_0004.jpg`).
- **Perceptual Near-Duplicates**: 16 test images exhibit perceptual hash similarity (Hamming distance &le; 2 on dHash) to training samples.
- **Cause**: The upstream source repository (`desireemcv/face-real-vs-fake` on Hugging Face) contains overlapping synthetic faces across its train and test partitions.
- **Impact**: Test accuracy is marginally inflated by memorization on these duplicated samples.

---

## 5. Face Detector Dependency & Close-Up Crops

- **Observation**: OpenCV's Haar Cascade classifier (`haarcascade_frontalface_default.xml`) missed faces on **36% of pre-cropped 224x224 test images** because the bounding boxes lacked forehead, hair, and chin contextual margins.
- **Safety Interception**: Whenever Haar Cascade misses a face, CyberGuard's pipeline guard safely intercepts the image and returns `NO FACE DETECTED` with model inference `NOT RUN`.
- **Limitation**: While this guard prevents misleading whole-image classifications, tight face crops or macro portraits frequently trigger the unsupported guard. An upgraded face detector (such as MTCNN, SSD, or RetinaFace) is required to reliably localize tight crops.

---

## 6. Generalization Across Generators

- **Scope**: The current model was trained exclusively on a single synthetic face distribution (primarily StyleGAN/ProGAN artifacts).
- **Unverified Domains**:
  - Diffusion-based deepfakes (Stable Diffusion, Midjourney, Flux)
  - Face-swap deepfakes (Roop, SimSwap, DeepFaceLab)
  - Video-compression artifacts (H.264 / H.265 compression noise)
  - Heavily occluded, low-resolution, or profile-view faces
- Performance on these unrepresented generative architectures is unmeasured and should be assumed lower than benchmark figures.
