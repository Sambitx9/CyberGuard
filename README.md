# CYBERGUARD
### AI-Powered Deepfake Detection & Prevention
*Detect. Analyze. Verify.*

CyberGuard is a **hackathon MVP**: a local web app that uses a real transfer-learning model (MobileNetV2) to estimate whether an uploaded face image is **REAL** or **FAKE**. It is a demonstration system, **not forensic software**. It ships with **no pre-trained model and no dataset**: you train it yourself, and until you do, the app shows `MODEL NOT TRAINED` instead of making predictions.

## Features
- Image upload (JPG, JPEG, PNG, WEBP) with real content validation (corrupt or renamed files are rejected)
- REAL / FAKE prediction, model confidence, LOW / MEDIUM / HIGH risk
- Face detection (OpenCV Haar cascade) with bounding boxes
- **Grad-CAM** heatmap; if Grad-CAM fails, a clearly labelled Error Level Analysis (ELA) fallback is shown, never called Grad-CAM
- Downloadable HTML analysis report (generated in memory)
- Session-only history with Clear History
- Dark dashboard: Dashboard, Analyze Media, About, System Information
- 100% local: images are never sent to external services and never written to disk
- CLI: `python predict.py image.jpg`

## How results are computed
- The network outputs `P(FAKE)` (one sigmoid unit; class order is `real=0`, `fake=1`).
- **Prediction** = FAKE if `P(FAKE) >= 0.5`, otherwise REAL.
- **Confidence** = `max(P(FAKE), 1 - P(FAKE))`. This is the model's certainty, **not proof** of authenticity or manipulation.
- **Risk** is based on `P(FAKE)` as a 0-100 score: 0-39 LOW, 40-69 MEDIUM, 70-100 HIGH. So a confident REAL prediction is LOW risk.

## Architecture
```
Upload -> validate -> detect face -> preprocess -> MobileNetV2 inference
       -> prediction / confidence / risk -> Grad-CAM -> UI + report
```
The in-model `preprocess` layer scales pixels to [-1, 1], so training and inference cannot disagree on normalisation. `src/detector.py` defines a `BaseDetector` interface so video, audio, lip-sync, C2PA, watermark checks, an API, or a browser extension can be added later as new detectors without touching the UI.

## Tech stack
Python 3.11+, TensorFlow/Keras (MobileNetV2), OpenCV, NumPy, Pillow, scikit-learn, Matplotlib, Streamlit.

## Folder structure
```
CyberGuard/
├── app.py              Streamlit dashboard
├── train_model.py      dataset checks, training, evaluation, graphs
├── predict.py          command-line prediction
├── config.py           paths, thresholds, hyper-parameters
├── requirements.txt
├── .streamlit/config.toml   dark theme
├── model/              cyberguard_model.keras + training_metadata.json (created by training)
├── dataset/real/  dataset/fake/   your training images
├── src/
│   ├── preprocessing.py   validation, face detection, model input
│   ├── detector.py        model loading + analysis pipeline
│   ├── explainability.py  Grad-CAM and ELA fallback
│   └── utils.py           risk bands, text, HTML report
├── assets/             logo
└── reports/            training graphs
```

## Installation
Requires Python 3.11 or newer (TensorFlow does not support every newest Python release; use 3.11 or 3.12 if install fails).

```bash
python -m venv venv
```
Windows:
```bash
venv\Scripts\activate
```
macOS / Linux:
```bash
source venv/bin/activate
```
Then:
```bash
pip install -r requirements.txt
```

## Dataset & Benchmark
CyberGuard includes an automated dataset acquisition pipeline via `download_and_organize_dataset.py` fetching from the public benchmark [`desireemcv/face-real-vs-fake`](https://huggingface.co/datasets/desireemcv/face-real-vs-fake):
- **Source:** Hugging Face benchmark `desireemcv/face-real-vs-fake`
- **Class Labels:** 224x224 RGB faces (REAL vs FAKE)
- **Dataset Structure:**
  - `dataset/real/` (250 images, authentic human faces)
  - `dataset/fake/` (250 images, AI-generated/deepfake faces)
  - `test_data/real/` (50 unseen held-out test images)
  - `test_data/fake/` (50 unseen held-out test images)

To download and organize the dataset:
```bash
python download_and_organize_dataset.py
```

## Model Training & Architecture
- **Architecture:** MobileNetV2 backbone (transfer learning from ImageNet) + custom binary classification head (`GlobalAveragePooling2D` -> `Dropout(0.3)` -> `Dense(1)` -> `Sigmoid`).
- **Input Dimensions:** 224 x 224 x 3 RGB images.
- **Normalization:** In-model `Rescaling(1/127.5, offset=-1.0)` guaranteeing training and inference preprocessing parity.
- **Explainability:** Native Grad-CAM gradient tracking back to the pre-sigmoid logit, with robust Error Level Analysis (ELA) fallback.

To train the model:
```bash
python train_model.py --epochs-head 5 --epochs-finetune 4
```

Outputs:
- Model weights: `model/cyberguard_model.keras`
- Training metadata: `model/training_metadata.json`
- Visualization plots: `reports/training_history.png`, `reports/confusion_matrix.png`

## Evaluation Results
The model was rigorously evaluated on 100 completely unseen test images (`test_data/`):
- **Test Accuracy:** 88.00%
- **Test Precision (Fake detection):** 100.00% (Zero false positives on real images)
- **Test Recall:** 76.00%
- **F1 Score:** 86.36%
- **ROC AUC:** 0.9588
- **Held-out Validation Accuracy:** 92.00% (Validation AUC: 0.9943)

Evaluation metrics are saved at `reports/model_evaluation.json`. To re-evaluate:
```bash
python evaluate_model.py
```

## Running the Application
### 1. Interactive Streamlit Web Dashboard
```bash
streamlit run app.py
```
Access the application at `http://localhost:8501`.

### 2. Command-Line Inference (CLI)
```bash
python predict.py path/to/image.jpg --heatmap reports/heatmap.png --report reports/report.html
```

## Privacy
Uploads are decoded and analysed in memory on your machine. No external API is called and uploads are not written to disk. The report is built in memory and downloaded by your browser; it is not stored on the server.

## Troubleshooting
| Problem | Fix |
|---|---|
| App shows `MODEL NOT TRAINED` | Add dataset images and run `python train_model.py`, then refresh the app. |
| `[dataset error] Missing folder` / too few images | Create `dataset/real` and `dataset/fake` and add images (see above). |
| `corrupted/unreadable image(s)` | Delete the listed files and retry. |
| `ModuleNotFoundError: tensorflow` | Activate the venv and run `pip install -r requirements.txt`. Use Python 3.11 or 3.12. |
| Error downloading MobileNetV2 weights | Connect to the internet for the first training run, or use `--no-pretrained` (poor accuracy). |
| `ModuleNotFoundError: src` / `config` | Run commands from inside the `CyberGuard` folder. |
| Model failed to load / wrong input size | Retrain with the same TensorFlow/Keras major version, and check `IMG_SIZE` in `config.py`. |
| Heatmap titled "ELA (fallback)" | Grad-CAM could not run for that image; the reason is shown under the image. |
| Port already in use | `streamlit run app.py --server.port 8502` |
| Native Windows GPU not used | Recent TensorFlow versions are CPU-only on native Windows. MobileNetV2 trains fine on CPU, just slower. |

## Limitations
- **Only images** are analysed. Video, audio, lip-sync, C2PA/content-credentials and watermark checks are **not implemented**.
- Accuracy depends wholly on your training data. Detectors generalise poorly to unseen generators, compression and post-processing.
- Confidence scores can be badly calibrated and wrong while appearing certain.
- The Haar-cascade face detector misses profile, tilted, small or occluded faces; if no face is found the whole image is analysed and a warning is shown.
- Grad-CAM shows where the model looked, not proof of editing.
- Random train/validation splits can leak near-duplicate frames and overstate accuracy.
- A REAL result does not prove an image is authentic; a FAKE result is not evidence of manipulation.

> CyberGuard is a demonstration AI system. Results should not be treated as definitive forensic evidence.

## Future improvements
Video frame sampling and temporal models, audio deepfake and lip-sync detection, C2PA verification, invisible watermark checks, a REST API, a browser extension, face-crop training pipeline, probability calibration, cross-dataset evaluation, and ensemble models.
