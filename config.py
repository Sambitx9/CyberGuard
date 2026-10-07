"""Central configuration for CyberGuard.

Every path is derived from this file's location, so the project can be moved
or cloned anywhere without editing anything.
"""
from __future__ import annotations

from pathlib import Path

# --------------------------------------------------------------------------- #
# Branding
# --------------------------------------------------------------------------- #
APP_NAME = "CYBERGUARD"
APP_SUBTITLE = "AI-Powered Deepfake Detection & Prevention"
TAGLINE = "Detect. Analyze. Verify."
VERSION = "0.1.0-mvp"

DISCLAIMER = (
    "CyberGuard is a demonstration AI system. Results should not be treated "
    "as definitive forensic evidence."
)
CONFIDENCE_NOTE = (
    "Confidence reflects model certainty, not proof that an image is authentic or manipulated."
)

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "model"
MODEL_PATH = MODEL_DIR / "cyberguard_model.keras"
METADATA_PATH = MODEL_DIR / "training_metadata.json"
DATASET_DIR = BASE_DIR / "dataset"
REPORTS_DIR = BASE_DIR / "reports"
ASSETS_DIR = BASE_DIR / "assets"

# --------------------------------------------------------------------------- #
# Model / training
# --------------------------------------------------------------------------- #
# Class order matters: Keras assigns label 0 to the first name and 1 to the
# second. The network's single sigmoid output is therefore P(FAKE).
CLASS_NAMES = ("real", "fake")
IMG_SIZE = (224, 224)  # MobileNetV2 default input size

BATCH_SIZE = 32
VALIDATION_SPLIT = 0.2
SEED = 42

HEAD_EPOCHS = 10          # phase 1: train only the new classification head
FINE_TUNE_EPOCHS = 10     # phase 2: also fine-tune the top backbone layers
FINE_TUNE_LAYERS = 30     # how many backbone layers to unfreeze in phase 2
HEAD_LR = 1e-3
FINE_TUNE_LR = 1e-5
EARLY_STOP_PATIENCE = 4

MIN_IMAGES_PER_CLASS = 20          # hard minimum to attempt training
RECOMMENDED_IMAGES_PER_CLASS = 500  # below this we warn about weak results
TRAIN_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp")  # formats Keras can decode

# Layer names are shared between train_model.py (builds the network) and
# explainability.py (Grad-CAM looks the layers up by name).
LAYER_PREPROCESS = "preprocess"
LAYER_BACKBONE = "backbone"
CAM_HEAD_LAYERS = ("gap", "dropout", "logit")  # layers between backbone and logit

# --------------------------------------------------------------------------- #
# Inference
# --------------------------------------------------------------------------- #
UPLOAD_EXTENSIONS = ("jpg", "jpeg", "png", "webp")
ALLOWED_FORMATS = {"JPEG", "PNG", "WEBP"}  # checked against real file content
MAX_UPLOAD_MB = 15
MIN_IMAGE_SIDE = 32
MAX_IMAGE_PIXELS = 50_000_000  # guards against decompression bombs

FAKE_THRESHOLD = 0.5  # P(FAKE) >= threshold -> FAKE

# Risk is derived from the model's FAKE probability (0-100):
#   0-39 LOW, 40-69 MEDIUM, 70-100 HIGH
RISK_MEDIUM_START = 40
RISK_HIGH_START = 70

# Face handling. The model is trained on whatever images are in dataset/, so
# inference must match: leave this False unless your training images are face
# crops too (see README, "Dataset setup").
USE_FACE_CROP = False
FACE_CROP_MARGIN = 0.25
FACE_DETECT_MAX_SIDE = 800
