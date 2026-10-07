"""Upload validation, face detection and model-input preparation."""
from __future__ import annotations

import io
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

import config

Image.MAX_IMAGE_PIXELS = config.MAX_IMAGE_PIXELS

Box = tuple[int, int, int, int]  # x, y, width, height


class ImageValidationError(ValueError):
    """Raised when an upload cannot be used. The message is safe to show to users."""


@dataclass
class ValidatedImage:
    """A decoded, EXIF-corrected RGB image plus basic file facts."""

    filename: str
    image: Image.Image
    file_format: str
    size_bytes: int

    @property
    def width(self) -> int:
        return self.image.width

    @property
    def height(self) -> int:
        return self.image.height


def validate_upload(filename: str, data: bytes) -> ValidatedImage:
    """Validate an uploaded file entirely in memory (nothing is written to disk).

    The real content is inspected, not just the extension, so a renamed or
    truncated file is rejected with a clear message.
    """
    if not data:
        raise ImageValidationError("The uploaded file is empty.")

    extension = Path(filename).suffix.lower().lstrip(".")
    if extension not in config.UPLOAD_EXTENSIONS:
        raise ImageValidationError(
            f"Unsupported file type '.{extension}'. Supported types: JPG, JPEG, PNG, WEBP."
        )
    if len(data) > config.MAX_UPLOAD_MB * 1024 * 1024:
        raise ImageValidationError(f"File is larger than the {config.MAX_UPLOAD_MB} MB limit.")

    try:
        with Image.open(io.BytesIO(data)) as probe:
            probe.verify()  # cheap integrity check; invalidates the handle
        image = Image.open(io.BytesIO(data))
        file_format = image.format or ""
        image.load()  # force a full decode so truncated files fail here
    except (UnidentifiedImageError, OSError, SyntaxError, ValueError, Image.DecompressionBombError) as exc:
        raise ImageValidationError(
            "The file could not be read as an image. It may be corrupted or not a real image."
        ) from exc

    if file_format not in config.ALLOWED_FORMATS:
        raise ImageValidationError(
            f"File content is '{file_format or 'unknown'}', which is not a supported image format."
        )

    image = ImageOps.exif_transpose(image).convert("RGB")
    if min(image.size) < config.MIN_IMAGE_SIDE:
        raise ImageValidationError(
            f"Image is too small ({image.width}x{image.height}). Minimum side is {config.MIN_IMAGE_SIDE}px."
        )
    return ValidatedImage(filename=filename, image=image, file_format=file_format, size_bytes=len(data))


# --------------------------------------------------------------------------- #
# Face detection (OpenCV Haar cascade - lightweight, ships with opencv)
# --------------------------------------------------------------------------- #
@dataclass
class FaceDetection:
    boxes: list[Box]
    detector_available: bool = True

    @property
    def count(self) -> int:
        return len(self.boxes)

    @property
    def summary(self) -> str:
        if not self.detector_available:
            return "Face detector unavailable"
        if self.count == 0:
            return "No face detected"
        if self.count == 1:
            return "1 face detected (analyzed)"
        return f"{self.count} faces detected (primary analyzed)"


@lru_cache(maxsize=1)
def _load_cascade() -> cv2.CascadeClassifier | None:
    if not hasattr(cv2, "CascadeClassifier") or not hasattr(cv2, "data") or not hasattr(cv2.data, "haarcascades"):
        return None
    path = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
    if not path.is_file():
        return None
    try:
        cascade = cv2.CascadeClassifier(str(path))
        return None if cascade.empty() else cascade
    except Exception:
        return None


def detect_faces(image: Image.Image) -> FaceDetection:
    """Detect frontal faces. Haar cascades miss profile/occluded faces - see README."""
    cascade = _load_cascade()
    if cascade is None:
        return FaceDetection(boxes=[], detector_available=False)

    gray = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2GRAY)
    scale = min(1.0, config.FACE_DETECT_MAX_SIDE / max(gray.shape))
    if scale < 1.0:
        gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

    min_side = max(30, int(0.05 * min(gray.shape)))
    found = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(min_side, min_side))
    boxes = [(int(x / scale), int(y / scale), int(w / scale), int(h / scale)) for x, y, w, h in found]
    boxes.sort(key=lambda b: b[2] * b[3], reverse=True)  # largest face first
    return FaceDetection(boxes=boxes)


def crop_face(image: Image.Image, box: Box, margin: float = config.FACE_CROP_MARGIN) -> Image.Image:
    """Crop a face with extra context around it, clamped to the image bounds."""
    x, y, w, h = box
    pad_x, pad_y = int(w * margin), int(h * margin)
    left, top = max(0, x - pad_x), max(0, y - pad_y)
    right, bottom = min(image.width, x + w + pad_x), min(image.height, y + h + pad_y)
    return image.crop((left, top, right, bottom))


def draw_face_boxes(image: Image.Image, boxes: list[Box]) -> Image.Image:
    """Return a copy of the image with rectangles around detected faces."""
    canvas = np.array(image)
    thickness = max(2, max(image.size) // 300)
    for i, (x, y, w, h) in enumerate(boxes):
        color = (0, 229, 255) if i == 0 else (148, 163, 184)
        cv2.rectangle(canvas, (x, y), (x + w, y + h), color, thickness)
        if len(boxes) > 1:
            label = "PRIMARY" if i == 0 else f"FACE #{i+1}"
            cv2.putText(canvas, label, (x, max(15, y - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
    return Image.fromarray(canvas)


# --------------------------------------------------------------------------- #
# Model input
# --------------------------------------------------------------------------- #
def prepare_model_input(image: Image.Image) -> np.ndarray:
    """Resize to the network size and return a (1, H, W, 3) float32 array in 0-255.

    Pixel scaling to [-1, 1] happens *inside* the model (the "preprocess"
    layer), so training and inference can never disagree about normalisation.
    """
    resized = image.resize(config.IMG_SIZE, Image.BILINEAR)
    return np.asarray(resized, dtype=np.float32)[np.newaxis, ...]
