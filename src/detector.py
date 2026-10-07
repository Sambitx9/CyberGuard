"""Image deepfake detector. Loads the trained Keras model and runs the pipeline.

Pipeline: validated image -> face detection -> model inference -> prediction,
confidence, risk -> visual explanation.

``BaseDetector`` is the extension point: future video / audio / lip-sync
detectors can implement the same interface and return the same result type.
"""
from __future__ import annotations

import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from time import perf_counter

import numpy as np
from PIL import Image

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")  # quieter TensorFlow logs

import config  # noqa: E402
from src import utils  # noqa: E402
from src.explainability import ExplanationVisual, explain  # noqa: E402
from src.preprocessing import (  # noqa: E402
    ValidatedImage, crop_face, detect_faces, draw_face_boxes, prepare_model_input,
)


class ModelNotTrainedError(FileNotFoundError):
    """The trained model file does not exist yet."""


class ModelLoadError(RuntimeError):
    """The model file exists but could not be loaded."""


class InferenceError(RuntimeError):
    """The model produced an unusable output."""


@dataclass
class AnalysisResult:
    media_type: str
    filename: str
    timestamp: datetime
    prediction: str            # "REAL" | "FAKE" | "NO FACE DETECTED"
    confidence: float | None   # 0-100, certainty in `prediction` (None if no face)
    fake_probability: float | None # 0-100, raw model output P(FAKE) (None if no face)
    risk: str                  # LOW | MEDIUM | HIGH | N/A
    signal: str
    face_summary: str
    face_count: int
    processing_ms: float
    explanation: str
    visual: ExplanationVisual
    annotated_image: Image.Image
    warnings: list[str] = field(default_factory=list)


class BaseDetector(ABC):
    """Common interface for all media detectors (image today, more later)."""

    media_type: str = "unknown"

    @abstractmethod
    def analyze(self, media) -> AnalysisResult:  # pragma: no cover - interface
        ...


class ImageDetector(BaseDetector):
    media_type = "image"

    def __init__(self, model_path: Path = config.MODEL_PATH) -> None:
        self.model_path = model_path
        self._model = None

    @property
    def model_exists(self) -> bool:
        return self.model_path.is_file()

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    def load(self) -> None:
        """Load the trained model. Raises ModelNotTrainedError / ModelLoadError."""
        if not self.model_exists:
            raise ModelNotTrainedError(f"No trained model found at {self.model_path}")
        try:
            from tensorflow import keras
        except ImportError as exc:
            raise ModelLoadError("TensorFlow is not installed. Run: pip install -r requirements.txt") from exc
        try:
            model = keras.models.load_model(self.model_path, compile=False)
        except Exception as exc:
            raise ModelLoadError(f"Model file could not be loaded: {exc}") from exc
        if tuple(model.input_shape[1:3]) != tuple(config.IMG_SIZE):
            raise ModelLoadError(
                f"Model expects input {model.input_shape[1:3]} but config.IMG_SIZE is {config.IMG_SIZE}. "
                "Retrain the model or fix config.py."
            )
        self._model = model

    def analyze(self, media: ValidatedImage) -> AnalysisResult:
        if self._model is None:
            raise ModelNotTrainedError("Model is not loaded.")

        started = perf_counter()
        warnings: list[str] = []
        image = media.image

        # 1) Face detection
        faces = detect_faces(image)
        annotated = draw_face_boxes(image, faces.boxes)
        if not faces.detector_available:
            warnings.append("The OpenCV face detector is unavailable.")

        # IF NO FACE: do NOT call deepfake classifier or generate misleading Grad-CAM
        if faces.count == 0:
            explanation = (
                "No detectable human face was found in this image. "
                "CyberGuard's current model is designed for face deepfake detection, "
                "so a REAL/FAKE classification was not performed."
            )
            visual = ExplanationVisual(
                image=annotated,
                method="NOT RUN",
                title="Face detection scan (0 faces found)",
                note="No human face was detected. Deepfake model inference and Grad-CAM were not performed.",
                is_gradcam=False,
            )
            return AnalysisResult(
                media_type=self.media_type,
                filename=media.filename,
                timestamp=datetime.now(),
                prediction="NO FACE DETECTED",
                confidence=None,
                fake_probability=None,
                risk="N/A",
                signal="N/A (No face detected)",
                face_summary=faces.summary,
                face_count=0,
                processing_ms=(perf_counter() - started) * 1000,
                explanation=explanation,
                visual=visual,
                annotated_image=annotated,
                warnings=warnings,
            )

        # 2) Face detected -> Deterministic Multi-Face Policy
        if faces.count > 1:
            primary_box = faces.boxes[0]
            warnings.append(
                f"Multiple faces detected ({faces.count}). "
                f"Analyzing primary face (largest: {primary_box[2]}x{primary_box[3]} px at x={primary_box[0]}, y={primary_box[1]})."
            )
            model_image = crop_face(image, primary_box)
        elif config.USE_FACE_CROP:
            model_image = crop_face(image, faces.boxes[0])
        else:
            model_image = image

        batch = prepare_model_input(model_image)
        raw = np.asarray(self._model(batch, training=False)).reshape(-1)
        p_fake = float(raw[0]) if raw.size else float("nan")
        if not np.isfinite(p_fake) or not 0.0 <= p_fake <= 1.0:
            raise InferenceError("The model returned an invalid probability.")

        prediction = "FAKE" if p_fake >= config.FAKE_THRESHOLD else "REAL"
        confidence = max(p_fake, 1.0 - p_fake) * 100
        fake_probability = p_fake * 100
        risk = utils.risk_level(fake_probability)

        # 3) Explanation (Dynamic Grad-CAM for the predicted class, or labelled fallback)
        visual = explain(self._model, batch, model_image, prediction=prediction)

        return AnalysisResult(
            media_type=self.media_type,
            filename=media.filename,
            timestamp=datetime.now(),
            prediction=prediction,
            confidence=confidence,
            fake_probability=fake_probability,
            risk=risk,
            signal=utils.manipulation_signal(risk),
            face_summary=faces.summary,
            face_count=faces.count,
            processing_ms=(perf_counter() - started) * 1000,
            explanation=utils.build_explanation(prediction, confidence, fake_probability, risk,
                                                faces.count, visual.method),
            visual=visual,
            annotated_image=annotated,
            warnings=warnings,
        )
