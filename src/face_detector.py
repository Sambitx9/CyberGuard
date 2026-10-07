"""Face detector abstraction layer for CyberGuard.

Provides a unified interface across face localization backends:
1. HaarCascadeFaceDetector: The existing production OpenCV Haar cascade.
2. EnsembleHaarFaceDetector: Multi-cascade ensemble tuned for close-ups.
3. OpenCVDNNFaceDetector: Deep learning face detector (OpenCV DNN / YuNet).
"""
from __future__ import annotations

import os
from abc import ABC, abstractmethod
from functools import lru_cache
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
from PIL import Image

import config

Box = Tuple[int, int, int, int]


class BaseFaceDetector(ABC):
    """Abstract interface for face detection."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable detector name."""
        ...

    @property
    @abstractmethod
    def is_available(self) -> bool:
        """Whether detector dependencies/weights are loaded."""
        ...

    @abstractmethod
    def detect_faces(self, image: Image.Image) -> List[Box]:
        """Detect faces in an image. Returns sorted list of boxes (x, y, w, h) descending by area."""
        ...


class HaarCascadeFaceDetector(BaseFaceDetector):
    """Current production OpenCV Haar cascade face detector."""

    def __init__(self, scale_factor: float = 1.1, min_neighbors: int = 5, min_size: int = 30):
        self._scale_factor = scale_factor
        self._min_neighbors = min_neighbors
        self._min_size = min_size
        self._cascade = self._load_cascade()

    @property
    def name(self) -> str:
        return "OpenCV Haar Cascade (Default)"

    @property
    def is_available(self) -> bool:
        return self._cascade is not None

    def _load_cascade(self) -> cv2.CascadeClassifier | None:
        if not hasattr(cv2, "data") or not hasattr(cv2.data, "haarcascades"):
            return None
        p = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
        if not p.is_file():
            return None
        try:
            casc = cv2.CascadeClassifier(str(p))
            return None if casc.empty() else casc
        except Exception:
            return None

    def detect_faces(self, image: Image.Image) -> List[Box]:
        if not self.is_available:
            return []
        gray = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2GRAY)
        scale = min(1.0, config.FACE_DETECT_MAX_SIDE / max(gray.shape))
        if scale < 1.0:
            gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

        min_side = max(self._min_size, int(0.05 * min(gray.shape)))
        found = self._cascade.detectMultiScale(
            gray,
            scaleFactor=self._scale_factor,
            minNeighbors=self._min_neighbors,
            minSize=(min_side, min_side)
        )
        boxes = [(int(x / scale), int(y / scale), int(w / scale), int(h / scale)) for x, y, w, h in found]
        boxes.sort(key=lambda b: b[2] * b[3], reverse=True)
        return boxes


class EnsembleHaarFaceDetector(BaseFaceDetector):
    """Multi-scale ensemble combining default and alt2 Haar cascades for higher sensitivity on close-up crops."""

    def __init__(self):
        self._c1 = self._load_cascade("haarcascade_frontalface_default.xml")
        self._c2 = self._load_cascade("haarcascade_frontalface_alt2.xml")

    @property
    def name(self) -> str:
        return "Ensemble Multi-Scale Haar Cascade (Default + Alt2)"

    @property
    def is_available(self) -> bool:
        return (self._c1 is not None) and (self._c2 is not None)

    def _load_cascade(self, filename: str) -> cv2.CascadeClassifier | None:
        if not hasattr(cv2, "data") or not hasattr(cv2.data, "haarcascades"):
            return None
        p = Path(cv2.data.haarcascades) / filename
        if not p.is_file():
            return None
        try:
            casc = cv2.CascadeClassifier(str(p))
            return None if casc.empty() else casc
        except Exception:
            return None

    def detect_faces(self, image: Image.Image) -> List[Box]:
        if not self.is_available:
            return []
        gray = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2GRAY)
        scale = min(1.0, config.FACE_DETECT_MAX_SIDE / max(gray.shape))
        if scale < 1.0:
            gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

        f1 = self._c1.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=4, minSize=(30, 30))
        f2 = self._c2.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=4, minSize=(30, 30))

        # Union and non-maximum deduplication
        raw_boxes = []
        for x, y, w, h in list(f1) + list(f2):
            raw_boxes.append((int(x / scale), int(y / scale), int(w / scale), int(h / scale)))

        deduped: List[Box] = []
        for b in raw_boxes:
            bx, by, bw, bh = b
            overlap = False
            for d in deduped:
                dx, dy, dw, dh = d
                # Check intersection over union (IoU)
                ix1, iy1 = max(bx, dx), max(by, dy)
                ix2, iy2 = min(bx + bw, dx + dw), min(by + bh, dy + dh)
                if ix2 > ix1 and iy2 > iy1:
                    inter_area = (ix2 - ix1) * (iy2 - iy1)
                    union_area = (bw * bh) + (dw * dh) - inter_area
                    if (inter_area / union_area) > 0.4:
                        overlap = True
                        break
            if not overlap:
                deduped.append(b)

        deduped.sort(key=lambda b: b[2] * b[3], reverse=True)
        return deduped


class OpenCVDNNFaceDetector(BaseFaceDetector):
    """OpenCV DNN / YuNet Deep Learning Face Detector wrapper."""

    def __init__(self, weights_path: Path | None = None):
        self._weights_path = weights_path or (config.BASE_DIR / "model" / "face_detection_yunet_2023mar.onnx")
        self._detector = None
        if self._weights_path.is_file() and hasattr(cv2, "FaceDetectorYN"):
            try:
                self._detector = cv2.FaceDetectorYN.create(
                    model=str(self._weights_path),
                    config="",
                    input_size=(320, 320),
                    score_threshold=0.6,
                    nms_threshold=0.3,
                    top_k=5000
                )
            except Exception:
                self._detector = None

    @property
    def name(self) -> str:
        return "OpenCV YuNet Deep Learning Face Detector"

    @property
    def is_available(self) -> bool:
        return self._detector is not None

    def detect_faces(self, image: Image.Image) -> List[Box]:
        if not self.is_available:
            return []
        arr = np.asarray(image)
        h, w = arr.shape[:2]
        self._detector.setInputSize((w, h))
        bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
        _, faces = self._detector.detect(bgr)
        if faces is None:
            return []
        boxes = []
        for face in faces:
            fx, fy, fw, fh = face[:4].astype(int)
            boxes.append((int(max(0, fx)), int(max(0, fy)), int(fw), int(fh)))
        boxes.sort(key=lambda b: b[2] * b[3], reverse=True)
        return boxes


@lru_cache(maxsize=1)
def get_default_face_detector() -> BaseFaceDetector:
    """Return the active production face detector (Haar Cascade)."""
    return HaarCascadeFaceDetector()
