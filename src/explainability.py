"""Visual explanations: real Grad-CAM, with a clearly labelled ELA fallback."""
from __future__ import annotations

import io
from dataclasses import dataclass

import cv2
import numpy as np
from PIL import Image, ImageChops

import config


class ExplanationError(RuntimeError):
    """Raised when Grad-CAM cannot be computed reliably."""


@dataclass
class ExplanationVisual:
    image: Image.Image
    method: str        # "Grad-CAM" or "ELA"
    title: str         # caption shown in the UI / report
    note: str          # what the visual does and does not mean
    is_gradcam: bool


def compute_gradcam(model, batch: np.ndarray, prediction: str = "FAKE") -> np.ndarray:
    """Grad-CAM for the predicted class logit. Returns a (h, w) map scaled to 0-1.

    When prediction is REAL, we compute gradients with respect to the REAL logit
    (-fake_logit), highlighting regions that most influenced the REAL score.
    When prediction is FAKE, we compute gradients with respect to the FAKE logit,
    highlighting regions that most influenced the FAKE score.
    """
    import tensorflow as tf

    try:
        preprocess = model.get_layer(config.LAYER_PREPROCESS)
        backbone = model.get_layer(config.LAYER_BACKBONE)
        head = [model.get_layer(name) for name in config.CAM_HEAD_LAYERS]
    except ValueError as exc:  # layer missing -> model was not built by train_model.py
        raise ExplanationError(f"Model layout not compatible with Grad-CAM: {exc}") from exc

    x = preprocess(tf.convert_to_tensor(batch))
    with tf.GradientTape() as tape:
        features = backbone(x, training=False)  # (1, 7, 7, 1280)
        tape.watch(features)
        out = features
        for layer in head:
            out = layer(out, training=False)
        fake_logit = out[:, 0]
        # Target logit corresponding to the predicted class
        target_logit = -fake_logit if prediction == "REAL" else fake_logit

    grads = tape.gradient(target_logit, features)
    if grads is None:
        raise ExplanationError("No gradient could be computed.")

    channel_weights = tf.reduce_mean(grads, axis=(1, 2))                       # (1, C)
    cam = tf.reduce_sum(features * channel_weights[:, None, None, :], axis=-1)[0]
    cam = tf.nn.relu(cam).numpy()

    peak = float(cam.max())
    if not np.isfinite(peak) or peak <= 1e-8:
        raise ExplanationError(f"Heatmap is empty (no region increased the {prediction} prediction).")
    return cam / peak


def overlay_heatmap(image: Image.Image, heatmap: np.ndarray, alpha: float = 0.45, max_side: int = 640) -> Image.Image:
    """Blend a 0-1 heatmap (JET colour map) over the image."""
    base = image.copy()
    base.thumbnail((max_side, max_side))
    arr = np.asarray(base.convert("RGB"))
    resized = cv2.resize(heatmap.astype(np.float32), (arr.shape[1], arr.shape[0]), interpolation=cv2.INTER_CUBIC)
    colored = cv2.applyColorMap(np.uint8(255 * np.clip(resized, 0, 1)), cv2.COLORMAP_JET)
    colored = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)
    blended = cv2.addWeighted(arr, 1 - alpha, colored, alpha, 0)
    return Image.fromarray(blended)


def error_level_analysis(image: Image.Image, quality: int = 90, max_side: int = 640) -> Image.Image:
    """Error Level Analysis: amplified difference after a JPEG re-save.

    This is a classic forensic heuristic and is NOT derived from the model.
    """
    base = image.convert("RGB")
    base.thumbnail((max_side, max_side))
    buffer = io.BytesIO()
    base.save(buffer, format="JPEG", quality=quality)
    buffer.seek(0)
    recompressed = Image.open(buffer).convert("RGB")
    diff = ImageChops.difference(base, recompressed)
    peak = max(channel_max for _, channel_max in diff.getextrema()) or 1
    return diff.point(lambda value: min(255, int(value * 255.0 / peak)))


def explain(model, batch: np.ndarray, image: Image.Image, prediction: str = "FAKE") -> ExplanationVisual:
    """Return Grad-CAM for the predicted class if it works, otherwise an honestly labelled ELA image."""
    try:
        heatmap = compute_gradcam(model, batch, prediction=prediction)
        return ExplanationVisual(
            image=overlay_heatmap(image, heatmap),
            method="Grad-CAM",
            title=f"Grad-CAM heatmap ({prediction})",
            note=(
                f"Warm colours (red/yellow) mark regions that most influenced the model's {prediction} prediction. "
                "This shows where the model looked, not proof that those regions were edited."
            ),
            is_gradcam=True,
        )
    except Exception as exc:  # fall back rather than fail the whole analysis
        return ExplanationVisual(
            image=error_level_analysis(image),
            method="ELA",
            title="Error Level Analysis (fallback - NOT Grad-CAM)",
            note=(
                f"Grad-CAM was unavailable ({exc}). This fallback is a JPEG re-compression "
                "heuristic that is independent of the AI model; it is not model attention."
            ),
            is_gradcam=False,
        )
