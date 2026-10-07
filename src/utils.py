"""Shared helpers: risk bands, text generation, formatting and HTML reports."""
from __future__ import annotations

import base64
import html
import io
import json
from typing import TYPE_CHECKING, Any

from PIL import Image

import config

if TYPE_CHECKING:  # avoid a runtime import cycle with detector.py
    from src.detector import AnalysisResult

RISK_COLORS = {"LOW": "#22c55e", "MEDIUM": "#f59e0b", "HIGH": "#ef4444", "N/A": "#94a3b8"}


def risk_level(risk_score: float) -> str:
    """Map a 0-100 FAKE-probability score to LOW / MEDIUM / HIGH."""
    if risk_score >= config.RISK_HIGH_START:
        return "HIGH"
    if risk_score >= config.RISK_MEDIUM_START:
        return "MEDIUM"
    return "LOW"


def manipulation_signal(risk: str) -> str:
    return {
        "LOW": "No significant manipulation signal",
        "MEDIUM": "Weak / inconclusive manipulation signal",
        "HIGH": "Strong manipulation signal",
        "N/A": "N/A (No face detected)",
    }.get(risk, "No data")


def build_explanation(prediction: str, confidence: float | None, fake_probability: float | None,
                      risk: str, faces: int, method: str) -> str:
    """Plain-language summary of the result. Describes the model, never certainty."""
    if prediction == "NO FACE DETECTED":
        return (
            "No detectable human face was found in this image. "
            "CyberGuard's current model is designed for face deepfake detection, "
            "so a REAL/FAKE classification was not performed."
        )

    parts = [
        f"The model classified this image as {prediction} with {confidence:.1f}% confidence "
        f"(estimated FAKE probability {fake_probability:.1f}%, risk {risk})."
    ]
    if faces == 0:
        parts.append("No face was found, so the whole image was analysed; results are less reliable for non-face images.")
    elif faces > 1:
        parts.append(f"Multiple faces were found ({faces}); the primary face was selected and analyzed.")
    if method == "Grad-CAM":
        parts.append(f"The Grad-CAM heatmap highlights the regions that most influenced the model's {prediction} prediction.")
    else:
        parts.append("Grad-CAM was unavailable, so an independent Error Level Analysis image is shown instead.")
    parts.append("Confidence reflects model certainty, not proof that an image is authentic or manipulated.")
    return " ".join(parts)


def format_size(num_bytes: int) -> str:
    size = float(num_bytes)
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024 or unit == "GB":
            return f"{size:.0f} {unit}" if unit == "B" else f"{size:.1f} {unit}"
        size /= 1024
    return f"{num_bytes} B"


def load_training_metadata() -> dict[str, Any] | None:
    """Metrics written by train_model.py. Returns None if absent or unreadable."""
    try:
        return json.loads(config.METADATA_PATH.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def image_to_data_uri(image: Image.Image) -> str:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


def safe_stem(filename: str) -> str:
    stem = filename.rsplit(".", 1)[0]
    cleaned = "".join(c if c.isalnum() or c in "-_" else "_" for c in stem)
    return cleaned[:40] or "image"


def build_report_html(result: "AnalysisResult", metadata: dict[str, Any] | None) -> str:
    """Self-contained HTML report (open in a browser; print to PDF if needed)."""
    e = html.escape
    color = RISK_COLORS.get(result.risk, "#94a3b8")
    is_no_face = (result.prediction == "NO FACE DETECTED")

    if is_no_face:
        status_line = "STATUS: <strong>UNSUPPORTED &mdash; NO FACE DETECTED</strong>"
        pred_val = "NO FACE DETECTED"
        conf_val = "N/A"
        prob_val = "N/A"
        risk_val = "N/A"
        model_status = "NOT RUN (no human face detected for face-deepfake model)"
        gradcam_status = "NOT RUN"
    else:
        status_line = f"Status: Model prediction: <strong>{e(result.prediction)}</strong> (Risk: <span class=\"risk\">{e(result.risk)}</span>)"
        pred_val = e(result.prediction)
        conf_val = f"{result.confidence:.1f}%" if result.confidence is not None else "N/A"
        prob_val = f"{result.fake_probability:.1f}%" if result.fake_probability is not None else "N/A"
        risk_val = e(result.risk)
        model_status = f"Trained model loaded ({config.MODEL_PATH.name})"
        if metadata and "validation_accuracy" in metadata:
            model_status += (f"; validation accuracy on its own held-out split: "
                             f"{metadata['validation_accuracy'] * 100:.1f}% "
                             f"({metadata.get('validation_images', '?')} images)")
        gradcam_status = e(result.visual.method)

    visual_html = (
        f'<h3>{e(result.visual.title)}</h3>'
        f'<img src="{image_to_data_uri(result.visual.image)}" alt="visualisation">'
        f'<p class="small">{e(result.visual.note)}</p>'
    )
    warnings_html = "".join(f"<li>{e(w)}</li>" for w in result.warnings)
    warnings_block = f"<h3>Warnings</h3><ul>{warnings_html}</ul>" if warnings_html else ""
    rows = [
        ("File name", result.filename),
        ("Timestamp", result.timestamp.strftime("%Y-%m-%d %H:%M:%S")),
        ("Prediction", pred_val),
        ("Confidence (model)", conf_val),
        ("Estimated FAKE probability", prob_val),
        ("Risk", risk_val),
        ("Manipulation signal", result.signal),
        ("Model status", model_status),
        ("Grad-CAM status", gradcam_status),
        ("Face detection", result.face_summary),
        ("Processing time", f"{result.processing_ms:.0f} ms"),
    ]
    table = "".join(f"<tr><th>{e(k)}</th><td>{v}</td></tr>" for k, v in rows)
    return f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8"><title>CyberGuard Report - {e(result.filename)}</title>
<style>
body{{font-family:Segoe UI,Arial,sans-serif;background:#0a0e17;color:#e5e7eb;max-width:820px;margin:32px auto;padding:0 20px;line-height:1.55}}
h1{{color:#00e5ff;margin-bottom:0}} .tag{{color:#94a3b8;margin-top:4px}}
table{{width:100%;border-collapse:collapse;margin:18px 0}} th,td{{text-align:left;padding:9px 12px;border-bottom:1px solid #1f2937}}
th{{width:34%;color:#94a3b8;font-weight:600}} .risk{{color:{color};font-weight:700}}
img{{max-width:100%;border-radius:8px;border:1px solid #1f2937}} .small{{color:#94a3b8;font-size:.9em}}
.box{{background:#111827;border-left:3px solid #00e5ff;padding:12px 16px;border-radius:4px}}
@media print{{body{{background:#fff;color:#111}}h1{{color:#0369a1}}.box{{background:#f1f5f9}}}}
</style></head><body>
<h1>CYBERGUARD</h1><div class="tag">{e(config.TAGLINE)} &mdash; Image Analysis Report</div>
<table>{table}</table>
<p>{status_line}</p>
<h3>Explanation</h3><div class="box">{e(result.explanation)}</div>
<p class="small">Confidence reflects model certainty, not proof that an image is authentic or manipulated.</p>
{visual_html}{warnings_block}
<hr><p class="small"><strong>Disclaimer:</strong> {e(config.DISCLAIMER)}</p>
</body></html>"""
