"""CyberGuard - Streamlit dashboard. Run with:  streamlit run app.py"""
from __future__ import annotations

import hashlib
import html
import importlib.metadata as metadata
import importlib.util
import platform
import sys
from datetime import datetime

import cv2
import streamlit as st

import config
from src import utils
from src.detector import AnalysisResult, ImageDetector, InferenceError, ModelLoadError
from src.preprocessing import ImageValidationError, validate_upload

esc = html.escape

st.set_page_config(page_title="CyberGuard", page_icon="🛡️", layout="wide")

CSS = """
<style>
:root{--cg-bg:#0a0e17;--cg-panel:#111827;--cg-line:#1f2937;--cg-cyan:#00e5ff;--cg-blue:#38bdf8;--cg-muted:#94a3b8}
.stApp{background:radial-gradient(1200px 500px at 85% -10%,#0d2236 0%,var(--cg-bg) 60%)}
section[data-testid="stSidebar"]{background:#0b1220;border-right:1px solid var(--cg-line)}
.cg-brand{display:flex;gap:12px;align-items:center;margin-bottom:2px}
.cg-brand b{font-size:1.35rem;letter-spacing:.14em;color:var(--cg-cyan)}
.cg-tag{color:var(--cg-muted);font-size:.85rem;margin:0 0 14px 46px}
.cg-hero{font-size:2rem;font-weight:700;margin:.2rem 0 .1rem}
.cg-hero span{color:var(--cg-cyan)}
.cg-sub{color:var(--cg-muted);margin-bottom:1.2rem}
.cg-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(190px,1fr));gap:14px;margin:8px 0 16px}
.cg-card{background:var(--cg-panel);border:1px solid var(--cg-line);border-radius:10px;padding:14px 16px}
.cg-label{color:var(--cg-muted);font-size:.78rem;letter-spacing:.06em}
.cg-value{font-size:1.5rem;font-weight:700;margin-top:4px;word-break:break-word}
.cg-value.sm{font-size:1.05rem}
.cg-note{color:var(--cg-muted);font-size:.8rem;margin-top:2px}
.cg-title{font-size:1.15rem;font-weight:700;letter-spacing:.12em;color:var(--cg-cyan);border-bottom:1px solid var(--cg-line);padding-bottom:8px;margin:18px 0 8px}
.cg-banner{border:1px solid #7f1d1d;background:#1a0d12;border-left:4px solid #ef4444;border-radius:8px;padding:14px 18px;margin:8px 0 16px}
.cg-banner h3{margin:0 0 6px;color:#fca5a5;letter-spacing:.08em}
.cg-banner code{background:#0a0e17;padding:2px 6px;border-radius:4px}
.cg-box{background:var(--cg-panel);border-left:3px solid var(--cg-cyan);border-radius:6px;padding:12px 16px;margin:6px 0 12px}
.cg-dot{display:inline-block;width:9px;height:9px;border-radius:50%;margin-right:7px;animation:cgpulse 2.4s ease-in-out infinite}
@keyframes cgpulse{50%{opacity:.35}}
@media (prefers-reduced-motion:reduce){.cg-dot{animation:none}}
div.stButton>button,div.stDownloadButton>button{border:1px solid var(--cg-cyan);color:var(--cg-cyan);background:transparent;font-weight:600;letter-spacing:.06em}
div.stButton>button:hover,div.stDownloadButton>button:hover{background:var(--cg-cyan);color:#06222b}
div.stButton>button:focus-visible{outline:2px solid var(--cg-blue)}
</style>
"""

# --------------------------------------------------------------------------- #
# Engine / state
# --------------------------------------------------------------------------- #
@st.cache_resource(show_spinner="Loading AI model...")
def load_detector(model_mtime: float) -> tuple[ImageDetector, str | None]:
    """Load the model once per model-file version. Returns (detector, error message)."""
    detector = ImageDetector()
    if not detector.model_exists:
        return detector, None
    try:
        detector.load()
    except ModelLoadError as exc:
        return detector, str(exc)
    return detector, None


def get_engine() -> tuple[ImageDetector, str | None]:
    mtime = config.MODEL_PATH.stat().st_mtime if config.MODEL_PATH.exists() else 0.0
    return load_detector(mtime)


def init_state() -> None:
    st.session_state.setdefault("history", [])
    st.session_state.setdefault("analyzed_count", 0)
    st.session_state.setdefault("result", None)       # (file_key, AnalysisResult)
    st.session_state.setdefault("report", None)       # (file_key, filename, html)


def tensorflow_installed() -> bool:
    return importlib.util.find_spec("tensorflow") is not None


def package_version(name: str) -> str:
    for pkg in (name, f"{name}-cpu", f"{name}-headless"):
        try:
            return metadata.version(pkg)
        except metadata.PackageNotFoundError:
            continue
    return "not installed"


# --------------------------------------------------------------------------- #
# UI helpers
# --------------------------------------------------------------------------- #
def card(label: str, value: str, note: str = "", color: str = "#e5e7eb", small: bool = False) -> str:
    size = " sm" if small else ""
    return (f'<div class="cg-card"><div class="cg-label">{esc(label)}</div>'
            f'<div class="cg-value{size}" style="color:{color}">{esc(value)}</div>'
            f'<div class="cg-note">{esc(note)}</div></div>')


def grid(cards: list[str]) -> None:
    st.markdown(f'<div class="cg-grid">{"".join(cards)}</div>', unsafe_allow_html=True)


def not_trained_banner() -> None:
    st.markdown(
        '<div class="cg-banner"><h3>MODEL NOT TRAINED</h3>'
        'CyberGuard will not guess. There is no trained model at <code>model/cyberguard_model.keras</code>, '
        'so no predictions can be made.<br><br>'
        '1. Put training images in <code>dataset/real/</code> and <code>dataset/fake/</code><br>'
        '2. Run <code>python train_model.py</code><br>'
        '3. Restart or refresh this app</div>',
        unsafe_allow_html=True,
    )


def status_summary() -> tuple[str, str, str, str]:
    """Return (engine_text, engine_color, model_text, model_color)."""
    detector, error = get_engine()
    engine_ok = tensorflow_installed()
    engine = ("ONLINE", "#22c55e") if engine_ok else ("OFFLINE", "#ef4444")
    if not detector.model_exists:
        model = ("NOT TRAINED", "#f59e0b")
    elif error:
        model = ("LOAD ERROR", "#ef4444")
    else:
        model = ("READY", "#22c55e")
    return engine[0], engine[1], model[0], model[1]


def history_table() -> None:
    history = st.session_state.history
    if not history:
        st.caption("No analyses yet this session. Upload an image on the Analyze Media page.")
        return
    st.dataframe(history, hide_index=True)
    if st.button("CLEAR HISTORY"):
        st.session_state.history = []
        st.rerun()


# --------------------------------------------------------------------------- #
# Pages
# --------------------------------------------------------------------------- #
def page_dashboard() -> None:
    st.markdown('<div class="cg-hero">Security <span>Dashboard</span></div>'
                f'<div class="cg-sub">{esc(config.APP_SUBTITLE)}</div>', unsafe_allow_html=True)
    engine, engine_c, model, model_c = status_summary()
    last = st.session_state.result[1] if st.session_state.result else None
    risk_text = last.risk if last else "NO DATA"
    grid([
        card("AI ENGINE", engine, "TensorFlow runtime", engine_c),
        card("MODEL STATUS", model, config.MODEL_PATH.name, model_c),
        card("FILES ANALYZED", str(st.session_state.analyzed_count), "this session", "#38bdf8"),
        card("CURRENT RISK", risk_text, last.filename if last else "analyze an image", utils.RISK_COLORS.get(risk_text, "#94a3b8")),
    ])
    if model == "NOT TRAINED":
        not_trained_banner()

    meta = utils.load_training_metadata()
    if meta and model == "READY":
        st.markdown('<div class="cg-title">TRAINING SUMMARY</div>', unsafe_allow_html=True)
        auc = meta.get("validation_auc")
        grid([
            card("VALIDATION ACCURACY", f"{meta['validation_accuracy'] * 100:.1f}%", f"{meta['validation_images']} held-out images"),
            card("VALIDATION AUC", f"{auc:.3f}" if auc is not None else "n/a", "held-out split"),
            card("TRAINED", meta["trained_at"].replace("T", " "), f"{meta['backbone']} transfer learning", small=True),
        ])
        st.caption("Measured on a held-out split of the dataset this model was trained with. "
                   "It does not guarantee accuracy on other images, generators or real-world content.")

    st.markdown('<div class="cg-title">RECENT ANALYSES</div>', unsafe_allow_html=True)
    history_table()


def render_result(result: AnalysisResult) -> None:
    if result.prediction == "NO FACE DETECTED":
        pred_color = "#94a3b8"
        conf_str = "N/A"
        risk_note = "Face model not run"
    elif result.prediction == "FAKE":
        pred_color = "#ef4444"
        conf_str = f"{result.confidence:.1f}%" if result.confidence is not None else "N/A"
        risk_note = f"FAKE probability {result.fake_probability:.1f}%" if result.fake_probability is not None else ""
    else:
        pred_color = "#22c55e"
        conf_str = f"{result.confidence:.1f}%" if result.confidence is not None else "N/A"
        risk_note = f"FAKE probability {result.fake_probability:.1f}%" if result.fake_probability is not None else ""

    st.markdown('<div class="cg-title">CYBERGUARD ANALYSIS</div>', unsafe_allow_html=True)
    grid([
        card("PREDICTION", result.prediction, "", pred_color),
        card("CONFIDENCE", conf_str, "model certainty" if result.confidence is not None else "N/A", "#38bdf8"),
        card("RISK", result.risk, risk_note, utils.RISK_COLORS.get(result.risk, "#94a3b8")),
    ])
    st.caption("Confidence reflects model certainty, not proof that an image is authentic or manipulated.")
    grid([
        card("FACE DETECTION", result.face_summary, small=True),
        card("MANIPULATION SIGNAL", result.signal, small=True),
        card("PROCESSING TIME", f"{result.processing_ms:.0f} ms", "local, on this machine", small=True),
    ])
    for warning in result.warnings:
        st.warning(warning)

    st.markdown(f'<div class="cg-box"><b>Explanation.</b> {esc(result.explanation)}</div>', unsafe_allow_html=True)
    left, right = st.columns(2)
    with left:
        st.image(result.annotated_image, caption="Face detection (cyan boxes)")
    with right:
        st.image(result.visual.image, caption=result.visual.title)
        st.caption(result.visual.note)


def page_analyze() -> None:
    st.markdown('<div class="cg-hero">Analyze <span>Media</span></div>'
                '<div class="cg-sub">Images are processed locally and are not stored or sent anywhere.</div>',
                unsafe_allow_html=True)
    detector, error = get_engine()
    ready = detector.is_loaded and error is None
    if not detector.model_exists:
        not_trained_banner()
    elif error:
        st.error(f"The model file exists but failed to load: {error}")

    uploaded = st.file_uploader("UPLOAD MEDIA", type=list(config.UPLOAD_EXTENSIONS),
                                help="JPG, JPEG, PNG or WEBP, up to %d MB" % config.MAX_UPLOAD_MB)
    if uploaded is None:
        st.info("Upload an image to begin. Video and audio analysis are not part of this MVP.")
        st.markdown('<div class="cg-title">SESSION HISTORY</div>', unsafe_allow_html=True)
        history_table()
        return

    data = uploaded.getvalue()
    try:
        media = validate_upload(uploaded.name, data)
    except ImageValidationError as exc:
        st.error(str(exc))
        return

    file_key = hashlib.sha256(data).hexdigest()
    grid([
        card("FILENAME", media.filename, small=True),
        card("TYPE", media.file_format, small=True),
        card("SIZE", utils.format_size(media.size_bytes), small=True),
        card("DIMENSIONS", f"{media.width} x {media.height} px", small=True),
    ])
    preview, action = st.columns([1, 1])
    with preview:
        st.image(media.image, caption="Uploaded image")
    with action:
        if st.button("ANALYZE MEDIA", disabled=not ready):
            try:
                with st.spinner("Analyzing..."):
                    result = detector.analyze(media)
            except InferenceError as exc:
                st.error(f"Analysis failed: {exc}")
            except Exception as exc:  # keep the UI alive on any unexpected failure
                st.error(f"Unexpected error during analysis: {exc}")
            else:
                st.session_state.result = (file_key, result)
                st.session_state.report = None
                st.session_state.analyzed_count += 1
                st.session_state.history.insert(0, {
                    "Time": result.timestamp.strftime("%H:%M:%S"),
                    "Filename": result.filename,
                    "Prediction": result.prediction,
                    "Confidence": f"{result.confidence:.1f}%" if result.confidence is not None else "N/A",
                    "Risk": result.risk,
                })

    stored = st.session_state.result
    if stored and stored[0] == file_key:
        result = stored[1]
        render_result(result)
        if st.button("GENERATE REPORT"):
            report_html = utils.build_report_html(result, utils.load_training_metadata())
            name = f"cyberguard_report_{utils.safe_stem(result.filename)}_{datetime.now():%Y%m%d_%H%M%S}.html"
            st.session_state.report = (file_key, name, report_html)
        report = st.session_state.report
        if report and report[0] == file_key:
            st.download_button("DOWNLOAD REPORT", data=report[2].encode("utf-8"), file_name=report[1],
                               mime="text/html")
            st.caption("The report is generated in memory and downloaded by your browser. It is not saved on the server.")

    st.markdown('<div class="cg-title">SESSION HISTORY</div>', unsafe_allow_html=True)
    history_table()


def page_about() -> None:
    st.markdown('<div class="cg-hero">About <span>CyberGuard</span></div>'
                f'<div class="cg-sub">{esc(config.TAGLINE)}</div>', unsafe_allow_html=True)
    st.markdown(
        """
CyberGuard is a hackathon MVP that estimates whether a face image is **REAL** or **FAKE** (AI-generated or manipulated).

**How it works**
1. The upload is validated and decoded in memory.
2. An OpenCV detector looks for faces.
3. A MobileNetV2 network, fine-tuned by you with `train_model.py`, outputs the probability that the image is fake.
4. The probability becomes a prediction, a confidence and a LOW / MEDIUM / HIGH risk level.
5. Grad-CAM highlights the regions that influenced the score.

**Risk levels** are based on the model's FAKE probability: 0-39 LOW, 40-69 MEDIUM, 70-100 HIGH.

**Limits you should know about**
- Quality depends entirely on the dataset you train with. Models often fail on generators and editing methods they have not seen.
- Confidence is the model's certainty, not proof of authenticity.
- Image only. Video, audio, lip-sync, C2PA and watermark checks are not implemented.
        """
    )
    st.warning(config.DISCLAIMER)


def page_system() -> None:
    st.markdown('<div class="cg-hero">System <span>Information</span></div>', unsafe_allow_html=True)
    detector, error = get_engine()
    model_path = config.MODEL_PATH
    if model_path.exists():
        stat = model_path.stat()
        model_file = f"{model_path.name} ({utils.format_size(stat.st_size)}, "
        model_file += f"modified {datetime.fromtimestamp(stat.st_mtime):%Y-%m-%d %H:%M})"
    else:
        model_file = "not found (model not trained)"
    rows = {
        "CyberGuard version": config.VERSION,
        "Python": sys.version.split()[0],
        "Platform": platform.platform(),
        "TensorFlow": package_version("tensorflow"),
        "Keras": package_version("keras"),
        "OpenCV": cv2.__version__,
        "Streamlit": package_version("streamlit"),
        "Model file": model_file,
        "Model loaded": "yes" if detector.is_loaded else ("failed: " + error if error else "no"),
        "Model input size": f"{config.IMG_SIZE[0]}x{config.IMG_SIZE[1]}",
        "Face detector": "OpenCV Haar cascade (frontal faces)",
        "Face crop before inference": "on" if config.USE_FACE_CROP else "off (whole image is analysed)",
        "Data handling": "Local processing; uploads are kept in memory only",
    }
    st.dataframe([{"Component": k, "Value": v} for k, v in rows.items()], hide_index=True)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> None:
    init_state()
    st.markdown(CSS, unsafe_allow_html=True)

    logo_path = config.ASSETS_DIR / "logo.svg"
    logo = logo_path.read_text(encoding="utf-8") if logo_path.exists() else "🛡️"
    with st.sidebar:
        st.markdown(f'<div class="cg-brand">{logo}<b>{config.APP_NAME}</b></div>'
                    f'<div class="cg-tag">{esc(config.TAGLINE)}</div>', unsafe_allow_html=True)
        page = st.radio("Navigation", ["Dashboard", "Analyze Media", "About", "System Information"],
                        label_visibility="collapsed")
        _, _, model, model_color = status_summary()
        st.markdown(f'<span class="cg-dot" style="background:{model_color}"></span>Model: {model}',
                    unsafe_allow_html=True)
        st.caption(f"v{config.VERSION}")

    {"Dashboard": page_dashboard, "Analyze Media": page_analyze,
     "About": page_about, "System Information": page_system}[page]()


main()
