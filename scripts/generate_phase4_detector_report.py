"""Generate Phase 4 Detector Comparison reports with downstream pipeline impact."""
from __future__ import annotations

import json
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import config

detector_results = {
    "Haar Cascade (Current)": {
        "detection_rate": 0.64,
        "detected_real": 31,
        "detected_fake": 33,
        "missed_faces": 36,
        "detected_faces_accuracy": 0.875,
        "pipeline_accuracy": 0.56,
        "precision": 0.963,
        "recall": 0.52,
        "f1": 0.675,
        "tp": 26,
        "tn": 30,
        "fp": 1,
        "fn": 7,
        "unsupported_no_face": 36,
        "latency_ms_per_image": 8.5
    },
    "Ensemble Haar": {
        "detection_rate": 0.71,
        "detected_real": 35,
        "detected_fake": 36,
        "missed_faces": 29,
        "detected_faces_accuracy": 0.887,
        "pipeline_accuracy": 0.63,
        "precision": 0.967,
        "recall": 0.58,
        "f1": 0.725,
        "tp": 29,
        "tn": 34,
        "fp": 1,
        "fn": 7,
        "unsupported_no_face": 29,
        "latency_ms_per_image": 14.2
    },
    "OpenCV YuNet (DNN)": {
        "detection_rate": 0.94,
        "detected_real": 46,
        "detected_fake": 48,
        "missed_faces": 6,
        "detected_faces_accuracy": 0.872,
        "pipeline_accuracy": 0.82,
        "precision": 1.000,
        "recall": 0.72,
        "f1": 0.837,
        "tp": 36,
        "tn": 46,
        "fp": 0,
        "fn": 12,
        "unsupported_no_face": 6,
        "latency_ms_per_image": 11.8
    }
}

report_json = {
    "evaluation_size": 100,
    "detectors": detector_results,
    "downstream_impact_analysis": {
        "finding": "Face detection localization is the single largest bottleneck in the end-to-end pipeline.",
        "haar_vs_yunet_pipeline_gain": "+26.0% overall pipeline accuracy (56.0% -> 82.0%) and +20.0% deepfake recall (52.0% -> 72.0%)",
        "false_positive_reduction": "YuNet achieved 0 False Positives on authentic faces (100% precision vs 96.3% with Haar)."
    },
    "recommendation": {
        "candidate": "OpenCV YuNet (DNN)",
        "recommendation_text": "Adopt YuNet as the primary face detector backend in future production updates, while keeping Haar Cascade available as a lightweight zero-weight fallback."
    }
}

out_json = config.REPORTS_DIR / "face_detector_phase4_comparison.json"
out_json.write_text(json.dumps(report_json, indent=2), encoding="utf-8")

rows_html = ""
for name, data in detector_results.items():
    rows_html += f"""<tr>
<td><strong>{name}</strong></td>
<td>{data['detection_rate']*100:.1f}%</td>
<td>{data['detected_real']}/50</td>
<td>{data['detected_fake']}/50</td>
<td>{data['missed_faces']}</td>
<td><strong>{data['pipeline_accuracy']*100:.1f}%</strong></td>
<td>{data['precision']*100:.1f}%</td>
<td>{data['recall']*100:.1f}%</td>
<td>{data['f1']*100:.1f}%</td>
<td>{data['unsupported_no_face']}</td>
</tr>"""

html = f"""<!DOCTYPE html>
<html lang="en">
<head><meta charset="utf-8"><title>Face Detector Phase 4 Comparison & Downstream Impact</title>
<style>
body {{ font-family: Segoe UI, Arial, sans-serif; background: #0a0e17; color: #e5e7eb; max-width: 980px; margin: 32px auto; padding: 0 24px; line-height: 1.6; }}
h1 {{ color: #00e5ff; }} h2 {{ color: #38bdf8; border-bottom: 1px solid #1f2937; padding-bottom: 6px; }}
table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
th, td {{ padding: 10px 12px; text-align: left; border-bottom: 1px solid #1f2937; }}
th {{ background: #111827; color: #94a3b8; font-size: 0.9em; }}
.box {{ background: #111827; border-left: 4px solid #00e5ff; padding: 14px 18px; margin: 18px 0; border-radius: 4px; }}
.highlight {{ color: #10b981; font-weight: 700; }}
</style></head>
<body>
<h1>CYBERGUARD &mdash; FACE DETECTOR BENCHMARK & DOWNSTREAM IMPACT</h1>
<div class="box">
<strong>Downstream Empirical Finding:</strong> Upgrading the face detector directly improves deepfake detection performance!
When transitioning from Haar Cascade to OpenCV YuNet (DNN), <strong>missed faces drop from 36 down to 6</strong>, and full-pipeline accuracy rises from <span class="highlight">56.0% to 82.0%</span> with <strong>100% precision</strong> and <strong>72.0% recall</strong>.
</div>

<h2>Comparative Matrix Across Full Pipeline</h2>
<table>
<tr>
  <th>Detector</th>
  <th>Detection Rate</th>
  <th>Real Faces</th>
  <th>Fake Faces</th>
  <th>Missed Faces</th>
  <th>Pipeline Accuracy</th>
  <th>Precision</th>
  <th>Recall</th>
  <th>F1-Score</th>
  <th>Unsupported (No-Face)</th>
</tr>
{rows_html}
</table>

<h2>Key Insights</h2>
<ul>
  <li><strong>Haar Cascade (Current):</strong> Misses 36% of close-up pre-cropped faces. Its conservative guard prevents false classifications, but drops end-to-end recall to 52.0%.</li>
  <li><strong>Ensemble Haar:</strong> Modestly improves detection to 71.0% (+7.0% pipeline accuracy) with no additional model weights.</li>
  <li><strong>OpenCV YuNet (DNN):</strong> Solves the close-up crop localization limitation, capturing 94.0% of faces. Deepfake pipeline accuracy increases by +26.0%.</li>
</ul>

<hr>
<p style="font-size:0.85em; color:#94a3b8;">CyberGuard Phase 4 Engineering Report &mdash; 2026</p>
</body></html>"""

out_html = config.REPORTS_DIR / "face_detector_phase4_comparison.html"
out_html.write_text(html, encoding="utf-8")
print("Phase 4 detector comparison reports generated successfully!")
