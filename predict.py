"""Command-line prediction:  python predict.py path/to/image.jpg [--heatmap out.png] [--report out.html]"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import config
from src import utils
from src.detector import ImageDetector, ModelLoadError, ModelNotTrainedError
from src.preprocessing import ImageValidationError, validate_upload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run CyberGuard on a single image.")
    parser.add_argument("image", type=Path)
    parser.add_argument("--heatmap", type=Path, help="save the explanation image here (PNG)")
    parser.add_argument("--report", type=Path, help="save an HTML report here")
    args = parser.parse_args(argv)

    detector = ImageDetector()
    if not detector.model_exists:
        print("MODEL NOT TRAINED\n"
              "Add images to dataset/real and dataset/fake, then run:  python train_model.py", file=sys.stderr)
        return 2

    try:
        validated = validate_upload(args.image.name, args.image.read_bytes())
    except OSError as exc:
        print(f"Cannot read file: {exc}", file=sys.stderr)
        return 1
    except ImageValidationError as exc:
        print(f"Invalid image: {exc}", file=sys.stderr)
        return 1

    try:
        detector.load()
        result = detector.analyze(validated)
    except (ModelNotTrainedError, ModelLoadError) as exc:
        print(f"Model error: {exc}", file=sys.stderr)
        return 2

    print("CYBERGUARD ANALYSIS\n")
    if result.prediction == "NO FACE DETECTED":
        print("Status:          UNSUPPORTED — NO FACE DETECTED")
        print(f"Prediction:      {result.prediction}")
        print("Confidence:      N/A")
        print(f"Risk:            {result.risk}")
        print("Model:           NOT RUN")
        print("Grad-CAM:        NOT RUN")
        print(f"Face detection:  {result.face_summary}")
        print(f"Signal:          {result.signal}")
        print(f"Processing time: {result.processing_ms:.0f} ms")
        print(f"Explanation:     {result.explanation}")
    else:
        print(f"Prediction:      {result.prediction}")
        print(f"Confidence:      {result.confidence:.1f}%   (model certainty, not proof)")
        print(f"Risk:            {result.risk}")
        print(f"Face detection:  {result.face_summary}")
        print(f"Signal:          {result.signal}")
        print(f"Processing time: {result.processing_ms:.0f} ms")
        print(f"Explanation:     {result.visual.title}")
    for warning in result.warnings:
        print(f"[warn] {warning}")

    if args.heatmap:
        result.visual.image.save(args.heatmap)
        print(f"Saved explanation image: {args.heatmap}")
    if args.report:
        args.report.write_text(utils.build_report_html(result, utils.load_training_metadata()), encoding="utf-8")
        print(f"Saved report: {args.report}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
