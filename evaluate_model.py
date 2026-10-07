"""CyberGuard Model Evaluation Runner.

Delegates directly to scripts.evaluate_model.run_evaluation().
"""
from __future__ import annotations

import sys
from pathlib import Path

# Add project root
BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from scripts.evaluate_model import run_evaluation

if __name__ == "__main__":
    run_evaluation()
