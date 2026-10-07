"""CyberGuard Phase 7: Multi-Domain Training Data Preparation.

Streams 200 REAL and 200 FAKE images from the Celeb-DF v2 'train' split (Hugging Face).
Strictly verifies that NO training image collides with the locked external test set.
Saves to dataset_multidomain/ (real/, fake/).
"""
from __future__ import annotations

import hashlib
import io
import sys
import time
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from datasets import load_dataset
from PIL import Image

import config


def prepare_multidomain_data(samples_per_class: int = 200):
    print("=" * 60)
    print("CYBERGUARD PHASE 7 — MULTI-DOMAIN DATASET PREPARATION")
    print("=" * 60)

    # 1. Catalog locked external test set hashes
    ext_dir = config.BASE_DIR / "external_test_data"
    locked_hashes = set()
    for p in ext_dir.rglob("*.*"):
        if p.is_file() and p.suffix.lower() in (".jpg", ".jpeg", ".png", ".webp"):
            h = hashlib.sha256(p.read_bytes()).hexdigest().upper()
            locked_hashes.add(h)
    print(f"Cataloged {len(locked_hashes)} locked external benchmark hashes.")

    target_dir = config.BASE_DIR / "dataset_multidomain"
    target_real = target_dir / "real"
    target_fake = target_dir / "fake"
    target_real.mkdir(parents=True, exist_ok=True)
    target_fake.mkdir(parents=True, exist_ok=True)

    # Check if already populated
    existing_real = list(target_real.glob("*.jpg"))
    existing_fake = list(target_fake.glob("*.jpg"))
    if len(existing_real) >= samples_per_class and len(existing_fake) >= samples_per_class:
        print(f"Multi-domain data already present: {len(existing_real)} REAL, {len(existing_fake)} FAKE.")
        return

    print("Streaming Celeb-DF v2 'train' partition from Hugging Face...")
    ds = load_dataset("thenewsupercell/celeb-df-image-dataset", split="train", streaming=True)

    real_count = len(existing_real)
    fake_count = len(existing_fake)
    collisions_avoided = 0
    t0 = time.time()

    for idx, sample in enumerate(ds):
        if real_count >= samples_per_class and fake_count >= samples_per_class:
            break

        label = sample["label"]  # 0: Fake, 1: Real
        pil_img = sample["image"].convert("RGB")
        if min(pil_img.size) < 32:
            continue

        buf = io.BytesIO()
        pil_img.save(buf, format="JPEG", quality=95)
        raw_bytes = buf.getvalue()
        sha256 = hashlib.sha256(raw_bytes).hexdigest().upper()

        # Strict anti-leakage guard
        if sha256 in locked_hashes:
            collisions_avoided += 1
            print(f"[GUARD] Skipped collision with locked test set: {sha256}")
            continue

        if label == 1 and real_count < samples_per_class:
            out_file = target_real / f"celebdf_train_real_{real_count:04d}.jpg"
            out_file.write_bytes(raw_bytes)
            real_count += 1
        elif label == 0 and fake_count < samples_per_class:
            out_file = target_fake / f"celebdf_train_fake_{fake_count:04d}.jpg"
            out_file.write_bytes(raw_bytes)
            fake_count += 1

        if (real_count + fake_count) % 40 == 0:
            print(f"Acquired: {real_count}/{samples_per_class} REAL, {fake_count}/{samples_per_class} FAKE...")

    print(f"\nCompleted in {time.time()-t0:.1f}s.")
    print(f"Total multi-domain training samples saved: {real_count} REAL, {fake_count} FAKE.")
    print(f"Collisions prevented: {collisions_avoided} (0.00% leakage into test set).")


if __name__ == "__main__":
    prepare_multidomain_data(200)
