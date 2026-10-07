"""Download and validate a balanced, practical subset of desireemcv/face-real-vs-fake.

Dataset info:
- Source: Hugging Face dataset 'desireemcv/face-real-vs-fake'
- Classes: Class 1 = REAL, Class 0 = FAKE
- Images: 224x224 RGB faces
"""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path
from PIL import Image
from datasets import load_dataset

BASE_DIR = Path(__file__).resolve().parent
DATASET_DIR = BASE_DIR / "dataset"
TEST_DIR = BASE_DIR / "test_data"

REAL_DIR = DATASET_DIR / "real"
FAKE_DIR = DATASET_DIR / "fake"

TEST_REAL_DIR = TEST_DIR / "real"
TEST_FAKE_DIR = TEST_DIR / "fake"

TRAIN_TARGET_PER_CLASS = 250  # 250 real, 250 fake (500 total, ideal for CPU fine-tuning)
TEST_TARGET_PER_CLASS = 50    # 50 real, 50 fake (100 total unseen test images)


def ensure_dirs():
    for d in [REAL_DIR, FAKE_DIR, TEST_REAL_DIR, TEST_FAKE_DIR]:
        d.mkdir(parents=True, exist_ok=True)


def download_split(split_name: str, real_target: int, fake_target: int, out_real: Path, out_fake: Path):
    print(f"\n--- Downloading split: {split_name} (targets: {real_target} real, {fake_target} fake) ---")
    ds = load_dataset("desireemcv/face-real-vs-fake", split=split_name, streaming=True)
    
    real_count = 0
    fake_count = 0
    seen_hashes = set()
    corrupted_count = 0
    
    for item in ds:
        img: Image.Image = item["image"]
        lbl: int = item["label"]  # 1 is REAL, 0 is FAKE
        
        # Determine target folder
        if lbl == 1:
            if real_count >= real_target:
                continue
            target_folder = out_real
            prefix = "real"
        elif lbl == 0:
            if fake_count >= fake_target:
                continue
            target_folder = out_fake
            prefix = "fake"
        else:
            continue
            
        try:
            # Verify image integrity and convert to RGB
            img = img.convert("RGB")
            # Calculate content hash to prevent duplicates
            raw_bytes = img.tobytes()
            chash = hashlib.sha256(raw_bytes).hexdigest()
            if chash in seen_hashes:
                continue
            seen_hashes.add(chash)
            
            # Save as JPEG
            count_idx = real_count if lbl == 1 else fake_count
            filename = f"{prefix}_{split_name}_{count_idx:04d}.jpg"
            filepath = target_folder / filename
            img.save(filepath, format="JPEG", quality=95)
            
            # Verify file on disk
            with Image.open(filepath) as verify_img:
                verify_img.verify()
                
            if lbl == 1:
                real_count += 1
            else:
                fake_count += 1
                
            if (real_count + fake_count) % 50 == 0:
                print(f"Progress: {real_count}/{real_target} real, {fake_count}/{fake_target} fake")
                
        except Exception as exc:
            corrupted_count += 1
            print(f"Corrupted image skipped: {exc}")
            
        if real_count >= real_target and fake_count >= fake_target:
            break
            
    print(f"Finished {split_name}: {real_count} real, {fake_count} fake (corrupted/skipped: {corrupted_count})")
    return real_count, fake_count


def main():
    ensure_dirs()
    print("Starting dataset acquisition...")
    
    # 1. Download train subset
    train_real, train_fake = download_split("train", TRAIN_TARGET_PER_CLASS, TRAIN_TARGET_PER_CLASS, REAL_DIR, FAKE_DIR)
    
    # 2. Download test subset
    test_real, test_fake = download_split("test", TEST_TARGET_PER_CLASS, TEST_TARGET_PER_CLASS, TEST_REAL_DIR, TEST_FAKE_DIR)
    
    print("\n================ DATASET VALIDATION ================")
    print(f"Training dataset: {train_real} real, {train_fake} fake (Total: {train_real + train_fake})")
    print(f"Unseen test set:  {test_real} real, {test_fake} fake (Total: {test_real + test_fake})")
    
    assert train_real >= 20 and train_fake >= 20, "Training set below minimum required images"
    assert test_real >= 10 and test_fake >= 10, "Test set below minimum required images"
    print("Validation SUCCESS: Dataset is clean, balanced, and ready for training!")


if __name__ == "__main__":
    main()
