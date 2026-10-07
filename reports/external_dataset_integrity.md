# External Dataset Integrity Audit Report

**Audit Date:** 2026-10-07 08:02:11  
**Dataset Source:** Celeb-DF v2 Benchmark (`thenewsupercell/celeb-df-image-dataset`)  
**Source Split:** `test` (strictly independent test partition)  
**Total Images:** 200  
**Class Distribution:** 100 REAL, 100 FAKE (50.0% / 50.0% perfectly balanced)  

## Integrity & Verification Checklist
1. **Machine-Readable Labels:** Verified against official binary ground truth (`0 = Fake, 1 = Real`).
2. **Corrupted File Filter:** 100% of images verified decodable by PIL; minimum side >= 32px.
3. **Cryptographic Leakage Audit:**
   - Scanned against 601 internal training, validation, and test images.
   - **Exact SHA-256 Collisions:** 0 (0.00% leakage).
4. **Generalization Value:** Celeb-DF v2 represents an independent, real-world generative distribution created by advanced face-swapping algorithms on celebrity video sequences.

## Storage Paths
- Real images: `external_test_data/real/` (100 files)
- Fake images: `external_test_data/fake/` (100 files)
