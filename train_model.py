"""Train the CyberGuard REAL/FAKE image classifier (MobileNetV2 transfer learning).

Usage:
    python train_model.py
    python train_model.py --epochs-head 5 --epochs-finetune 5
    python train_model.py --no-pretrained     # offline; expect much weaker results

Expected data:  dataset/real/*  and  dataset/fake/*   (jpg, jpeg, png, bmp)
Outputs:        model/cyberguard_model.keras, model/training_metadata.json,
                reports/training_history.png, reports/confusion_matrix.png
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import matplotlib

matplotlib.use("Agg")  # no display needed
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, UnidentifiedImageError
from sklearn.metrics import (ConfusionMatrixDisplay, classification_report,
                             confusion_matrix, roc_auc_score)
from sklearn.utils.class_weight import compute_class_weight

import config


class DatasetError(Exception):
    """Dataset is missing, too small or contains unreadable files."""


# --------------------------------------------------------------------------- #
# Dataset checks
# --------------------------------------------------------------------------- #
def scan_dataset() -> dict[str, int]:
    """Validate dataset/real and dataset/fake; return image counts per class."""
    counts: dict[str, int] = {}
    corrupted: list[Path] = []
    skipped: Counter[str] = Counter()

    for name in config.CLASS_NAMES:
        folder = config.DATASET_DIR / name
        if not folder.is_dir():
            raise DatasetError(f"Missing folder: {folder}\nCreate it and add images (see README, 'Dataset setup').")
        valid = 0
        for path in folder.rglob("*"):
            if not path.is_file() or path.name.startswith("."):
                continue
            if path.suffix.lower() not in config.TRAIN_EXTENSIONS:
                skipped[path.suffix.lower() or "(no extension)"] += 1
                continue
            try:
                with Image.open(path) as img:
                    img.verify()
                valid += 1
            except (UnidentifiedImageError, OSError, SyntaxError, ValueError):
                corrupted.append(path)
        counts[name] = valid

    if corrupted:
        listing = "\n".join(f"  - {p}" for p in corrupted[:10])
        extra = f"\n  ... and {len(corrupted) - 10} more" if len(corrupted) > 10 else ""
        raise DatasetError(f"{len(corrupted)} corrupted/unreadable image(s) found. Delete them and retry:\n{listing}{extra}")

    for name, count in counts.items():
        if count < config.MIN_IMAGES_PER_CLASS:
            raise DatasetError(
                f"dataset/{name}/ has {count} usable image(s); at least {config.MIN_IMAGES_PER_CLASS} are required "
                f"(hundreds to thousands recommended). Supported: {', '.join(config.TRAIN_EXTENSIONS)}"
            )

    if skipped:
        print(f"[warn] Ignoring unsupported files: {dict(skipped)} (training reads {', '.join(config.TRAIN_EXTENSIONS)})")
    low, high = min(counts.values()), max(counts.values())
    if low < config.RECOMMENDED_IMAGES_PER_CLASS:
        print(f"[warn] Only {low} images in the smaller class. Small datasets give unreliable models.")
    if high > 3 * low:
        print("[warn] Classes are imbalanced; class weights will be applied automatically.")
    return counts


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #
def build_model(pretrained: bool = True):
    """Frozen-backbone MobileNetV2 + small binary head. Output = P(FAKE)."""
    from tensorflow import keras
    L = keras.layers

    inputs = keras.Input(shape=(*config.IMG_SIZE, 3), name="image")  # raw pixels, 0-255

    # Augmentation layers are only active during training; they are no-ops at inference.
    augment = keras.Sequential(
        [L.RandomFlip("horizontal"), L.RandomRotation(0.05), L.RandomZoom(0.1), L.RandomContrast(0.1)],
        name="augmentation",
    )
    x = augment(inputs)
    # MobileNetV2 expects inputs in [-1, 1]. Doing it inside the model keeps training/inference consistent.
    x = L.Rescaling(1.0 / 127.5, offset=-1.0, name=config.LAYER_PREPROCESS)(x)

    base = keras.applications.MobileNetV2(
        input_shape=(*config.IMG_SIZE, 3), include_top=False,
        weights="imagenet" if pretrained else None,
    )
    # Wrap in a named Model so Grad-CAM can find the backbone by name.
    backbone = keras.Model(base.input, base.output, name=config.LAYER_BACKBONE)
    backbone.trainable = False  # phase 1: only the new head learns

    x = backbone(x, training=False)  # keep BatchNorm in inference mode
    x = L.GlobalAveragePooling2D(name="gap")(x)
    x = L.Dropout(0.3, name="dropout")(x)
    logit = L.Dense(1, name="logit")(x)                           # pre-sigmoid score (used by Grad-CAM)
    outputs = L.Activation("sigmoid", name="fake_probability")(logit)
    return keras.Model(inputs, outputs, name="cyberguard")


def compile_model(model, learning_rate: float) -> None:
    from tensorflow import keras
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="binary_crossentropy",
        metrics=["accuracy", keras.metrics.AUC(name="auc")],
    )


def unfreeze_top_layers(model, n_layers: int) -> None:
    """Fine-tuning: unfreeze the last n backbone layers (BatchNorm stays frozen)."""
    from tensorflow import keras
    backbone = model.get_layer(config.LAYER_BACKBONE)
    backbone.trainable = True
    for layer in backbone.layers[:-n_layers]:
        layer.trainable = False
    for layer in backbone.layers:
        if isinstance(layer, keras.layers.BatchNormalization):
            layer.trainable = False


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
def load_datasets():
    import tensorflow as tf
    from tensorflow import keras
    options = dict(
        directory=str(config.DATASET_DIR), labels="inferred", label_mode="binary",
        class_names=list(config.CLASS_NAMES),  # real -> 0, fake -> 1
        color_mode="rgb", image_size=config.IMG_SIZE, batch_size=config.BATCH_SIZE,
        seed=config.SEED, validation_split=config.VALIDATION_SPLIT,
    )
    # Same seed + shuffle for both calls => disjoint, reproducible train/validation split.
    train_ds = keras.utils.image_dataset_from_directory(subset="training", **options)
    val_ds = keras.utils.image_dataset_from_directory(subset="validation", **options)
    return train_ds.prefetch(tf.data.AUTOTUNE), val_ds.prefetch(tf.data.AUTOTUNE)


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #
def save_history_plot(history: dict[str, list[float]], finetune_start: int | None) -> Path:
    config.REPORTS_DIR.mkdir(exist_ok=True)
    fig, (ax_acc, ax_loss) = plt.subplots(1, 2, figsize=(12, 4.5))
    for ax, key, title in ((ax_acc, "accuracy", "Accuracy"), (ax_loss, "loss", "Loss (binary cross-entropy)")):
        ax.plot(history[key], label="train")
        ax.plot(history[f"val_{key}"], label="validation")
        if finetune_start:
            ax.axvline(finetune_start - 0.5, color="grey", linestyle="--", label="fine-tuning starts")
        ax.set_title(title)
        ax.set_xlabel("epoch")
        ax.grid(alpha=0.3)
        ax.legend()
    fig.tight_layout()
    path = config.REPORTS_DIR / "training_history.png"
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def evaluate(model, val_ds) -> dict:
    """Evaluate the saved best model on the validation split (single pass)."""
    y_true, y_prob = [], []
    for images, labels in val_ds:
        y_prob.append(np.asarray(model(images, training=False)).reshape(-1))
        y_true.append(np.asarray(labels).reshape(-1))
    y_true = np.concatenate(y_true).astype(int)
    y_prob = np.concatenate(y_prob)
    y_pred = (y_prob >= config.FAKE_THRESHOLD).astype(int)

    names = [n.upper() for n in config.CLASS_NAMES]
    report = classification_report(y_true, y_pred, target_names=names, output_dict=True, zero_division=0)
    print("\nValidation report (held-out split of YOUR dataset):")
    print(classification_report(y_true, y_pred, target_names=names, zero_division=0))

    matrix = confusion_matrix(y_true, y_pred, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(4.5, 4))
    ConfusionMatrixDisplay(matrix, display_labels=names).plot(ax=ax, colorbar=False)
    ax.set_title("Validation confusion matrix")
    fig.tight_layout()
    fig.savefig(config.REPORTS_DIR / "confusion_matrix.png", dpi=130)
    plt.close(fig)

    auc = float(roc_auc_score(y_true, y_prob)) if len(set(y_true)) == 2 else None
    return {
        "validation_images": int(len(y_true)),
        "validation_accuracy": float((y_pred == y_true).mean()),
        "validation_auc": auc,
        "classification_report": report,
        "confusion_matrix": matrix.tolist(),
    }


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the CyberGuard image classifier.")
    parser.add_argument("--epochs-head", type=int, default=config.HEAD_EPOCHS, help="epochs with frozen backbone")
    parser.add_argument("--epochs-finetune", type=int, default=config.FINE_TUNE_EPOCHS,
                        help="epochs fine-tuning top backbone layers (0 to skip)")
    parser.add_argument("--no-pretrained", action="store_true",
                        help="do not download ImageNet weights (offline use; much weaker model)")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        counts = scan_dataset()
    except DatasetError as exc:
        print(f"[dataset error] {exc}", file=sys.stderr)
        return 1
    print(f"Dataset OK: {counts}")

    try:
        from tensorflow import keras
    except ImportError:
        print("[error] TensorFlow is not installed. Run: pip install -r requirements.txt", file=sys.stderr)
        return 1

    keras.utils.set_random_seed(config.SEED)
    config.MODEL_DIR.mkdir(exist_ok=True)
    config.REPORTS_DIR.mkdir(exist_ok=True)

    train_ds, val_ds = load_datasets()

    # Counter-act class imbalance: rarer class gets a larger loss weight.
    labels = np.array([0] * counts["real"] + [1] * counts["fake"])
    weights = compute_class_weight("balanced", classes=np.array([0, 1]), y=labels)
    class_weight = {0: float(weights[0]), 1: float(weights[1])}

    try:
        model = build_model(pretrained=not args.no_pretrained)
    except Exception as exc:
        print(f"[error] Could not build the model: {exc}\n"
              "If this is a network error, ImageNet weights could not be downloaded. "
              "Connect to the internet once, or retry with --no-pretrained.", file=sys.stderr)
        return 1

    # One checkpoint object across both phases so the best val_loss overall is what stays on disk.
    checkpoint = keras.callbacks.ModelCheckpoint(str(config.MODEL_PATH), monitor="val_loss",
                                                 save_best_only=True, verbose=1)

    def early_stopping():
        return keras.callbacks.EarlyStopping(monitor="val_loss", patience=config.EARLY_STOP_PATIENCE,
                                             restore_best_weights=True, verbose=1)

    history: dict[str, list[float]] = {}

    def run_phase(epochs: int) -> int:
        result = model.fit(train_ds, validation_data=val_ds, epochs=epochs, class_weight=class_weight,
                           callbacks=[early_stopping(), checkpoint])
        for key, values in result.history.items():
            history.setdefault(key, []).extend(values)
        return len(result.history["loss"])

    print("\n=== Phase 1: training classification head (backbone frozen) ===")
    compile_model(model, config.HEAD_LR)
    head_epochs_run = run_phase(args.epochs_head)

    finetune_start = None
    if args.epochs_finetune > 0:
        print("\n=== Phase 2: fine-tuning top backbone layers ===")
        unfreeze_top_layers(model, config.FINE_TUNE_LAYERS)
        compile_model(model, config.FINE_TUNE_LR)  # recompile after changing trainable flags
        finetune_start = head_epochs_run
        run_phase(args.epochs_finetune)

    print(f"\nTraining graph saved to: {save_history_plot(history, finetune_start)}")

    # Evaluate the checkpoint that is actually saved (best val_loss), not the last epoch.
    best = keras.models.load_model(config.MODEL_PATH, compile=False)
    metrics = evaluate(best, val_ds)

    metadata = {
        "trained_at": datetime.now().isoformat(timespec="seconds"),
        "backbone": "MobileNetV2",
        "pretrained_imagenet": not args.no_pretrained,
        "input_size": list(config.IMG_SIZE),
        "classes": [n.upper() for n in config.CLASS_NAMES],
        "images_per_class": counts,
        "epochs_run": len(history["loss"]),
        "note": "Metrics are measured on a held-out split of the user's own dataset only.",
        **metrics,
    }
    config.METADATA_PATH.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(f"Model saved to: {config.MODEL_PATH}")
    print(f"Validation accuracy: {metrics['validation_accuracy'] * 100:.1f}% on {metrics['validation_images']} images "
          "(from YOUR dataset; this does not predict real-world performance).")
    print("\nNext step: streamlit run app.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
