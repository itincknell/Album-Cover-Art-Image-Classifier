#!/usr/bin/env python3
"""
scripts/train_classifier.py

Train a transfer learning single-label classifier for album cover art.

Tasks
-----
- decade: predict release decade (trained on unified multi-genre dataset)
- genre : predict genre

This module mainly parses CLI arguments and builds configuration data 
class objects before executing run_model
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import tensorflow as tf

# Make src/ importable when running from scripts/
REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from config import DataConfig, TrainingConfig, CallbackConfig
from run_model import run_model

def _configure_accelerators(set_memory_growth: bool) -> None:
    """
    Print visible GPUs and (optionally) enable memory growth for each GPU.
    Safe no-op if no GPUs are present.
    """
    gpus = tf.config.list_physical_devices("GPU")
    if not gpus:
        print("[tf] Visible GPUs: none")
        return

    print("[tf] Visible GPUs:")
    for i, gpu in enumerate(gpus):
        print(f"  - GPU {i}: {gpu}")

    if set_memory_growth:
        for gpu in gpus:
            try:
                tf.config.experimental.set_memory_growth(gpu, True)
            except Exception as e:
                print(f"[tf] WARNING: could not set memory growth for {gpu}: {e}")


def _enable_mixed_precision(enabled: bool) -> None:
    """
    Enable mixed precision (float16 compute) when supported by the runtime/GPU.
    On CPU-only runs this usually provides no benefit and can hurt performance.
    """
    if not enabled:
        return

    try:
        from tensorflow.keras import mixed_precision
        mixed_precision.set_global_policy("mixed_float16")
        print(f"[tf] Mixed precision enabled: {mixed_precision.global_policy()}")
    except Exception as e:
        print(f"[tf] WARNING: failed to enable mixed precision: {e}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train album-cover classifier (DenseNet201).")

    parser.add_argument(
        "--task",
        type=str,
        required=True,
        choices=["decade", "genre"],
        help="Classification task.",
    )
    parser.add_argument(
        "--data_root",
        type=str,
        default=str(REPO_ROOT / "data"),
        help="Path to dataset root directory (contains CSVs + image subfolders).",
    )

    # Data split / filtering
    parser.add_argument("--min_examples_per_decade", type=int, default=0)
    parser.add_argument("--test_size", type=float, default=0.25)
    parser.add_argument("--val_size", type=float, default=0.20)
    parser.add_argument("--seed", type=int, default=42)

    # tf.data options
    parser.add_argument("--cache", action="store_true", help="Enable ds.cache() after preprocessing.")
    parser.add_argument("--cache_to_disk", action="store_true", help="Cache to disk instead of RAM.")
    parser.add_argument("--cache_path", type=str, default=None, help="Optional cache file path.")
    parser.add_argument("--repeat", action="store_true", help="Repeat the training dataset indefinitely.")

    # Backbone
    parser.add_argument(
        "--backbone",
        type=str,
        default="densenet201",
        help="Backbone key from BACKBONES (e.g., densenet201, resnet50, vgg16, efficientnetb0).",
    )

    # Training hyperparams
    parser.add_argument("--image_size", type=int, default=250)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--stage1_epochs", type=int, default=25)
    parser.add_argument("--stage2_epochs", type=int, default=25)
    parser.add_argument("--stage1_lr", type=float, default=1e-3)
    parser.add_argument("--stage2_lr", type=float, default=1e-4)
    parser.add_argument("--fine_tune_last_n", type=int, default=10)

    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--dense_units", type=int, default=1024)
    parser.add_argument("--l2_reg", type=float, default=0.0)

    # Callbacks (stage-specific, minimal surface)
    # EarlyStopping
    parser.add_argument("--stage1_es_patience", type=int, default=8)
    parser.add_argument("--stage1_es_min_delta", type=float, default=0.005)
    parser.add_argument("--stage2_es_patience", type=int, default=8)
    parser.add_argument("--stage2_es_min_delta", type=float, default=0.005)

    # ReduceLROnPlateau
    parser.add_argument("--stage1_rlr_patience", type=int, default=4)
    parser.add_argument("--stage1_rlr_min_delta", type=float, default=0.005)
    parser.add_argument("--stage1_rlr_factor", type=float, default=0.2)
    parser.add_argument("--stage1_rlr_min_lr", type=float, default=1e-6)

    parser.add_argument("--stage2_rlr_patience", type=int, default=4)
    parser.add_argument("--stage2_rlr_min_delta", type=float, default=0.005)
    parser.add_argument("--stage2_rlr_factor", type=float, default=0.2)
    parser.add_argument("--stage2_rlr_min_lr", type=float, default=1e-6)

    # Fit-loop shortening / control
    parser.add_argument(
        "--steps_per_epoch",
        type=int,
        default=None,
        help="Optional cap on training steps per epoch (None = full epoch).",
    )
    parser.add_argument(
        "--validation_steps",
        type=int,
        default=None,
        help="Optional cap on validation steps (None = full validation).",
    )

    # Runtime / accelerator options
    parser.add_argument(
        "--set_memory_growth",
        action="store_true",
        help="If GPUs are present, enable TF memory growth.",
    )
    parser.add_argument(
        "--mixed_precision",
        action="store_true",
        help="Enable mixed precision (mixed_float16) when supported.",
    )

    # Output
    parser.add_argument("--out_root", type=str, default=str(REPO_ROOT / "outputs"))

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    _configure_accelerators(set_memory_growth=args.set_memory_growth)
    _enable_mixed_precision(enabled=args.mixed_precision)

    data_root = Path(args.data_root)
    if not data_root.exists():
        raise FileNotFoundError(f"--data_root does not exist: {data_root}")

    task = args.task
    out_root = Path(args.out_root)

    cfg_data = DataConfig(
        data_root=data_root,
        min_examples_per_decade=args.min_examples_per_decade,
        test_size=args.test_size,
        val_size=args.val_size,
        random_seed=args.seed,
    )

    cfg_train = TrainingConfig(
        image_size=args.image_size,
        batch_size=args.batch_size,
        stage1_epochs=args.stage1_epochs,
        stage2_epochs=args.stage2_epochs,
        stage1_lr=args.stage1_lr,
        stage2_lr=args.stage2_lr,
        dropout=args.dropout,
        dense_units=args.dense_units,
        fine_tune_last_n=args.fine_tune_last_n,
        l2_reg=args.l2_reg,
        backbone=args.backbone,
        steps_per_epoch=args.steps_per_epoch,
        validation_steps=args.validation_steps,    
        cache=args.cache,
        repeat=args.repeat,
        cache_path=args.cache_path,
    )

    cfg_cb = CallbackConfig(
        stage1_es_patience=args.stage1_es_patience,
        stage1_es_min_delta=args.stage1_es_min_delta,
        stage2_es_patience=args.stage2_es_patience,
        stage2_es_min_delta=args.stage2_es_min_delta,
        stage1_rlr_patience=args.stage1_rlr_patience,
        stage1_rlr_min_delta=args.stage1_rlr_min_delta,
        stage1_rlr_factor=args.stage1_rlr_factor,
        stage1_rlr_min_lr=args.stage1_rlr_min_lr,
        stage2_rlr_patience=args.stage2_rlr_patience,
        stage2_rlr_min_delta=args.stage2_rlr_min_delta,
        stage2_rlr_factor=args.stage2_rlr_factor,
        stage2_rlr_min_lr=args.stage2_rlr_min_lr,
    )

    if cfg_train.repeat and cfg_train.steps_per_epoch is None:
        raise ValueError("--repeat requires --steps_per_epoch (otherwise epochs never terminate).")

    run_model(cfg_data, cfg_train, cfg_cb, task=task, out_root=out_root)


if __name__ == "__main__":
    main()
