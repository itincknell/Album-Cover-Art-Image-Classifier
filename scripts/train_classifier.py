#!/usr/bin/env python3
"""
scripts/train_classifier.py

Train a DenseNet201 single-label classifier for album cover art.

Tasks
-----
- decade: predict release decade (trained on unified multi-genre dataset)
- genre : predict genre 

Repo layout
---------------------
repo_root/
  src/
    config.py
    data.py
    dataset.py
    model.py
  scripts/
    train_classifier.py
  data/
    rock_df.csv, pop_df.csv, ...
    rock/, pop/, ...   (images)

Outputs
-------
outputs/
  models/
  logs/
  metrics/
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from dataclasses import asdict
import sys
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd

import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

# Make src/ importable when running from scripts/
REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from config import DataConfig, TrainingConfig, CallbackConfig
from data import (
    SUPPORTED_GENRES,
    load_and_prepare_multi_genre,
    make_splits,
)
from dataset import make_dataset
from model import build_model, compile_model
from eval import evaluate_and_save

def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _git_commit_hash(repo_root: Path) -> str | None:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=str(repo_root),
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        return out or None
    except Exception:
        return None


def _ensure_dirs(out_root: Path) -> Dict[str, Path]:
    models_dir = out_root / "models"
    logs_dir = out_root / "logs"
    metrics_dir = out_root / "metrics"
    models_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    return {"models": models_dir, "logs": logs_dir, "metrics": metrics_dir}


# parse_args + main with only the requested callback args (stage-specific)
# (backbone args included as before)

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
    parser.add_argument("--min_examples_per_decade", type=int, default=2500)
    parser.add_argument("--test_size", type=float, default=0.25)
    parser.add_argument("--val_size", type=float, default=0.20)
    parser.add_argument("--seed", type=int, default=42)

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
    parser.add_argument("--l2_strength", type=float, default=0.0)

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

    # Output
    parser.add_argument("--out_root", type=str, default=str(REPO_ROOT / "outputs"))

    return parser.parse_args()


def make_run_tag(backbone: str, task: str, cfg_data, cfg_train, cfg_cb, n: int = 8) -> str:
    """<backbone>_<task>_<short-hash> where hash is over the 3 config dataclasses."""
    payload = {
        "data": asdict(cfg_data),
        "train": asdict(cfg_train),
        "callbacks": asdict(cfg_cb),
    }
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    h = hashlib.sha256(blob).hexdigest()[:n]

    prefix = re.sub(r"[^a-z0-9._-]+", "-", f"{backbone}_{task}".lower()).strip("-")
    return f"{prefix}_{h}"


def run_one(cfg_data: DataConfig, cfg_train: TrainingConfig, cfg_cb: CallbackConfig, task: str, out_root: Path) -> None:
    np.random.seed(cfg_data.random_seed)
    tf.random.set_seed(cfg_data.random_seed)

    dirs = _ensure_dirs(out_root)

    label_col = "decade" if task == "decade" else "genre"

    # -------------------------
    # Load data
    # -------------------------
    df = load_and_prepare_multi_genre(cfg_data, SUPPORTED_GENRES)

    # -------------------------
    # Splits
    # -------------------------
    X_train, X_val, X_test, y_train, y_val, y_test, class_names = make_splits(
        df, cfg_data, label_col=task
    )

    train_ds = make_dataset(X_train, y_train, cfg_train, training=True, seed=cfg_data.random_seed)
    val_ds = make_dataset(X_val, y_val, cfg_train, training=False, seed=cfg_data.random_seed)
    test_ds = make_dataset(X_test, y_test, cfg_train, training=False, seed=cfg_data.random_seed)

    # -------------------------
    # Build model
    # -------------------------
    model_tag = make_run_tag(cfg_train.backbone, task, cfg_data, cfg_train, cfg_cb)
    output_name = task  # keep output name aligned with task
    model, base_model = build_model(
        num_classes=len(class_names),
        cfg=cfg_train,
        output_name=output_name,
        model_name=model_tag,
    )

    # -------------------------
    # Run metadata (start)
    # -------------------------
    run_meta = {
        "started_at_utc": _utc_now_iso(),
        "task": task,
        "label_col": label_col,
        "data_root": str(cfg_data.data_root),
        "dataset_size": int(len(df)),
        "class_names": class_names,
        "class_counts": df[label_col].astype(str).value_counts().to_dict(),
        "seed": cfg_data.random_seed,
        "test_size": cfg_data.test_size,
        "val_size": cfg_data.val_size,
        "min_examples_per_decade": cfg_data.min_examples_per_decade,
        "training": {
            "image_size": cfg_train.image_size,
            "batch_size": cfg_train.batch_size,
            "stage1_epochs": cfg_train.stage1_epochs,
            "stage2_epochs": cfg_train.stage2_epochs,
            "stage1_lr": cfg_train.stage1_lr,
            "stage2_lr": cfg_train.stage2_lr,
            "fine_tune_last_n": cfg_train.fine_tune_last_n,
            "dropout": cfg_train.dropout,
            "dense_units": cfg_train.dense_units,
            "l2_strength": cfg_train.l2_strength,
        },
        "env": {
            "python": sys.version.split()[0],
            "tensorflow": tf.__version__,
        },
        "git_commit": _git_commit_hash(REPO_ROOT),
    }

    # -------------------------
    # Callbacks
    # -------------------------
    ckpt_stage1 = dirs["models"] / f"{model_tag}_best_stage1.keras"
    ckpt_stage2 = dirs["models"] / f"{model_tag}_best_stage2.keras"

    callbacks_stage1 = [
        ModelCheckpoint(str(ckpt_stage1), monitor="val_loss", save_best_only=True, mode="min"),
        EarlyStopping(
            monitor="val_loss",
            patience=cfg_cb.stage1_es_patience,
            min_delta=cfg_cb.stage1_es_min_delta,
            restore_best_weights=True,
            verbose=0,
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=cfg_cb.stage1_rlr_factor,
            patience=cfg_cb.stage1_rlr_patience,
            min_delta=cfg_cb.stage1_rlr_min_delta,
            min_lr=cfg_cb.stage1_rlr_min_lr,
            verbose=1,
        ),
    ]

    callbacks_stage2 = [
        ModelCheckpoint(str(ckpt_stage2), monitor="val_loss", save_best_only=True, mode="min"),
        EarlyStopping(
            monitor="val_loss",
            patience=cfg_cb.stage2_es_patience,
            min_delta=cfg_cb.stage2_es_min_delta,
            restore_best_weights=True,
            verbose=0,
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=cfg_cb.stage2_rlr_factor,
            patience=cfg_cb.stage2_rlr_patience,
            min_delta=cfg_cb.stage2_rlr_min_delta,
            min_lr=cfg_cb.stage2_rlr_min_lr,
            verbose=1,
        ),
    ]

    # -------------------------
    # Stage 1: head training
    # -------------------------
    print(f"\n[{model_tag}] Stage 1: training classifier head (backbone frozen)")
    compile_model(model, lr=cfg_train.stage1_lr)

    history1 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=cfg_train.stage1_epochs,
        callbacks=callbacks_stage1,
        verbose=1,
    )

    pd.DataFrame(history1.history).to_csv(
        dirs["logs"] / f"{model_tag}_history_stage1.csv", index=False
    )

    # -------------------------
    # Stage 2: fine-tune last N layers
    # -------------------------
    print(f"\n[{model_tag}] Stage 2: fine-tuning last {cfg_train.fine_tune_last_n} backbone layers")
    base_model.trainable = True
    for layer in base_model.layers[:-cfg_train.fine_tune_last_n]:
        layer.trainable = False

    compile_model(model, lr=cfg_train.stage2_lr)

    history2 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=cfg_train.stage2_epochs,
        callbacks=callbacks_stage2,
        verbose=1,
    )

    pd.DataFrame(history2.history).to_csv(
        dirs["logs"] / f"{model_tag}_history_stage2.csv", index=False
    )

    # -------------------------
    # Save final model
    # -------------------------
    final_model_path = dirs["models"] / f"{model_tag}.keras"
    model.save(final_model_path)
    print(f"\n[{model_tag}] Saved model: {final_model_path}")

    # -------------------------
    # Evaluation
    # -------------------------
    print(f"\n[{model_tag}] Evaluating on test set")
    test_metrics = model.evaluate(test_ds, verbose=1)
    print(dict(zip(model.metrics_names, test_metrics)))

    y_pred_prob = model.predict(test_ds, verbose=1)
    y_pred_int = np.argmax(y_pred_prob, axis=1)
    y_true_int = np.argmax(y_test, axis=1)

    # update metadata with finish time + artifact paths
    run_meta["finished_at_utc"] = _utc_now_iso()
    run_meta["artifacts"] = {
        "model": str(final_model_path),
        "history_stage1": str(dirs["logs"] / f"{model_tag}_history_stage1.csv"),
        "history_stage2": str(dirs["logs"] / f"{model_tag}_history_stage2.csv"),
    }

    evaluate_and_save(
        y_true_int=y_true_int,
        y_pred_int=y_pred_int,
        class_names=class_names,
        metrics_dir=dirs["metrics"],
        tag=model_tag,
        meta=run_meta,
    )

    print(f"\n[{model_tag}] Saved metrics to: {dirs['metrics']}")


def main() -> None:
    args = parse_args()

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
        l2_strength=args.l2_strength,
        backbone=args.backbone,
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

    run_one(cfg_data, cfg_train, cfg_cb, task=task, out_root=out_root)


if __name__ == "__main__":
    main()
