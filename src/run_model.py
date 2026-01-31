"""
run_model.py

Orchestrate a single end-to-end training run for the album-cover classifier.

This module wires together:
- multi-genre CSV loading and stratified train/val/test splits
- tf.data pipeline construction (shuffle/cache/repeat/batch/prefetch)
- backbone + softmax head construction and compilation
- two-stage training (frozen backbone → fine-tune last N layers)
- checkpointing, early stopping, LR scheduling
- evaluation + artifact export (model, histories, reports, confusion matrix, metadata)
"""

from __future__ import annotations

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

from config import DataConfig, TrainingConfig, CallbackConfig
from data import (
    SUPPORTED_GENRES,
    load_and_prepare_multi_genre,
    make_splits,
)
from dataset import make_dataset
from model import build_model, compile_model
from eval import evaluate_and_save

# Make src/ importable when running from scripts/
REPO_ROOT = Path(__file__).resolve().parents[1]


def _ensure_dirs(out_root: Path) -> Dict[str, Path]:
    models_dir = out_root / "models"
    logs_dir = out_root / "logs"
    metrics_dir = out_root / "metrics"
    models_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    return {"models": models_dir, "logs": logs_dir, "metrics": metrics_dir}


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
    

def make_run_tag(backbone: str, task: str, cfg_data, cfg_train, cfg_cb, n: int = 8) -> str:
    """<backbone>_<task>_<short-hash> where hash is over the 3 config dataclasses."""
    payload = {
        "data": asdict(cfg_data),
        "train": asdict(cfg_train),
        "callbacks": asdict(cfg_cb),
    }
    payload["data"].pop("data_root", None) # PosixPath not JSON serializable
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    h = hashlib.sha256(blob).hexdigest()[:n]

    prefix = re.sub(r"[^a-z0-9._-]+", "-", f"{backbone}_{task}".lower()).strip("-")
    return f"{prefix}_{h}"


def run_model(cfg_data: DataConfig, cfg_train: TrainingConfig, cfg_cb: CallbackConfig, task: str, out_root: Path) -> None:
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
    model, base_model = build_model(
        num_classes=len(class_names),
        cfg=cfg_train,
        task_name=task,
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
            "l2_reg": cfg_train.l2_reg,
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
        steps_per_epoch=cfg_train.steps_per_epoch,
        validation_steps=cfg_train.validation_steps,
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
        steps_per_epoch=cfg_train.steps_per_epoch,
        validation_steps=cfg_train.validation_steps,
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