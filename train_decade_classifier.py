#!/usr/bin/env python3
"""
Album Cover Decade Classification (TensorFlow / DenseNet201)

Purpose
-------
This script trains an image classifier that predicts the release decade of an album
based on its cover art.

Model
-----
- Backbone: DenseNet201 pretrained on ImageNet
- Head: GlobalAveragePooling -> Dense(1024) -> Dropout -> Softmax over decades
- Training: two-stage training
  1) Train head with backbone frozen
  2) Fine-tune the last N layers of the backbone at a lower learning rate

Data Layout (expected)
----------------------
A folder containing one CSV per genre:

    data_root/
      rock_df.csv
      pop_df.csv
      jazz_df.csv
      classical_df.csv
      electronic_df.csv
      rock/        (image folder)
      pop/
      ...

Each CSV must contain at least:
- image_file : filename OR relative path to the image
- decade     : label such as "1970s", "1980s", etc. (string or int)
- genre_name : (optional) original genre label

The script will:
- drop columns that are not needed (if present)
- filter out decades with fewer than --min_examples_per_decade examples
- validate that image paths exist
- stratify the split by decade (single-label stratification)

How to Run
----------
Example (train a single genre):
    python train_decade_classifier.py --data_root ./data --genre rock

Train all supported genres:
    python train_decade_classifier.py --data_root ./data --genre all

Outputs
-------
Creates:
    outputs/
      models/
        DenseNet201_<genre>_DecadeClassifier.keras
      metrics/
        DenseNet201_<genre>_classification_report.csv
        DenseNet201_<genre>_confusion_matrix.png
      logs/
        DenseNet201_<genre>_history_stage1.csv
        DenseNet201_<genre>_history_stage2.csv

Dependencies
------------
pip install tensorflow pandas numpy scikit-learn matplotlib seaborn

Notes
-----
- For large datasets: consider turning off dataset caching or caching to disk.
- If running on Apple Silicon: TensorFlow + Metal acceleration can help.
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple, List, Dict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras.applications import DenseNet201
from tensorflow.keras.layers import (
    Input,
    GlobalAveragePooling2D,
    Dense,
    Dropout,
)
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, classification_report

import seaborn as sns


# -----------------------------
# Configuration / Defaults
# -----------------------------

SUPPORTED_GENRES = ["rock", "pop", "jazz", "classical", "electronic"]


@dataclass
class TrainConfig:
    data_root: Path
    genre: str
    image_size: int = 250
    batch_size: int = 32
    stage1_epochs: int = 25
    stage2_epochs: int = 25
    stage1_lr: float = 1e-3
    stage2_lr: float = 1e-4
    dropout: float = 0.5
    dense_units: int = 1024
    min_examples_per_decade: int = 2500
    val_size: float = 0.20  # fraction of train set used for validation
    test_size: float = 0.25
    random_seed: int = 42
    fine_tune_last_n: int = 10


# -----------------------------
# Data Loading / Cleaning
# -----------------------------

DROP_COLS_IF_PRESENT = [
    "release_group_mbid",
    "release_id",
    "cover_art_id",
    "imUrl",
    "release_year",
]


def load_and_prepare_dataframe(cfg: TrainConfig) -> pd.DataFrame:
    """
    Load one genre CSV, drop unused columns (if present), filter rare decades,
    and construct full image paths.

    Returns a clean DataFrame with columns:
        image_path (absolute path)
        decade     (string label)
        genre      (string)
    """
    csv_path = cfg.data_root / f"{cfg.genre}_df.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing CSV: {csv_path}")

    df = pd.read_csv(csv_path)

    # Normalize the genre column name (some datasets store as 'genre_name')
    if "genre_name" in df.columns and "genre" not in df.columns:
        df = df.rename(columns={"genre_name": "genre"})
    elif "genre" not in df.columns:
        df["genre"] = cfg.genre

    # Drop columns that are irrelevant for this model (if present)
    cols_to_drop = [c for c in DROP_COLS_IF_PRESENT if c in df.columns]
    if cols_to_drop:
        df = df.drop(columns=cols_to_drop)

    # Basic schema checks
    required = {"image_file", "decade"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{csv_path.name} is missing required columns: {sorted(missing)}")

    # Filter decades with too few examples (helps avoid extreme imbalance)
    decade_counts = df["decade"].value_counts()
    keep_decades = decade_counts[decade_counts >= cfg.min_examples_per_decade].index
    df = df[df["decade"].isin(keep_decades)].copy()

    # Remove duplicate images (safe guard if CSV has duplicates)
    df = df.drop_duplicates(subset="image_file")

    # Construct absolute image paths.
    # Expected location: data_root/<genre>/<image_file>
    genre_folder = cfg.data_root / cfg.genre
    df["image_path"] = df["image_file"].apply(lambda x: str((genre_folder / x).resolve()))

    # Remove rows where the image is missing on disk
    exists_mask = df["image_path"].apply(lambda p: os.path.exists(p))
    df = df[exists_mask].copy()

    # Keep only the columns we actually use downstream
    df = df[["image_path", "decade", "genre"]].reset_index(drop=True)

    print(f"\n[{cfg.genre}] Prepared dataset")
    print(f"  Examples: {len(df):,}")
    print(f"  Decades:  {sorted(df['decade'].unique())}")
    print("  Counts:")
    print(df["decade"].value_counts().sort_index())

    return df


# -----------------------------
# Splits and Label Encoding
# -----------------------------

def make_splits(
    df: pd.DataFrame,
    cfg: TrainConfig
) -> Tuple[pd.Series, pd.Series, pd.Series, np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """
    Create stratified train/val/test splits and return:
        X_train, X_val, X_test  (pd.Series of image paths)
        y_train, y_val, y_test  (one-hot arrays)
        class_names             (list of decade labels)
    """
    # Encode decade labels into integer classes
    le = LabelEncoder()
    y_int = le.fit_transform(df["decade"].astype(str))
    class_names = list(le.classes_)

    # Train/test split (stratified)
    X_train, X_test, y_train_int, y_test_int = train_test_split(
        df["image_path"],
        y_int,
        test_size=cfg.test_size,
        random_state=cfg.random_seed,
        stratify=y_int,
    )

    # Train/val split (also stratified)
    X_train, X_val, y_train_int, y_val_int = train_test_split(
        X_train,
        y_train_int,
        test_size=cfg.val_size,
        random_state=cfg.random_seed,
        stratify=y_train_int,
    )

    # One-hot for softmax classification
    num_classes = len(class_names)
    y_train = tf.keras.utils.to_categorical(y_train_int, num_classes=num_classes)
    y_val = tf.keras.utils.to_categorical(y_val_int, num_classes=num_classes)
    y_test = tf.keras.utils.to_categorical(y_test_int, num_classes=num_classes)

    # Reset indexes so dataset iteration is clean
    X_train = X_train.reset_index(drop=True)
    X_val = X_val.reset_index(drop=True)
    X_test = X_test.reset_index(drop=True)

    return X_train, X_val, X_test, y_train, y_val, y_test, class_names


# -----------------------------
# tf.data Pipeline
# -----------------------------

def preprocess_image(path: tf.Tensor, label: tf.Tensor, image_size: int) -> Tuple[tf.Tensor, tf.Tensor]:
    """
    Read an image file and convert it to a float32 tensor in [0, 1].
    """
    img_bytes = tf.io.read_file(path)
    img = tf.image.decode_image(img_bytes, channels=3, expand_animations=False)
    img = tf.image.resize(img, [image_size, image_size])
    img = tf.cast(img, tf.float32) / 255.0

    # Set static shape for performance/stability
    img.set_shape([image_size, image_size, 3])
    return img, label


def make_dataset(
    x_paths: pd.Series,
    y_onehot: np.ndarray,
    cfg: TrainConfig,
    training: bool,
) -> tf.data.Dataset:
    """
    Build a performant tf.data.Dataset with shuffling/batching/prefetch.
    """
    ds = tf.data.Dataset.from_tensor_slices((x_paths.values, y_onehot))

    if training:
        ds = ds.shuffle(buffer_size=len(x_paths), seed=cfg.random_seed, reshuffle_each_iteration=True)

    ds = ds.map(
        lambda p, y: preprocess_image(p, y, cfg.image_size),
        num_parallel_calls=tf.data.AUTOTUNE,
    )

    ds = ds.batch(cfg.batch_size)
    ds = ds.prefetch(tf.data.AUTOTUNE)

    # If you have corrupt images, ignore_errors prevents crashing the run
    ds = ds.ignore_errors()
    return ds


# -----------------------------
# Model Definition
# -----------------------------

def build_model(num_classes: int, cfg: TrainConfig) -> Tuple[Model, tf.keras.Model]:
    """
    Create a DenseNet201 classifier.

    Returns:
        model      : full classification model
        base_model : the pretrained feature extractor (for fine-tuning control)
    """
    inputs = Input(shape=(cfg.image_size, cfg.image_size, 3), name="image")

    # NOTE: augmentation is done in-model so it's only active during training
    x = tf.keras.layers.RandomFlip("horizontal_and_vertical")(inputs)
    x = tf.keras.layers.RandomRotation(0.2)(x)
    x = tf.keras.layers.RandomZoom(0.2)(x)
    x = tf.keras.layers.RandomContrast(0.2)(x)

    base_model = DenseNet201(weights="imagenet", include_top=False, input_tensor=x)

    # Stage 1: freeze entire backbone
    base_model.trainable = False

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(cfg.dense_units, activation="relu")(x)
    x = Dropout(cfg.dropout)(x)

    outputs = Dense(num_classes, activation="softmax", name="decade")(x)

    model = Model(inputs=inputs, outputs=outputs, name="DenseNet201_DecadeClassifier")
    return model, base_model


# -----------------------------
# Training / Evaluation
# -----------------------------

def top_2_accuracy(y_true, y_pred):
    return tf.keras.metrics.top_k_categorical_accuracy(y_true, y_pred, k=2)


def compile_model(model: Model, lr: float) -> None:
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss="categorical_crossentropy",
        metrics=["accuracy", top_2_accuracy],
    )


def plot_and_save_confusion_matrix(
    y_true_int: np.ndarray,
    y_pred_int: np.ndarray,
    class_names: List[str],
    out_png: Path,
    title: str,
) -> None:
    cm = confusion_matrix(y_true_int, y_pred_int)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", xticklabels=class_names, yticklabels=class_names)
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(out_png)
    plt.close()


def run_training_for_genre(cfg: TrainConfig) -> None:
    # Output folders
    out_root = Path("outputs")
    models_dir = out_root / "models"
    metrics_dir = out_root / "metrics"
    logs_dir = out_root / "logs"
    models_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    model_tag = f"DenseNet201_{cfg.genre}_DecadeClassifier"
    model_path = models_dir / f"{model_tag}.keras"

    # Load / split data
    df = load_and_prepare_dataframe(cfg)
    X_train, X_val, X_test, y_train, y_val, y_test, class_names = make_splits(df, cfg)

    train_ds = make_dataset(X_train, y_train, cfg, training=True)
    val_ds = make_dataset(X_val, y_val, cfg, training=False)
    test_ds = make_dataset(X_test, y_test, cfg, training=False)

    # Build model
    model, base_model = build_model(num_classes=len(class_names), cfg=cfg)

    # Callbacks (stage 1)
    ckpt_path = models_dir / f"{model_tag}_best_stage1.keras"
    callbacks = [
        ModelCheckpoint(str(ckpt_path), monitor="val_loss", save_best_only=True, mode="min"),
        EarlyStopping(monitor="val_loss", patience=8, min_delta=0.005, restore_best_weights=True),
        ReduceLROnPlateau(monitor="val_loss", factor=0.2, patience=4, min_delta=0.005, min_lr=1e-6, verbose=1),
    ]

    # -------------------------
    # Stage 1: Train top layers
    # -------------------------
    print(f"\n[{cfg.genre}] Stage 1: training classifier head (backbone frozen)")
    compile_model(model, lr=cfg.stage1_lr)

    history1 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=cfg.stage1_epochs,
        callbacks=callbacks,
        verbose=1,
    )

    pd.DataFrame(history1.history).to_csv(logs_dir / f"{model_tag}_history_stage1.csv", index=False)

    # -------------------------
    # Stage 2: Fine-tune backbone
    # -------------------------
    print(f"\n[{cfg.genre}] Stage 2: fine-tuning last {cfg.fine_tune_last_n} backbone layers")
    base_model.trainable = True

    # Freeze all but last N layers
    for layer in base_model.layers[:-cfg.fine_tune_last_n]:
        layer.trainable = False

    # Reset callbacks (stage 2)
    ckpt_path2 = models_dir / f"{model_tag}_best_stage2.keras"
    callbacks2 = [
        ModelCheckpoint(str(ckpt_path2), monitor="val_loss", save_best_only=True, mode="min"),
        EarlyStopping(monitor="val_loss", patience=8, min_delta=0.005, restore_best_weights=True),
        ReduceLROnPlateau(monitor="val_loss", factor=0.2, patience=4, min_delta=0.005, min_lr=1e-6, verbose=1),
    ]

    compile_model(model, lr=cfg.stage2_lr)

    history2 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=cfg.stage2_epochs,
        callbacks=callbacks2,
        verbose=1,
    )

    pd.DataFrame(history2.history).to_csv(logs_dir / f"{model_tag}_history_stage2.csv", index=False)

    # Save final model
    model.save(model_path)
    print(f"\n[{cfg.genre}] Saved model: {model_path}")

    # -------------------------
    # Evaluation
    # -------------------------
    print(f"\n[{cfg.genre}] Evaluating on test set")
    test_metrics = model.evaluate(test_ds, verbose=1)
    metric_names = model.metrics_names
    print(dict(zip(metric_names, test_metrics)))

    # Predictions
    y_pred_prob = model.predict(test_ds)
    y_pred_int = np.argmax(y_pred_prob, axis=1)

    # Recover y_true integers from one-hot
    y_true_int = np.argmax(y_test, axis=1)

    # Save classification report
    report = classification_report(y_true_int, y_pred_int, target_names=class_names, output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    report_df.to_csv(metrics_dir / f"{model_tag}_classification_report.csv", index=True)

    # Save confusion matrix figure
    cm_png = metrics_dir / f"{model_tag}_confusion_matrix.png"
    plot_and_save_confusion_matrix(
        y_true_int=y_true_int,
        y_pred_int=y_pred_int,
        class_names=class_names,
        out_png=cm_png,
        title=f"Confusion Matrix - Decade ({model_tag})",
    )
    print(f"[{cfg.genre}] Saved metrics to: {metrics_dir}")


# -----------------------------
# CLI Entry Point
# -----------------------------

def parse_args() -> TrainConfig:
    parser = argparse.ArgumentParser(description="Train album-cover decade classifier (DenseNet201).")

    parser.add_argument("--data_root", type=str, required=True, help="Path to dataset root directory.")
    parser.add_argument("--genre", type=str, default="rock", help="One genre (rock/pop/...) or 'all'.")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--stage1_epochs", type=int, default=25)
    parser.add_argument("--stage2_epochs", type=int, default=25)
    parser.add_argument("--min_examples_per_decade", type=int, default=2500)
    parser.add_argument("--fine_tune_last_n", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    cfg = TrainConfig(
        data_root=Path(args.data_root),
        genre=args.genre.lower(),
        batch_size=args.batch_size,
        stage1_epochs=args.stage1_epochs,
        stage2_epochs=args.stage2_epochs,
        min_examples_per_decade=args.min_examples_per_decade,
        fine_tune_last_n=args.fine_tune_last_n,
        random_seed=args.seed,
    )

    if not cfg.data_root.exists():
        raise FileNotFoundError(f"--data_root does not exist: {cfg.data_root}")

    return cfg


def main() -> None:
    cfg = parse_args()

    if cfg.genre == "all":
        for g in SUPPORTED_GENRES:
            run_training_for_genre(TrainConfig(**{**cfg.__dict__, "genre": g}))
    else:
        if cfg.genre not in SUPPORTED_GENRES:
            raise ValueError(f"Unknown genre '{cfg.genre}'. Expected one of {SUPPORTED_GENRES} or 'all'.")
        run_training_for_genre(cfg)


if __name__ == "__main__":
    main()
