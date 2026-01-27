"""
data.py

Purpose
-------
Small data helpers shared by the training scripts.

This module keeps the dataset handling in one place:
- load one genre CSV
- drop unused columns (if present)
- filter out decades with too few examples
- build absolute image paths and remove missing images
- create stratified train/val/test splits (single-label: decade)

Expected data layout
--------------------
data_root/
  rock_df.csv
  pop_df.csv
  ...
  rock/        (image folder)
  pop/
  ...

Each CSV must contain at least:
- image_file : filename or relative path to image
- decade     : label like "1970s"

Optionally:
- genre_name : original genre label (will be normalized to "genre")
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from config import DataConfig

SUPPORTED_GENRES = ["rock", "pop", "jazz", "classical", "electronic"]

DROP_COLS_IF_PRESENT = [
    "release_group_mbid",
    "release_id",
    "cover_art_id",
    "imUrl",
    "release_year",
]

def load_and_prepare_dataframe(cfg: DataConfig) -> pd.DataFrame:
    """
    Load one genre CSV and normalize it into a clean DataFrame.

    Returns a DataFrame with:
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

    # Drop columns that are irrelevant for training (if present)
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

    # Remove duplicate images (safety check if CSV has duplicates)
    df = df.drop_duplicates(subset="image_file")

    # Build absolute image paths.
    # Expected location: data_root/<genre>/<image_file>
    genre_folder = cfg.data_root / cfg.genre
    df["image_path"] = df["image_file"].apply(lambda x: str((genre_folder / x).resolve()))

    # Remove rows where the image is missing on disk
    exists_mask = df["image_path"].apply(lambda p: os.path.exists(p))
    df = df[exists_mask].copy()

    # Keep only the columns used downstream
    df = df[["image_path", "decade", "genre"]].reset_index(drop=True)

    return df


def make_splits(
    df: pd.DataFrame,
    cfg: DataConfig,
) -> Tuple[pd.Series, pd.Series, pd.Series, np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """
    Create stratified train/val/test splits over the decade label.

    Returns:
      X_train, X_val, X_test  (pd.Series of image paths)
      y_train, y_val, y_test  (one-hot arrays)
      class_names             (list of decade labels, aligned with one-hot columns)
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
