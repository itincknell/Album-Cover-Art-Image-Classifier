"""
data.py

Data loading and split utilities for album-cover classification.

Dataset layout
--------------
data_root/
  rock_df.csv
  pop_df.csv
  ...
  rock/          (images)
  pop/
  ...

Each per-genre CSV must contain:
- image_file
- decade

Optional columns:
- genre or genre_name

This module supports single-label softmax classification by choosing a label column
(e.g., "genre" or "decade") at split time.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple, List

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

def _resolve_image_path(data_root: Path, genre: str, image_file: str) -> str:
    """
    Resolve image paths:
    - absolute path: use as-is
    - relative path: prefer data_root/<image_file> if it exists, else data_root/<genre>/<image_file>
    """
    p = Path(str(image_file))

    if p.is_absolute():
        return str(p)

    candidate1 = (data_root / p).resolve()
    if candidate1.exists():
        return str(candidate1)

    candidate2 = (data_root / genre / p).resolve()
    return str(candidate2)


def load_and_prepare_dataframe(cfg: DataConfig) -> pd.DataFrame:
    """
    Load one genre CSV, normalize schema, drop unused columns (if present),
    validate image paths, and return a clean DataFrame.

    Returns columns:
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

    # Remove duplicate images (safe guard if CSV has duplicates)
    df = df.drop_duplicates(subset="image_file")

    # Construct absolute image paths + validate existence
    df["image_path"] = df["image_file"].apply(lambda x: _resolve_image_path(cfg.data_root, cfg.genre, x))
    exists_mask = df["image_path"].apply(lambda p: os.path.exists(p))
    df = df[exists_mask].copy()

    df = df[["image_path", "decade", "genre"]].reset_index(drop=True)
    return df


def load_and_prepare_multi_genre(cfg: DataConfig, genres: List[str]) -> pd.DataFrame:
    """
    Load and concatenate multiple per-genre CSVs into one unified dataset.

    Filtering decades by min_examples_per_decade is applied after concatenation,
    based on overall decade counts in the unified dataset.
    """
    dfs = []
    for g in genres:
        g_cfg = DataConfig(**{**cfg.__dict__, "genre": g})
        dfs.append(load_and_prepare_dataframe(g_cfg))

    if not dfs:
        raise ValueError("No genres provided")

    unified = pd.concat(dfs, ignore_index=True)

    decade_counts = unified["decade"].value_counts()
    keep_decades = decade_counts[decade_counts >= cfg.min_examples_per_decade].index
    unified = unified[unified["decade"].isin(keep_decades)].copy()

    # Crosstabs: totals, genre totals, decade totals, and genre x decade matrix
    print("\n[dataset] Unified dataset after decade filtering")
    print(f"  Total examples: {len(unified):,}")

    print("\n[dataset] Counts by genre:")
    print(unified["genre"].value_counts().sort_index())

    print("\n[dataset] Counts by decade:")
    print(unified["decade"].value_counts().sort_index())

    print("\n[dataset] Genre x Decade crosstab:")
    ctab = pd.crosstab(
        unified["genre"],
        unified["decade"],
        rownames=["genre"],
        colnames=["decade"],
        dropna=False,
        margins=True,
        margins_name="Total",
    )
    print(ctab)

    unified = unified.reset_index(drop=True)
    return unified


def make_splits(
    df: pd.DataFrame,
    cfg: DataConfig,
    label_col: str = "decade",
) -> Tuple[pd.Series, pd.Series, pd.Series, np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """
    Create stratified train/val/test splits for a single-label softmax classifier.

    Returns:
        X_train, X_val, X_test  (pd.Series of image paths)
        y_train, y_val, y_test  (one-hot arrays)
        class_names             (sorted label names as strings)
    """
    if label_col not in df.columns:
        raise ValueError(f"label_col '{label_col}' not found in df columns: {sorted(df.columns)}")

    le = LabelEncoder()
    y_int = le.fit_transform(df[label_col].astype(str))
    class_names = list(le.classes_)

    X_train, X_test, y_train_int, y_test_int = train_test_split(
        df["image_path"],
        y_int,
        test_size=cfg.test_size,
        random_state=cfg.random_seed,
        stratify=y_int,
    )

    X_train, X_val, y_train_int, y_val_int = train_test_split(
        X_train,
        y_train_int,
        test_size=cfg.val_size,
        random_state=cfg.random_seed,
        stratify=y_train_int,
    )

    num_classes = len(class_names)
    y_train = tf.keras.utils.to_categorical(y_train_int, num_classes=num_classes)
    y_val = tf.keras.utils.to_categorical(y_val_int, num_classes=num_classes)
    y_test = tf.keras.utils.to_categorical(y_test_int, num_classes=num_classes)

    X_train = X_train.reset_index(drop=True)
    X_val = X_val.reset_index(drop=True)
    X_test = X_test.reset_index(drop=True)

    return X_train, X_val, X_test, y_train, y_val, y_test, class_names
