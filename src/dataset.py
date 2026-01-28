"""
dataset.py

tf.data input pipeline for album-cover classification.

This module is task-agnostic: it assumes labels are already one-hot encoded,
so it works for both genre-classification and decade-classification.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
import pandas as pd
import tensorflow as tf

from config import TrainingConfig


def preprocess_image(
    path: tf.Tensor,
    label: tf.Tensor,
    image_size: int,
) -> Tuple[tf.Tensor, tf.Tensor]:
    """
    Read a JPEG image file and convert it to a float32 tensor in [0, 1].
    """
    img_bytes = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img_bytes, channels=3)
    img = tf.image.resize(img, [image_size, image_size])
    img = tf.cast(img, tf.float32) / 255.0

    # Set static shape for performance/stability
    img.set_shape([image_size, image_size, 3])
    return img, label


def make_dataset(
    x_paths: pd.Series,
    y_onehot: np.ndarray,
    cfg: TrainingConfig,
    training: bool,
) -> tf.data.Dataset:
    """
    Build a performant tf.data.Dataset with shuffling/batching/prefetch.
    """
    ds = tf.data.Dataset.from_tensor_slices((x_paths.values, y_onehot))

    if training:
        ds = ds.shuffle(
            buffer_size=len(x_paths),
            seed=cfg.random_seed,
            reshuffle_each_iteration=True,
        )

    ds = ds.map(
        lambda p, y: preprocess_image(p, y, cfg.image_size),
        num_parallel_calls=tf.data.AUTOTUNE,
    )

    ds = ds.batch(cfg.batch_size)
    ds = ds.prefetch(tf.data.AUTOTUNE)

    # If you have corrupt images, ignore_errors prevents crashing the run
    ds = ds.ignore_errors()
    return ds
