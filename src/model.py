"""
model.py

Model definition utilities for album-cover classification.
"""

from __future__ import annotations

from typing import Tuple

import tensorflow as tf
from tensorflow.keras.applications import DenseNet201
from tensorflow.keras.layers import Input, GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2

from config import TrainingConfig


def build_model(
    num_classes: int,
    cfg: TrainingConfig,
    task_name: str = "label",
) -> Tuple[Model, tf.keras.Model]:
    """
    Create a DenseNet201 single-label softmax classifier.

    Args:
        num_classes: Number of classes for the softmax head.
        cfg: Training hyperparameters.
        task_name: Name of the output head (e.g., "decade" or "genre").

    Returns:
        model      : full classification model
        base_model : the pretrained feature extractor (for fine-tuning control)
    """
    inputs = Input(shape=(cfg.image_size, cfg.image_size, 3), name="image")

    # Data augmentation is in-model so it is only active during training.
    x = tf.keras.layers.RandomFlip("horizontal_and_vertical")(inputs)
    x = tf.keras.layers.RandomRotation(0.2)(x)
    x = tf.keras.layers.RandomZoom(0.2)(x)
    x = tf.keras.layers.RandomContrast(0.2)(x)

    base_model = DenseNet201(weights="imagenet", include_top=False, input_tensor=x)

    # Stage 1: freeze entire backbone
    base_model.trainable = False

    reg = l2(cfg.l2_reg) if getattr(cfg, "l2_reg", 0.0) and cfg.l2_reg > 0 else None

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(cfg.dense_units, activation="relu", kernel_regularizer=reg)(x)
    x = Dropout(cfg.dropout)(x)

    outputs = Dense(num_classes, activation="softmax", name=task_name)(x)

    model = Model(inputs=inputs, outputs=outputs, name=f"DenseNet201_{task_name.capitalize()}Classifier")
    return model, base_model
