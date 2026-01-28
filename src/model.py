"""
model.py

Model definition utilities for album-cover classification.
"""

from __future__ import annotations

from typing import Tuple

import tensorflow as tf
from tensorflow.keras.layers import Input, GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2

from config import TrainingConfig
from backbones import BACKBONES


def build_model(
    num_classes: int,
    cfg: TrainingConfig,
    task_name: str,
) -> Tuple[Model, tf.keras.Model]:
    """
    Create an ImageNet-pretrained backbone + single-label softmax head.

    Args:
        num_classes: Number of classes for the softmax head.
        cfg: Training hyperparameters.
        task_name: Name of the output head (e.g., "decade" or "genre").
        backbone_name: Key into BACKBONES (e.g., "densenet201", "resnet50").

    Returns:
        model      : full classification model
        base_model : the pretrained feature extractor (for fine-tuning control)
    """
    if cfg.backbone not in BACKBONES:
        raise ValueError(f"Unknown backbone '{cfg.backbone}'. Expected one of: {sorted(BACKBONES.keys())}")

    spec = BACKBONES[cfg.backbone]

    inputs = Input(shape=(cfg.image_size, cfg.image_size, 3), name="image")

    # Data augmentation stays in [0, 1] space.
    x = tf.keras.layers.RandomFlip("horizontal_and_vertical")(inputs)
    x = tf.keras.layers.RandomRotation(0.2)(x)
    x = tf.keras.layers.RandomZoom(0.2)(x)
    x = tf.keras.layers.RandomContrast(0.2)(x)

    # Model-specific preprocessing after augmentation.
    x = spec.preprocess_input(x)

    base_model = spec.ctor(weights="imagenet", include_top=False, input_tensor=x)

    # Stage 1: freeze entire backbone
    base_model.trainable = False

    reg = l2(cfg.l2_reg) if getattr(cfg, "l2_reg", 0.0) and cfg.l2_reg > 0 else None

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(cfg.dense_units, activation="relu", kernel_regularizer=reg)(x)
    x = Dropout(cfg.dropout)(x)

    outputs = Dense(num_classes, activation="softmax", name=task_name)(x)

    model = Model(
        inputs=inputs,
        outputs=outputs,
        name=f"{cfg.backbone}_{task_name.capitalize()}_Classifier",
    )
    return model, base_model
