"""
backbones.py

Backbone registry for ImageNet-pretrained Keras application models.

Each entry provides:
- ctor: model constructor (include_top=False, weights="imagenet", input_tensor=...)
- preprocess_input: callable to apply after augmentation in model.py

Use preprocess_input inside model.py after augmentation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict

import tensorflow as tf


@dataclass(frozen=True)
class BackboneSpec:
    ctor: Callable[..., tf.keras.Model]
    preprocess_input: Callable[[tf.Tensor], tf.Tensor]


def _scale_255(fn: Callable[[tf.Tensor], tf.Tensor]) -> Callable[[tf.Tensor], tf.Tensor]:
    """
    Keras applications preprocess_input typically expects 0..255-ish inputs, while our
    pipeline/augmentation uses float images in [0, 1]. Wrap preprocess_input so model.py
    can call spec.preprocess_input(x) directly after augmentation.
    """
    return lambda x: fn(x * 255.0)


BACKBONES: Dict[str, BackboneSpec] = {
    # DenseNet
    "densenet121": BackboneSpec(
        ctor=tf.keras.applications.DenseNet121,
        preprocess_input=_scale_255(tf.keras.applications.densenet.preprocess_input),
    ),
    "densenet201": BackboneSpec(
        ctor=tf.keras.applications.DenseNet201,
        preprocess_input=_scale_255(tf.keras.applications.densenet.preprocess_input),
    ),

    # ResNet
    "resnet50": BackboneSpec(
        ctor=tf.keras.applications.ResNet50,
        preprocess_input=_scale_255(tf.keras.applications.resnet.preprocess_input),
    ),

    # VGG
    "vgg16": BackboneSpec(
        ctor=tf.keras.applications.VGG16,
        preprocess_input=_scale_255(tf.keras.applications.vgg16.preprocess_input),
    ),

    # EfficientNet
    "efficientnetb0": BackboneSpec(
        ctor=tf.keras.applications.EfficientNetB0,
        preprocess_input=_scale_255(tf.keras.applications.efficientnet.preprocess_input),
    ),

    # MobileNetV3Small (good for test runs)
    "mobilenetv3small": BackboneSpec(
        ctor=tf.keras.applications.MobileNetV3Small,
        preprocess_input=_scale_255(tf.keras.applications.mobilenet_v3.preprocess_input),
    ),
}
