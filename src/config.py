"""
config.py

Shared configuration dataclasses for the album-cover classification project.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class DataConfig:
    data_root: Path
    min_examples_per_decade: int = 2500
    test_size: float = 0.25
    val_size: float = 0.20
    random_seed: int = 42


@dataclass
class TrainingConfig:
    image_size: int = 250
    batch_size: int = 32

    stage1_epochs: int = 25
    stage2_epochs: int = 25
    stage1_lr: float = 1e-3
    stage2_lr: float = 1e-4

    dropout: float = 0.5
    dense_units: int = 1024
    fine_tune_last_n: int = 10

    backbone: str = "densenet201"


@dataclass
class CallbackConfig:
    # EarlyStopping (stage-specific)
    stage1_es_patience: int = 8
    stage1_es_min_delta: float = 0.005
    stage2_es_patience: int = 8
    stage2_es_min_delta: float = 0.005

    # ReduceLROnPlateau (stage-specific)
    stage1_rlr_patience: int = 4
    stage1_rlr_min_delta: float = 0.005
    stage1_rlr_factor: float = 0.2
    stage1_rlr_min_lr: float = 1e-6

    stage2_rlr_patience: int = 4
    stage2_rlr_min_delta: float = 0.005
    stage2_rlr_factor: float = 0.2
    stage2_rlr_min_lr: float = 1e-6