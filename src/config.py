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
    genre: str
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
