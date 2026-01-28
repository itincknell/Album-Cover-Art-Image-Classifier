"""
eval.py

Evaluation utilities for album-cover classification models.

This module is intentionally model-agnostic:
given true/predicted class indices and class names, it writes common evaluation
artifacts (classification report, confusion matrix, summary).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)


def save_classification_report(
    y_true_int: np.ndarray,
    y_pred_int: np.ndarray,
    class_names: List[str],
    out_csv: Path,
) -> Dict:
    """
    Write a sklearn classification report.

    Returns the report dict (same structure as sklearn's output_dict=True).
    """
    report = classification_report(
        y_true_int,
        y_pred_int,
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )
    pd.DataFrame(report).transpose().to_csv(out_csv, index=True)

    return report


def save_confusion_matrix_png(
    y_true_int: np.ndarray,
    y_pred_int: np.ndarray,
    class_names: List[str],
    out_png: Path,
    title: str,
    normalize: Optional[str] = None,  # None | "true" | "pred" | "all" (sklearn semantics)
) -> np.ndarray:
    """
    Save a confusion matrix heatmap as a PNG.

    If normalize is provided, the plot shows normalized values but the returned
    matrix is the plotted matrix (normalized or raw).
    """
    cm = confusion_matrix(y_true_int, y_pred_int, normalize=normalize)

    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt=".2f" if normalize else "d",
        xticklabels=class_names,
        yticklabels=class_names,
    )
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(out_png)
    plt.close()

    return cm


def save_confusion_matrix_csv(
    y_true_int: np.ndarray,
    y_pred_int: np.ndarray,
    class_names: List[str],
    out_csv: Path,
    normalize: Optional[str] = None,
) -> pd.DataFrame:
    """
    Save the (optionally normalized) confusion matrix as a CSV with labels.
    """
    cm = confusion_matrix(y_true_int, y_pred_int, normalize=normalize)
    df = pd.DataFrame(cm, index=class_names, columns=class_names)
    df.to_csv(out_csv, index=True)
    return df


def save_summary_json(
    y_true_int: np.ndarray,
    y_pred_int: np.ndarray,
    report: Optional[Dict],
    out_json: Path,
) -> Dict:
    """
    Save a small headline-metrics summary. Designed to be easy to scan in a repo.
    """
    summary = {
        "n_examples": int(len(y_true_int)),
        "accuracy": float(accuracy_score(y_true_int, y_pred_int)),
        "f1_macro": float(f1_score(y_true_int, y_pred_int, average="macro")),
        "f1_weighted": float(f1_score(y_true_int, y_pred_int, average="weighted")),
    }

    if report is not None:
        # Pull common aggregates if present
        if "macro avg" in report:
            summary["precision_macro"] = float(report["macro avg"]["precision"])
            summary["recall_macro"] = float(report["macro avg"]["recall"])
        if "weighted avg" in report:
            summary["precision_weighted"] = float(report["weighted avg"]["precision"])
            summary["recall_weighted"] = float(report["weighted avg"]["recall"])

    out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def evaluate_and_save(
    y_true_int: np.ndarray,
    y_pred_int: np.ndarray,
    class_names: List[str],
    metrics_dir: Path,
    tag: str,
    task_name: str,
    meta,
) -> None:
    """
    One-call helper to write the standard artifact set:
    - classification_report.csv
    - confusion_matrix.png + .csv
    - summary.json

    `task_name` is used only for plot title clarity (e.g., "Genre", "Decade").
    """
    metrics_dir.mkdir(parents=True, exist_ok=True)

    report_csv = metrics_dir / f"{tag}_classification_report.csv"

    cm_png = metrics_dir / f"{tag}_confusion_matrix.png"
    cm_csv = metrics_dir / f"{tag}_confusion_matrix.csv"

    summary_json = metrics_dir / f"{tag}_summary.json" 

    meta_path = metrics_dir / f"{tag}_run_metadata.json"

    report = save_classification_report(
        y_true_int=y_true_int,
        y_pred_int=y_pred_int,
        class_names=class_names,
        out_csv=report_csv,
    )

    save_confusion_matrix_png(
        y_true_int=y_true_int,
        y_pred_int=y_pred_int,
        class_names=class_names,
        out_png=cm_png,
        title=f"Confusion Matrix - {task_name} ({tag})",
        normalize=None,
    )

    save_confusion_matrix_csv(
        y_true_int=y_true_int,
        y_pred_int=y_pred_int,
        class_names=class_names,
        out_csv=cm_csv,
        normalize=None,
    )

    save_summary_json(
        y_true_int=y_true_int,
        y_pred_int=y_pred_int,
        report=report,
        out_json=summary_json,
    )

    # update artifacts and save meta data
    meta["artifacts"].update({
    "classification_report": str(report_csv),
    "confusion_matrix_png": str(cm_png),
    "confusion_matrix_csv": str(cm_csv),
    "summary_json": str(summary_json),
    })
    meta_path.write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n", encoding="utf-8")
