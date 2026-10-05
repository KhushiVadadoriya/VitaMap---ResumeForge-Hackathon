"""
Model evaluation, metrics computation, confusion matrix plotting, and error analysis.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)


def compute_metrics(y_true: Any, y_pred: Any) -> Dict[str, float]:
    """
    Compute comprehensive classification metrics.
    Macro-F1 is the primary evaluation metric.
    """
    acc = accuracy_score(y_true, y_pred)
    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    weighted_p, weighted_r, weighted_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )

    return {
        "accuracy": float(acc),
        "macro_precision": float(macro_p),
        "macro_recall": float(macro_r),
        "macro_f1": float(macro_f1),
        "weighted_precision": float(weighted_p),
        "weighted_recall": float(weighted_r),
        "weighted_f1": float(weighted_f1),
    }


def compute_classification_report_df(y_true: Any, y_pred: Any) -> pd.DataFrame:
    """
    Generate classification report formatted as a pandas DataFrame.
    """
    report_dict = classification_report(
        y_true, y_pred, output_dict=True, zero_division=0
    )
    df_report = pd.DataFrame(report_dict).transpose()
    # Format support as integer
    if "support" in df_report.columns:
        df_report["support"] = df_report["support"].astype(int)
    return df_report


def plot_confusion_matrix(
    y_true: Any,
    y_pred: Any,
    classes: List[str],
    output_path: str,
    normalize: bool = True,
    title: str = "Confusion Matrix"
) -> None:
    """
    Plot and save confusion matrix with readable layout.
    """
    cm = confusion_matrix(y_true, y_pred, labels=classes)
    fmt = ".2f" if normalize else "d"

    if normalize:
        # Avoid division by zero
        row_sums = cm.sum(axis=1, keepdims=True)
        cm_display = np.divide(cm.astype("float"), row_sums, out=np.zeros_like(cm, dtype=float), where=row_sums != 0)
    else:
        cm_display = cm

    plt.figure(figsize=(16, 14))
    sns.heatmap(
        cm_display,
        annot=True,
        fmt=fmt,
        cmap="Blues",
        xticklabels=classes,
        yticklabels=classes,
        cbar=True,
        annot_kws={"size": 8}
    )
    plt.title(title, fontsize=14, pad=15, fontweight="bold")
    plt.xlabel("Predicted Category", fontsize=12, labelpad=10)
    plt.ylabel("Actual Category", fontsize=12, labelpad=10)
    plt.xticks(rotation=45, ha="right", fontsize=9)
    plt.yticks(rotation=0, fontsize=9)
    plt.tight_layout()

    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_file, dpi=300)
    plt.close()


def generate_error_analysis_df(
    df_subset: pd.DataFrame,
    y_true: pd.Series,
    y_pred: np.ndarray,
    pipeline: Any,
) -> pd.DataFrame:
    """
    Extract misclassified instances with actual vs predicted categories, text preview,
    and decision scores.
    """
    classifier = pipeline.named_steps["classifier"]
    classes = list(classifier.classes_)

    errors = []
    # Identify indices where prediction != actual
    is_error = (y_true.values != y_pred)
    error_indices = np.where(is_error)[0]

    # Precompute decision function or probabilities if available
    scores = None
    if hasattr(pipeline, "decision_function"):
        try:
            scores = pipeline.decision_function(df_subset["text"])
        except Exception:
            scores = None
    elif hasattr(pipeline, "predict_proba"):
        try:
            scores = pipeline.predict_proba(df_subset["text"])
        except Exception:
            scores = None

    for idx in error_indices:
        row = df_subset.iloc[idx]
        actual = str(y_true.iloc[idx])
        pred = str(y_pred[idx])
        raw_text = str(row["text"])
        preview = (raw_text[:200] + "...") if len(raw_text) > 200 else raw_text

        decision_info = ""
        if scores is not None:
            actual_idx = classes.index(actual) if actual in classes else -1
            pred_idx = classes.index(pred) if pred in classes else -1
            if actual_idx >= 0 and pred_idx >= 0:
                pred_score = float(scores[idx][pred_idx])
                actual_score = float(scores[idx][actual_idx])
                decision_info = f"predicted_score={pred_score:.3f}; actual_score={actual_score:.3f}"

        errors.append({
            "ID": row["ID"],
            "actual_category": actual,
            "predicted_category": pred,
            "text_length": len(raw_text),
            "text_preview": preview,
            "decision_score_if_available": decision_info,
        })

    return pd.DataFrame(errors)
