from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.inspection import permutation_importance


sns.set_theme(style="whitegrid")


def save_eda_plots(data, output_dir: str | Path) -> list[Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    saved_files: list[Path] = []

    plt.figure(figsize=(10, 6))
    sns.histplot(data["final_score"], bins=25, kde=True, color="#1f77b4")
    plt.title("Distribution of Final Scores")
    plt.xlabel("Final Score")
    plt.ylabel("Count")
    path = output_dir / "final_score_distribution.png"
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()
    saved_files.append(path)

    plt.figure(figsize=(10, 7))
    correlation = data.select_dtypes(include="number").corr()
    sns.heatmap(correlation, cmap="coolwarm", center=0, linewidths=0.2)
    plt.title("Numeric Feature Correlation Heatmap")
    path = output_dir / "correlation_heatmap.png"
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()
    saved_files.append(path)

    plt.figure(figsize=(9, 5))
    sns.countplot(data=data, x="performance_level", order=["Low", "Medium", "High"], color="#5b8def")
    plt.title("Performance Level Distribution")
    plt.xlabel("Performance Level")
    plt.ylabel("Students")
    path = output_dir / "performance_level_distribution.png"
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()
    saved_files.append(path)

    return saved_files


def save_model_plots(model, X_test, y_test, predictions, output_dir: str | Path) -> list[Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    saved_files: list[Path] = []

    plt.figure(figsize=(8, 8))
    sns.scatterplot(x=y_test, y=predictions, color="#2ca02c", alpha=0.7)
    min_value = min(y_test.min(), predictions.min())
    max_value = max(y_test.max(), predictions.max())
    plt.plot([min_value, max_value], [min_value, max_value], "r--", linewidth=2)
    plt.title("Actual vs Predicted Final Score")
    plt.xlabel("Actual Score")
    plt.ylabel("Predicted Score")
    path = output_dir / "actual_vs_predicted.png"
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()
    saved_files.append(path)

    residuals = y_test - predictions
    plt.figure(figsize=(10, 5))
    sns.histplot(residuals, bins=25, kde=True, color="#ff7f0e")
    plt.title("Residual Distribution")
    plt.xlabel("Actual - Predicted")
    plt.ylabel("Count")
    path = output_dir / "residual_distribution.png"
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()
    saved_files.append(path)

    try:
        importance = permutation_importance(model, X_test, y_test, n_repeats=5, random_state=42, n_jobs=-1)
        feature_names = X_test.columns
        sorted_indices = importance.importances_mean.argsort()[-10:]
        plt.figure(figsize=(10, 6))
        sns.barplot(
            x=importance.importances_mean[sorted_indices],
            y=feature_names[sorted_indices],
            color="#8e44ad",
        )
        plt.title("Top Feature Importance by Permutation")
        plt.xlabel("Mean Importance")
        plt.ylabel("Feature")
        path = output_dir / "feature_importance.png"
        plt.tight_layout()
        plt.savefig(path, dpi=160)
        plt.close()
        saved_files.append(path)
    except Exception:
        pass

    return saved_files
