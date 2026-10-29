"""Shared plotting helper for the training scripts."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

import config


def save_confusion_matrix(cm, class_names, path, title):
    short = [config.pretty_name(c) for c in class_names]
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Greens", xticklabels=short, yticklabels=short)
    plt.title(f"{title} - confusion matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.xticks(rotation=60, ha="right")
    plt.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=150)
    plt.close()
