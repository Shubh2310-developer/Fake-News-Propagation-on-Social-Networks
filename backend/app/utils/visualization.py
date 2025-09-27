# backend/app/utils/visualization.py

import io
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List
from sklearn.metrics import confusion_matrix


def plot_confusion_matrix(y_true: List[int], y_pred: List[int]) -> bytes:
    """Generate confusion matrix plot and return as PNG bytes."""
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix")

    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    plt.close(fig)
    buf.seek(0)
    return buf.read()


def plot_training_curves(history: dict) -> bytes:
    """Plot training and validation loss/accuracy curves."""
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))

    # Loss
    ax[0].plot(history.get("train_loss", []), label="Train Loss")
    ax[0].plot(history.get("val_loss", []), label="Val Loss")
    ax[0].set_title("Loss Curve")
    ax[0].legend()

    # Accuracy
    ax[1].plot(history.get("train_acc", []), label="Train Acc")
    ax[1].plot(history.get("val_acc", []), label="Val Acc")
    ax[1].set_title("Accuracy Curve")
    ax[1].legend()

    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    plt.close(fig)
    buf.seek(0)
    return buf.read()