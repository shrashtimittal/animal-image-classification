"""
plot_confusion.py

Generates a confusion matrix (and classification report) for the validation set
using the best saved model. Saves:
 - outputs/confusion_matrix.png
 - outputs/classification_report.txt
"""

import os
import yaml
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

# Import project utilities / model / dataloaders
from src.model import get_model
from src.data_module import get_dataloaders

# Load config
with open("config.yaml", "r") as f:
    cfg = yaml.safe_load(f)

MODEL_DIR = cfg["paths"]["models"]
MODEL_PATH_CANDIDATES = [
    os.path.join(MODEL_DIR, "best_model.pth"),
    os.path.join(MODEL_DIR, "best_model_phase1.pth"),
]
MODEL_PATH = None
for p in MODEL_PATH_CANDIDATES:
    if os.path.exists(p):
        MODEL_PATH = p
        break

if MODEL_PATH is None:
    raise FileNotFoundError(f"No model file found in {MODEL_DIR}. Expected one of: {MODEL_PATH_CANDIDATES}")

IMG_SIZE = cfg["dataset"]["img_size"]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
os.makedirs("outputs", exist_ok=True)


def evaluate_and_plot():
    # Get dataloaders (train, val, classes)
    train_loader, val_loader, classes = get_dataloaders()
    num_classes = len(classes)
    print(f"Found {num_classes} classes: {classes}")

    # Load model and weights
    model = get_model().to(device)
    state = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(state)
    model.eval()
    print(f"Loaded model weights from: {MODEL_PATH}")

    # Collect predictions and labels
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            preds = outputs.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    # Confusion matrix (raw counts)
    cm = confusion_matrix(all_labels, all_preds, labels=list(range(num_classes)))

    # Plot confusion matrix heatmap
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=classes, yticklabels=classes)
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.title("Confusion Matrix (validation set) — counts")
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    cm_path = "outputs/confusion_matrix.png"
    plt.savefig(cm_path, dpi=200)
    plt.show()
    print(f"Saved confusion matrix image to: {cm_path}")

    # Also save a normalized confusion matrix (rows sum to 1)
    cm_norm = cm.astype("float") / (cm.sum(axis=1)[:, np.newaxis] + 1e-12)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Blues",
                xticklabels=classes, yticklabels=classes)
    plt.xlabel("Predicted label (normalized)")
    plt.ylabel("True label")
    plt.title("Confusion Matrix (validation set) — normalized")
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    cm_norm_path = "outputs/confusion_matrix_normalized.png"
    plt.savefig(cm_norm_path, dpi=200)
    plt.show()
    print(f"Saved normalized confusion matrix image to: {cm_norm_path}")

    # Classification report (precision/recall/f1)
    report = classification_report(all_labels, all_preds, target_names=classes, digits=4)
    report_path = "outputs/classification_report.txt"
    with open(report_path, "w") as fh:
        fh.write(report)
    print(f"Saved classification report to: {report_path}")
    print("\nClassification report:\n")
    print(report)


if __name__ == "__main__":
    evaluate_and_plot()
