"""
Train script with two-phase transfer learning:
1) Freeze backbone, train head for freeze_epochs
2) Unfreeze all and fine-tune for remaining epochs (with lower LR)
"""

import os
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import matplotlib.pyplot as plt   # ✅ for plotting

from src.data_module import get_dataloaders
from src.model import get_model, freeze_backbone, unfreeze_all

# ✅ Lists for storing metrics
train_losses, val_losses = [], []
train_accuracies, val_accuracies = [], []

# Load config
with open("config.yaml", "r") as f:
    cfg = yaml.safe_load(f)

LR = float(cfg["training"]["learning_rate"])
FREEZE_EPOCHS = int(cfg["training"].get("freeze_epochs", 0))
FINE_TUNE_LR = float(cfg["training"].get("fine_tune_lr", LR * 0.1))
EPOCHS = int(cfg["training"]["epochs"])
BATCH_SIZE = int(cfg["training"]["batch_size"])
MODEL_DIR = cfg["paths"]["models"]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


def train_one_epoch(model, dataloader, criterion, optimizer):
    model.train()
    running_loss = 0.0
    correct, total = 0, 0
    for images, labels in tqdm(dataloader, desc="Training", leave=False):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return running_loss / total, correct / total


def validate(model, dataloader, criterion):
    model.eval()
    running_loss = 0.0
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Validating", leave=False):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    return running_loss / total, correct / total


def save_model(model, path):
    torch.save(model.state_dict(), path)
    print(f"Saved model: {path}")


def run_training():
    train_loader, val_loader, classes = get_dataloaders()

    # Build model
    model = get_model().to(device)

    # Loss
    criterion = nn.CrossEntropyLoss()

    best_acc = 0.0
    os.makedirs(MODEL_DIR, exist_ok=True)

    # ---- Phase 1: freeze backbone, train head only ----
    if FREEZE_EPOCHS > 0:
        freeze_backbone(model)
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=LR)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

        for epoch in range(1, FREEZE_EPOCHS + 1):
            print(f"\nPhase1 Epoch {epoch}/{FREEZE_EPOCHS} (head only)")
            train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer)
            val_loss, val_acc = validate(model, val_loader, criterion)

            # ✅ Store metrics
            train_losses.append(train_loss)
            train_accuracies.append(train_acc)
            val_losses.append(val_loss)
            val_accuracies.append(val_acc)

            print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
            print(f"Val   Loss: {val_loss:.4f}, Val   Acc: {val_acc:.4f}")

            if val_acc > best_acc:
                best_acc = val_acc
                save_model(model, os.path.join(MODEL_DIR, "best_model_phase1.pth"))

            scheduler.step()

    # ---- Phase 2: unfreeze & fine-tune ----
    remaining_epochs = EPOCHS - FREEZE_EPOCHS
    if remaining_epochs > 0:
        unfreeze_all(model)
        optimizer = optim.Adam(model.parameters(), lr=FINE_TUNE_LR)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

        for epoch in range(1, remaining_epochs + 1):
            ep_num = FREEZE_EPOCHS + epoch
            print(f"\nPhase2 Epoch {ep_num}/{EPOCHS} (fine-tune)")
            train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer)
            val_loss, val_acc = validate(model, val_loader, criterion)

            # ✅ Store metrics
            train_losses.append(train_loss)
            train_accuracies.append(train_acc)
            val_losses.append(val_loss)
            val_accuracies.append(val_acc)

            print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
            print(f"Val   Loss: {val_loss:.4f}, Val   Acc: {val_acc:.4f}")

            if val_acc > best_acc:
                best_acc = val_acc
                save_model(model, os.path.join(MODEL_DIR, "best_model.pth"))

            scheduler.step()

    print("\nTraining complete!")
    print(f"Best Validation Accuracy: {best_acc:.4f}")

    # ✅ Save plots after training
    epochs = range(1, len(train_losses) + 1)

    plt.figure(figsize=(8, 6))
    plt.plot(epochs, train_accuracies, label="Train Accuracy")
    plt.plot(epochs, val_accuracies, label="Validation Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.title("Training vs Validation Accuracy")
    plt.legend()
    plt.grid(True)
    plt.savefig("outputs/accuracy_curve.png")
    plt.close()

    plt.figure(figsize=(8, 6))
    plt.plot(epochs, train_losses, label="Train Loss")
    plt.plot(epochs, val_losses, label="Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training vs Validation Loss")
    plt.legend()
    plt.grid(True)
    plt.savefig("outputs/loss_curve.png")
    plt.close()

    print("✅ Saved plots: outputs/accuracy_curve.png and outputs/loss_curve.png")


if __name__ == "__main__":
    run_training()
