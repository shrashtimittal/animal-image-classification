"""
PyTorch Data Module (stratified split + stronger augmentations)
"""

import os
import yaml
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split

with open("config.yaml", "r") as f:
    cfg = yaml.safe_load(f)

DATASET_PATH = cfg["dataset"]["path"]
IMG_SIZE = cfg["dataset"]["img_size"]
BATCH_SIZE = cfg["training"]["batch_size"]
VAL_SPLIT = 0.2
NUM_WORKERS = 2

# Transforms
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(IMG_SIZE, scale=(0.8, 1.0)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
    transforms.RandomErasing(p=0.2, scale=(0.02, 0.2)),
])

val_transform = transforms.Compose([
    transforms.Resize(int(IMG_SIZE * 1.1)),
    transforms.CenterCrop(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

def get_dataloaders(val_split=VAL_SPLIT):
    # 1) create a dataset (no transform) to access labels
    full_dataset = datasets.ImageFolder(root=DATASET_PATH)
    targets = full_dataset.targets  # list of class indices, same order used by ImageFolder

    # 2) stratified split indices
    idx = list(range(len(targets)))
    train_idx, val_idx = train_test_split(idx, test_size=val_split, stratify=targets, random_state=42)

    # 3) create two ImageFolder datasets with different transforms and wrap in Subset
    train_dataset = Subset(datasets.ImageFolder(root=DATASET_PATH, transform=train_transform), train_idx)
    val_dataset = Subset(datasets.ImageFolder(root=DATASET_PATH, transform=val_transform), val_idx)

    # 4) DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=False)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=False)

    classes = full_dataset.classes
    print(f"DataLoaders ready — Train samples: {len(train_idx)}, Val samples: {len(val_idx)}")
    return train_loader, val_loader, classes

if __name__ == "__main__":
    tr, va, cls = get_dataloaders()
    print("Classes:", cls)
    # peek one batch
    batch = next(iter(tr))
    images, labels = batch
    print("Batch shapes:", images.shape, labels.shape)
