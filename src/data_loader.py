"""
Simple dataset sanity-check:
- Reads config.yaml for path
- Lists classes (folders) and counts images per class
- Shows a few random images (requires matplotlib)
"""

import os
import random
import yaml
from PIL import Image
import matplotlib.pyplot as plt

# load config
with open("config.yaml", "r") as f:
    cfg = yaml.safe_load(f)

DATASET_PATH = cfg["dataset"]["path"]

def list_classes_and_counts(path):
    classes = sorted([d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))])
    print(f"Found {len(classes)} classes:\n{classes}\n")
    counts = {}
    for c in classes:
        folder = os.path.join(path, c)
        files = [f for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f))]
        counts[c] = len(files)
    print("Image counts per class:")
    for k, v in counts.items():
        print(f"  {k}: {v}")
    return classes, counts

def show_random_images(path, classes, n=9):
    plt.figure(figsize=(8, 8))
    chosen = 0
    for i in range(n):
        cls = random.choice(classes)
        folder = os.path.join(path, cls)
        files = [f for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f))]
        if not files:
            continue
        fname = random.choice(files)
        img_path = os.path.join(folder, fname)
        try:
            img = Image.open(img_path).convert("RGB")
            plt.subplot(3, 3, i+1)
            plt.imshow(img)
            plt.title(cls)
            plt.axis("off")
        except Exception as e:
            print(f"Could not open {img_path}: {e}")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    classes, counts = list_classes_and_counts(DATASET_PATH)
    # If you have many classes with 0 images, check them manually.
    show_random_images(DATASET_PATH, classes, n=9)
