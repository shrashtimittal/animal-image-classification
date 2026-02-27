"""
Dataset Inspection & Cleaning Script
"""

import os
import yaml
import random
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image, UnidentifiedImageError
from collections import Counter

# 1. Load config
with open("config.yaml", "r") as f:
    cfg = yaml.safe_load(f)

DATASET_PATH = cfg["dataset"]["path"]

def get_class_counts(path):
    classes = sorted([d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))])
    counts = {}
    for c in classes:
        folder = os.path.join(path, c)
        files = [f for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f))]
        counts[c] = len(files)
    return counts

def plot_class_distribution(counts):
    plt.figure(figsize=(10, 5))
    plt.bar(counts.keys(), counts.values(), color="skyblue")
    plt.xticks(rotation=45)
    plt.ylabel("Number of images")
    plt.title("Class Distribution")
    plt.tight_layout()
    plt.show()

def check_images(path, classes):
    issues = []
    for cls in classes:
        folder = os.path.join(path, cls)
        files = os.listdir(folder)
        for f in files:
            img_path = os.path.join(folder, f)
            try:
                img = Image.open(img_path)
                img.verify()  # check file integrity
            except (UnidentifiedImageError, OSError) as e:
                issues.append((cls, img_path, str(e)))
    return issues

def show_random_samples(path, classes, n=9):
    plt.figure(figsize=(8, 8))
    for i in range(n):
        cls = random.choice(classes)
        folder = os.path.join(path, cls)
        files = os.listdir(folder)
        if not files:
            continue
        img_path = os.path.join(folder, random.choice(files))
        try:
            img = Image.open(img_path).convert("RGB")
            plt.subplot(3, 3, i+1)
            plt.imshow(img)
            plt.title(cls)
            plt.axis("off")
        except Exception as e:
            print(f"Error displaying {img_path}: {e}")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # class counts
    counts = get_class_counts(DATASET_PATH)
    print("Image counts per class:\n", counts)

    # save counts to CSV
    df = pd.DataFrame(list(counts.items()), columns=["Class", "Count"])
    df.to_csv("outputs/class_counts.csv", index=False)
    print("\nSaved class counts to outputs/class_counts.csv")

    # plot distribution
    plot_class_distribution(counts)

    # check for corrupt images
    issues = check_images(DATASET_PATH, counts.keys())
    if issues:
        print("\n⚠ Found issues with images:")
        for i in issues:
            print(i)
    else:
        print("\n✅ No corrupt images found.")

    # show random samples
    show_random_samples(DATASET_PATH, list(counts.keys()))
