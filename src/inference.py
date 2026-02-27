"""
Inference script for animal classification
Usage:
    python -m src.inference --img "path/to/image.jpg"
"""

import torch
import timm
import yaml
import argparse
from torchvision import transforms
from PIL import Image
from src.model import get_model

# Load config
with open("config.yaml", "r") as f:
    cfg = yaml.safe_load(f)

IMG_SIZE = cfg["dataset"]["img_size"]
MODEL_DIR = cfg["paths"]["models"]
MODEL_PATH = f"{MODEL_DIR}/best_model.pth"

# Same normalization used in training
val_transform = transforms.Compose([
    transforms.Resize(int(IMG_SIZE * 1.1)),
    transforms.CenterCrop(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

def predict(img_path):
    # Load model
    model = get_model()
    state_dict = torch.load(MODEL_PATH, map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()

    # Load and preprocess image
    image = Image.open(img_path).convert("RGB")
    image = val_transform(image).unsqueeze(0)  # add batch dim

    # Predict
    with torch.no_grad():
        outputs = model(image)
        probs = torch.softmax(outputs, dim=1)[0]

    # Get class names from dataset
    from torchvision import datasets
    dataset = datasets.ImageFolder(root=cfg["dataset"]["path"])
    idx_to_class = {v: k for k, v in dataset.class_to_idx.items()}

    # Top-1 prediction
    top1_prob, top1_idx = torch.max(probs, dim=0)
    print(f"Predicted: {idx_to_class[top1_idx.item()]} ({top1_prob.item()*100:.2f}%)")

    # Top-3 predictions
    top3_prob, top3_idx = torch.topk(probs, 3)
    print("\nTop-3 Predictions:")
    for i in range(3):
        cls = idx_to_class[top3_idx[i].item()]
        prob = top3_prob[i].item() * 100
        print(f"{cls}: {prob:.2f}%")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--img", type=str, required=True, help="Path to image")
    args = parser.parse_args()
    predict(args.img)
