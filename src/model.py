"""
EfficientNet Model for Animal Classification
- loads pretrained weights (if available)
- helper functions to freeze/unfreeze backbone
- graceful fallback: if pretrained download fails, uses pretrained=False
"""

import yaml
import timm
import torch
import torch.nn as nn

with open("config.yaml", "r") as f:
    cfg = yaml.safe_load(f)

NUM_CLASSES = cfg["dataset"]["num_classes"]
MODEL_NAME = cfg.get("model", {}).get("name", "efficientnet_b0")
MODEL_PRETRAINED = cfg.get("model", {}).get("pretrained", True)

def get_model(model_name=MODEL_NAME, pretrained=MODEL_PRETRAINED, num_classes=NUM_CLASSES):
    """
    Returns a timm model with its classifier set to `num_classes`.
    If pretrained weights can't be downloaded, falls back to pretrained=False.
    """
    try:
        model = timm.create_model(model_name, pretrained=pretrained, num_classes=num_classes)
        print(f"Loaded {model_name} with pretrained={pretrained}")
    except Exception as e:
        print(f"Warning: failed to load pretrained weights ({e}). Falling back to pretrained=False")
        model = timm.create_model(model_name, pretrained=False, num_classes=num_classes)
    return model

def freeze_backbone(model):
    """Freeze all params except classifier head."""
    for param in model.parameters():
        param.requires_grad = False
    # Unfreeze classifier parameters
    classifier = model.get_classifier()
    for param in classifier.parameters():
        param.requires_grad = True
    print("Backbone frozen — only classifier head will be trained.")

def unfreeze_all(model):
    """Unfreeze all parameters for fine-tuning."""
    for param in model.parameters():
        param.requires_grad = True
    print("All model parameters are now trainable (unfrozen).")

if __name__ == "__main__":
    # quick smoke test
    m = get_model()
    x = torch.randn(2, 3, 224, 224)
    out = m(x)
    print("Model output shape:", out.shape)  # should be [2, NUM_CLASSES]
