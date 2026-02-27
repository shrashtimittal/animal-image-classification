import streamlit as st
import torch
import yaml
import timm
from torchvision import transforms, datasets
from PIL import Image
from src.model import get_model

# Load config
with open("config.yaml", "r") as f:
    cfg = yaml.safe_load(f)

IMG_SIZE = cfg["dataset"]["img_size"]
MODEL_PATH = f"{cfg['paths']['models']}/best_model.pth"

# Preprocessing (same as validation)
val_transform = transforms.Compose([
    transforms.Resize(int(IMG_SIZE * 1.1)),
    transforms.CenterCrop(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# Load dataset class names
dataset = datasets.ImageFolder(root=cfg["dataset"]["path"])
idx_to_class = {v: k for k, v in dataset.class_to_idx.items()}

# Load trained model
@st.cache_resource
def load_model():
    model = get_model()
    state_dict = torch.load(MODEL_PATH, map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()
    return model

model = load_model()

# Streamlit UI
st.title("🐾 Animal Image Classification")
st.write("Upload an image and the system will identify the animal.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Show uploaded image
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess
    input_tensor = val_transform(image).unsqueeze(0)

    # Predict
    with torch.no_grad():
        outputs = model(input_tensor)
        probs = torch.softmax(outputs, dim=1)[0]

    # Top-3 predictions
    top3_prob, top3_idx = torch.topk(probs, 3)

    st.subheader("Predictions")
    for i in range(3):
        cls = idx_to_class[top3_idx[i].item()]
        prob = top3_prob[i].item() * 100
        st.write(f"**{cls}**: {prob:.2f}%")
