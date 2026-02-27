# 🐾 Animal Image Classification System

## 📌 Project Overview
A deep learning-based multi-class image classification system built using PyTorch and EfficientNet-B0.

The model classifies images into 15 animal categories including:
Bear, Bird, Cat, Cow, Deer, Dog, Dolphin, Elephant, Giraffe, Horse, Kangaroo, Lion, Panda, Tiger, Zebra.

All images are resized to 224x224x3 for training.

---

## 🎯 Objective
To build an accurate and scalable deep learning model capable of identifying animal species from images using transfer learning.

---

## 🧠 Model Architecture
- Transfer Learning using **EfficientNet-B0**
- Pretrained on ImageNet
- Custom classification head for 15 classes
- Fine-tuning after initial frozen training phase

---

## 🗂 Dataset
- 15 classes
- 224x224 RGB images
- Folder-based structure

Dataset is not included in this repository.

Place the dataset inside a folder named:

dataset/
 ├── Bear/
 ├── Bird/
 ├── Cat/
 ...
 └── Zebra/

Update the dataset path inside `config.yaml` if needed.

---

## 🛠 Tech Stack
- Python
- PyTorch
- Torchvision
- TIMM (for EfficientNet)
- NumPy
- OpenCV
- Streamlit (for deployment interface)

---

## 📊 Results
- Training Accuracy: (add your real number)
- Validation Accuracy: (add your real number)
- Model checkpoint saved as: models/best_model.pth

---

## 🚀 How to Run

1️⃣ Clone the repository:
git clone https://github.com/shrashtimittal/animal-image-classification.git

cd animal-image-classification

2️⃣ Install dependencies: pip install -r requirements.txt

3️⃣ Install PyTorch separately (based on your system):
Visit: https://pytorch.org/get-started/locally/

4️⃣ Run the application: streamlit run app.py

---

## 📌 Project Structure

animal-image-classification/
├── src/              # Model architecture and training logic
├── models/           # Saved model weights
├── app.py            # Streamlit app
├── config.yaml       # Configuration file
├── requirements.txt
└── README.md

---

## 🔮 Future Improvements
- Hyperparameter tuning
- Model optimization
- Deployment on cloud
- Adding more animal classes


