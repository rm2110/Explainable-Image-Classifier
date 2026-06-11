import streamlit as st
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
import timm
import numpy as np
import requests
from PIL import Image

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

# -----------------------------
# Config
# -----------------------------
classes = ['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -----------------------------
# Load ViT model
# -----------------------------
@st.cache_resource
def load_vit():
    model = timm.create_model('vit_tiny_patch16_224', pretrained=False)
    model.head = nn.Linear(model.head.in_features, 10)
    model.load_state_dict(torch.load("vit_model.pth", map_location=device))
    model = model.to(device)
    model.eval()
    return model

# -----------------------------
# Load ResNet model
# -----------------------------
@st.cache_resource
def load_resnet():
    model = models.resnet18(pretrained=False)
    model.fc = nn.Linear(model.fc.in_features, 10)
    model.load_state_dict(torch.load("resnet_model.pth", map_location=device))
    model = model.to(device)
    model.eval()
    return model

vit_model = load_vit()
resnet_model = load_resnet()

# -----------------------------
# Transforms
# -----------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])
])

# -----------------------------
# Prediction
# -----------------------------
def predict(image_tensor):
    with torch.no_grad():
        outputs = vit_model(image_tensor.to(device))
        probs = torch.softmax(outputs, dim=1)

        pred_class = torch.argmax(probs, dim=1).item()
        confidence = probs[0][pred_class].item()

    return pred_class, confidence, outputs

# -----------------------------
# Top-3 predictions
# -----------------------------
def get_top3(outputs):
    probs = torch.softmax(outputs, dim=1)
    top3 = torch.topk(probs, 3)

    result = []
    for i in range(3):
        cls = top3.indices[0][i].item()
        conf = top3.values[0][i].item()
        result.append(f"{classes[cls]} ({conf:.2f})")

    return result

# -----------------------------
# GradCAM heatmap
# -----------------------------
def generate_heatmap(image_tensor, pred_class):
    target_layer = resnet_model.layer4[-1]
    cam = GradCAM(model=resnet_model, target_layers=[target_layer])

    grayscale_cam = cam(
        input_tensor=image_tensor.to(device),
        targets=[ClassifierOutputTarget(pred_class)]
    )[0]

    img = image_tensor.squeeze().permute(1,2,0).cpu().numpy()
    img = (img - img.min()) / (img.max() - img.min())

    heatmap = show_cam_on_image(img, grayscale_cam, use_rgb=True)
    return heatmap

# -----------------------------
# LLM (Ollama)
# -----------------------------
def generate_llm_explanation(pred_class, confidence, top3):
    label = classes[pred_class]

    prompt = f"""
You are an AI explaining an image classification result.

Prediction: {label}
Confidence: {confidence:.2f}
Top predictions: {", ".join(top3)}

The model focused on the main object region.

Explain clearly why this prediction was made and what visual features were used.
Keep it concise and natural.
"""

    response = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model": "llama3:8b",
            "prompt": prompt,
            "stream": False
        }
    )

    result = response.json()
    return result.get("response", "LLM error")

# -----------------------------
# UI
# -----------------------------
st.title("Explainable AI Image Classifier")
st.write("Upload an image to get prediction, heatmap, and explanation.")

uploaded_file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    image_tensor = transform(image).unsqueeze(0)

    # Prediction
    pred_class, confidence, outputs = predict(image_tensor)
    top3 = get_top3(outputs)

    st.subheader("Prediction")
    st.write(f"Class: **{classes[pred_class]}**")
    st.write(f"Confidence: **{confidence:.2f}**")
    st.write(f"Top-3: {top3}")

    # Heatmap
    st.subheader("Heatmap (GradCAM)")
    heatmap = generate_heatmap(image_tensor, pred_class)
    st.image(heatmap, use_column_width=True)

    # Explanation
    st.subheader("Explanation (LLM)")
    explanation = generate_llm_explanation(pred_class, confidence, top3)
    st.write(explanation)