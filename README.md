# Explainable Vision Transformer with GenAI Interpretation

## Overview

This project presents an end-to-end Explainable AI (XAI) image classification system that combines Deep Learning, Explainable AI, and Generative AI into a single pipeline.

The system uses a **Vision Transformer (ViT)** to classify images, **GradCAM** to visualize the regions influencing model decisions, and **Llama3 (via Ollama)** to generate human-readable explanations of predictions. The entire solution is deployed through a **Streamlit web application**, allowing users to upload images and receive predictions, visual explanations, and textual interpretations in real time.

---

## Features

* Vision Transformer (ViT) based image classification
* Explainable AI using GradCAM heatmaps
* Large Language Model (Llama3) generated explanations
* Top-3 prediction analysis
* Interactive Streamlit web interface
* End-to-end AI pipeline from prediction to interpretation
* Local LLM execution using Ollama (no cloud API required)

---

## Project Architecture

```text
Input Image
      │
      ▼
Image Preprocessing
(Resize + Normalize)
      │
      ▼
Vision Transformer (ViT)
      │
      ├──► Prediction
      │
      ├──► Confidence Score
      │
      └──► Top-3 Predictions
      │
      ▼
GradCAM (ResNet)
      │
      ▼
Attention Heatmap
      │
      ▼
Llama3 (Ollama)
      │
      ▼
Natural Language Explanation
      │
      ▼
Streamlit Web Interface
```

---

## Dataset

### CIFAR-10

The project uses the CIFAR-10 dataset containing 60,000 images across 10 classes:

* Airplane
* Automobile
* Bird
* Cat
* Deer
* Dog
* Frog
* Horse
* Ship
* Truck

Dataset Characteristics:

* Total Images: 60,000
* Training Images: 50,000
* Testing Images: 10,000
* Original Image Size: 32×32
* Resized Input Size: 224×224

---

## Technologies Used

### Deep Learning

* PyTorch
* Vision Transformer (ViT)
* Transfer Learning
* CNN (ResNet18)

### Explainable AI

* GradCAM
* Model Interpretability

### Generative AI

* Ollama
* Llama3 8B
* Prompt Engineering

### Deployment

* Streamlit

### Supporting Libraries

* timm
* torchvision
* NumPy
* Matplotlib
* Pillow
* OpenCV
* requests

---

## Methodology

### 1. Image Classification using Vision Transformer

The uploaded image is resized to 224×224 and processed by a Vision Transformer model.

Unlike CNNs, ViT divides an image into patches and treats them similarly to tokens in Natural Language Processing.

Steps:

1. Split image into patches
2. Convert patches into embeddings
3. Add positional encodings
4. Process through Transformer Encoder
5. Generate class prediction

Advantages:

* Captures global image relationships
* Strong performance on image understanding tasks
* Modern alternative to CNN-based classification

---

### 2. Explainability using GradCAM

Deep learning models often act as black boxes.

To improve transparency, GradCAM is used to visualize the regions responsible for predictions.

Process:

1. Compute gradients of target class
2. Extract feature maps
3. Generate weighted activation maps
4. Overlay heatmap on original image

Output:

* Red regions indicate high importance
* Blue regions indicate low importance

---

### 3. Generative AI Interpretation

Prediction outputs are sent to a Large Language Model (Llama3).

Inputs to the LLM:

* Predicted class
* Confidence score
* Top-3 predictions

The LLM generates a natural language explanation describing:

* Why the prediction was made
* Which visual features were important
* Why the model is confident

---

## Web Application

The project includes a Streamlit application that enables users to:

* Upload images
* View predictions
* Analyze GradCAM heatmaps
* Read AI-generated explanations

### User Workflow

1. Upload image
2. Model predicts class
3. Heatmap is generated
4. LLM explanation is generated
5. Results displayed on dashboard

---

## Installation

### Clone Repository

```bash
git clone https://github.com/yourusername/explainable-vit-genai.git
cd explainable-vit-genai
```

### Create Environment

```bash
python -m venv dl_env
```

### Activate Environment

Windows:

```bash
dl_env\Scripts\activate
```

Linux/Mac:

```bash
source dl_env/bin/activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Install Ollama

Download Ollama:

https://ollama.com

Pull the Llama3 model:

```bash
ollama run llama3:8b
```

Keep Ollama running while using the application.

---

## Running the Application

```bash
python -m streamlit run app.py
```

The application will be available at:

```text
http://localhost:8501
```

---

## Results

The system successfully demonstrates:

* High classification performance on CIFAR-10
* Effective visual explanations through GradCAM
* Human-readable AI-generated explanations
* End-to-end deployment through Streamlit

Sample Outputs:

* Predicted Class
* Confidence Score
* Top-3 Predictions
* GradCAM Heatmap
* LLM Explanation

---

## Future Improvements

* Direct ViT Attention Visualization
* Support for custom datasets
* Cloud deployment
* Retrieval-Augmented Generation (RAG)
* Multi-modal reasoning
* Support for larger Vision Transformer models

---

## Learning Outcomes

Through this project, the following concepts were explored:

* Vision Transformers (ViT)
* Self-Attention Mechanisms
* Transfer Learning
* Explainable AI (XAI)
* GradCAM
* Large Language Models
* Prompt Engineering
* Streamlit Deployment
* End-to-End AI System Design

---

## Authors

* Rishi J Mahajan


This project is intended for educational and research purposes.
