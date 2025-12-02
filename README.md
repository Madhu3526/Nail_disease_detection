## Nail Disease Detection using Deep Learning

A deep-learning–based multi-class classification system that detects 6 nail disorders from images using a fine-tuned EfficientNet-B0 model.
The project includes:

    -> PyTorch training pipeline
    
    -> ONNX model export
    
    -> FastAPI backend for inference
    
    -> Gradio-based web UI
    
    -> GPU-accelerated inference support
    

# Project Overview

Early detection of nail abnormalities can help identify underlying medical conditions such as nutritional deficiencies, fungal infections, or cardiovascular issues. This project applies computer vision to automatically classify nail images into six categories using a state-of-the-art convolutional neural network.

The complete system includes model training, conversion to ONNX, and deployment through API + UI, making it suitable for real-world usage.

🩺 Classes Detected
1. Acral_Lentiginous_Melanoma
2. Healthy_Nail
3. Onychogryphosis
4. blue_finger
5. clubbing
6. pitting

# Model Architecture

Backbone: tf_efficientnet_b0_ns
Framework: PyTorch
Loss Function: Cross-Entropy
Optimizer: AdamW
Scheduler: Cosine Annealing
Image Size: 224 × 224

# Training Summary


✓ Model: tf_efficientnet_b0_ns
✓ Total epochs: 18
✓ Best validation F1-score: 0.9902
✓ Final validation accuracy: 0.9890
✓ Number of classes: 6
✓ Training samples: 3744
✓ Validation samples: 91

✓ Best model saved to: best_model.pth
✓ Final model saved to: final_model.pth

This high performance indicates:

Strong feature extraction from EfficientNet-B0
Good dataset curation and balancing
Effective augmentation strategy
No signs of overfitting (train/val difference small)

# ONNX Conversion

The trained PyTorch model was exported to ONNX for lightweight, hardware-accelerated inference.

Benefits:
 -> Faster inference
 -> Portable across platforms (Windows, Linux, Cloud)
 -> Compatible with FastAPI, mobile, and edge devices

## Deployment Architecture
Gradio UI  →  FastAPI Backend → ONNXRuntime → Prediction

Frontend (Gradio)
Clean image upload interface

Backend (FastAPI)
/predict endpoint
Accepts image input
Runs ONNX inference
Returns class + confidence score

Inference Engine

ONNX Runtime (CPU/GPU execution providers)

# How to Run Locally

1️⃣ Create virtual environment
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt

2️⃣ Start FastAPI backend
uvicorn app:app --reload

Backend runs at:

http://127.0.0.1:8000/predict

3️⃣ Launch the Gradio UI
python ui.py

You will get:
Local URL → http://0.0.0.0:7860
Public URL → generated automatically by Gradio

# Tech Stack
Component	Technology
Model	PyTorch + EfficientNet-B0
Deployment	FastAPI
UI	Gradio
Inference	ONNX Runtime
Training	NVIDIA GPU
Dataset	Custom curated nail disease dataset

# Results
Metric	Score
Validation Accuracy	98.90%
Best Validation F1-Score	0.9902
Number of Classes	6
Epochs	18

The model generalizes extremely well with no significant overfitting.
