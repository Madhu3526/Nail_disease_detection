# app.py
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import io
import numpy as np
import onnxruntime as ort

from preprocess import preprocess_image_pil

# ----- FastAPI app -----
app = FastAPI(title="Nail Disease Classifier API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ----- ONNX session -----
providers = []
if "CUDAExecutionProvider" in ort.get_available_providers():
    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
else:
    providers = ["CPUExecutionProvider"]

session = ort.InferenceSession(
    "nail_classifier.onnx",
    providers=providers
)

# same order you used in training (ImageFolder classes)
CLASSES = [
    "Acral_Lentiginous_Melanoma",
    "Healthy_Nail",
    "Onychogryphosis",
    "blue_finger",
    "clubbing",
    "pitting"
]

@app.get("/")
def root():
    return {"message": "Nail disease classifier ONNX API is running."}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # read file bytes
    image_bytes = await file.read()
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    # preprocess
    arr = preprocess_image_pil(img)          # (3,224,224)
    arr = np.expand_dims(arr, axis=0)        # (1,3,224,224)

    # run inference
    outputs = session.run(None, {"input": arr})[0]  # (1, num_classes)
    logits = outputs[0]
    exp = np.exp(logits - np.max(logits))
    probs = exp / exp.sum()

    pred_idx = int(np.argmax(probs))
    pred_class = CLASSES[pred_idx]
    confidence = float(probs[pred_idx])

    # optional: return full probability per class
    class_probs = {cls: float(p) for cls, p in zip(CLASSES, probs)}

    return {
        "predicted_class": pred_class,
        "confidence": confidence,
        "class_probabilities": class_probs,
        "provider": session.get_providers()
    }
