# ui.py
import gradio as gr
import requests
from PIL import Image

API_URL = "http://127.0.0.1:8000/predict"

def predict_ui(img: Image.Image):
    img.save("temp_upload.jpg")
    with open("temp_upload.jpg", "rb") as f:
        resp = requests.post(API_URL, files={"file": f})
    data = resp.json()
    return f"Prediction: {data['predicted_class']}\nConfidence: {data['confidence']:.2%}"

demo = gr.Interface(
    fn=predict_ui,
    inputs=gr.Image(type="pil", label="Upload nail image"),
    outputs=gr.Textbox(label="Result"),
    title="Nail Disease Classifier",
    description="Deep learning model (EfficientNet-B0, ONNXRuntime, GPU-accelerated)."
)

if __name__ == "__main__":
    demo.launch(share=True, server_name="0.0.0.0")
