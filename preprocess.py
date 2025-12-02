# preprocess.py
from PIL import Image, ImageFilter
import numpy as np
import cv2

IMG_SIZE = 224

def extract_roi(img):
    arr = np.array(img.convert("L"))
    mask = arr > 10
    coords = np.argwhere(mask)
    if coords.size == 0:
        return img
    y0, x0 = coords.min(axis=0)
    y1, x1 = coords.max(axis=0)
    return img.crop((x0, y0, x1, y1))

def apply_clahe(img):
    gray = np.array(img.convert("L"))
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    return Image.fromarray(enhanced).convert("RGB")

def preprocess_image_pil(img):
    # img: PIL Image
    img = extract_roi(img)
    img = apply_clahe(img)
    img = img.filter(ImageFilter.SHARPEN)
    img = img.resize((IMG_SIZE, IMG_SIZE))

    arr = np.array(img).astype("float32") / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype="float32")
    std = np.array([0.229, 0.224, 0.225], dtype="float32")
    arr = (arr - mean) / std
    arr = arr.transpose(2, 0, 1)  # HWC -> CHW
    return arr  # shape (3, 224, 224)
