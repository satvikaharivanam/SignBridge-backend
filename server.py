
"""
ASL Flask Inference Server (Render Ready)
========================================
- Loads model from Google Drive if not present
- Uses PyTorch EfficientNetB0
- Handles image preprocessing (white background)
- Exposes /health and /predict endpoints
"""

import os
import io
import json
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import cv2
from flask import Flask, request, jsonify
from flask_cors import CORS
import gdown

# Prevent CPU overload on Render
torch.set_num_threads(1)

app = Flask(__name__)
CORS(app)

# ── Global state ──────────────────────────────────────────────────────────────
model       = None
class_names = []
device      = None

# ── Config ────────────────────────────────────────────────────────────────────
MODEL_DIR = os.environ.get("MODEL_DIR", "./model_output")
MODEL_PATH = os.path.join(MODEL_DIR, "best_model.pth")

# 🔥 REPLACE THIS WITH YOUR GOOGLE DRIVE FILE ID
GDRIVE_FILE_ID = "YOUR_FILE_ID"

PORT = int(os.environ.get("PORT", 10000))
HOST = "0.0.0.0"

# ── Transforms ────────────────────────────────────────────────────────────────
infer_tf = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])

# ── Device ────────────────────────────────────────────────────────────────────
def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

# ── Model ─────────────────────────────────────────────────────────────────────
def build_model(num_classes):
    m = models.efficientnet_b0(weights=None)
    in_features = m.classifier[1].in_features
    m.classifier = nn.Sequential(
        nn.Dropout(p=0.4),
        nn.Linear(in_features, 512),
        nn.SiLU(),
        nn.Dropout(p=0.3),
        nn.Linear(512, num_classes),
    )
    return m

# ── Preprocessing ─────────────────────────────────────────────────────────────
def apply_white_background(img_rgb: np.ndarray) -> np.ndarray:
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    ycrcb   = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2YCrCb)

    lower = np.array([0, 133, 77], dtype=np.uint8)
    upper = np.array([255, 173, 127], dtype=np.uint8)

    mask = cv2.inRange(ycrcb, lower, upper)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, kernel, iterations=1)
    mask = cv2.GaussianBlur(mask, (5, 5), 0)

    _, mask = cv2.threshold(mask, 128, 255, cv2.THRESH_BINARY)

    white = np.ones_like(img_bgr) * 255
    m3 = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR) / 255.0
    result = (img_bgr * m3 + white * (1 - m3)).astype(np.uint8)

    return cv2.cvtColor(result, cv2.COLOR_BGR2RGB)

# ── Routes ────────────────────────────────────────────────────────────────────
@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok', 'classes': len(class_names)})

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400

    try:
        file = request.files['image']
        img_pil = Image.open(io.BytesIO(file.read())).convert('RGB')
        img_np = np.array(img_pil)

        img_wb = apply_white_background(img_np)
        img_pil2 = Image.fromarray(img_wb)

        tensor = infer_tf(img_pil2).unsqueeze(0).to(device)

        with torch.no_grad():
            logits = model(tensor)
            probs = torch.softmax(logits, dim=1).squeeze()

        top3_vals, top3_idxs = torch.topk(probs, 3)

        predictions = [
            {'letter': class_names[i], 'confidence': float(top3_vals[j])}
            for j, i in enumerate(top3_idxs)
        ]

        return jsonify({
            'predictions': predictions,
            'top': predictions[0]['letter'],
            'confidence': predictions[0]['confidence'],
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

# ── Startup ───────────────────────────────────────────────────────────────────
def load_model():
    global model, class_names, device

    device = get_device()
    print(f'🖥️ Device: {device}')

    # Download model if not present
    if not os.path.exists(MODEL_PATH):
        print("⬇️ Downloading model from Google Drive...")
        os.makedirs(MODEL_DIR, exist_ok=True)
        url = f"https://drive.google.com/uc?id={GDRIVE_FILE_ID}"
        gdown.download(url, MODEL_PATH, quiet=False)

    # Load class names
    with open(os.path.join(MODEL_DIR, 'class_names.json')) as f:
        class_names = json.load(f)

    # Load model
    ckpt = torch.load(MODEL_PATH, map_location=device)
    model = build_model(len(class_names)).to(device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    print(f'✅ Model loaded — {len(class_names)} classes')

# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    load_model()
    print(f'🚀 Server running on http://{HOST}:{PORT}')
    app.run(host=HOST, port=PORT, debug=False)
