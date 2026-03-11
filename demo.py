"""
ASL Alphabet Recognition - Real-Time Webcam Demo (Patched)
=================================================
Key fix: replaces background with white before inference,
matching the training data distribution (white bg dataset).

Usage:
    python demo.py --model_dir ./model_output --camera 1

Controls:
    Q / ESC  → quit
    S        → save current frame + prediction
    SPACE    → toggle prediction freeze
"""

import os
import sys
import argparse
import json
import time
import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms, models
from torchvision.models import EfficientNet_B0_Weights

try:
    import mediapipe as mp
    MEDIAPIPE_OK = True
except ImportError:
    MEDIAPIPE_OK = False
    print("⚠️  MediaPipe not found — hand cropping disabled, using full frame")

# ── Device ────────────────────────────────────────────────────────────────────
def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

# ── Model ─────────────────────────────────────────────────────────────────────
def build_model(num_classes: int) -> nn.Module:
    model = models.efficientnet_b0(weights=None)
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.4),
        nn.Linear(in_features, 512),
        nn.SiLU(),
        nn.Dropout(p=0.3),
        nn.Linear(512, num_classes),
    )
    return model

# ── Inference transform ───────────────────────────────────────────────────────
infer_tf = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])

# ── White background replacement ──────────────────────────────────────────────
def apply_white_background(crop_bgr, hand_landmarks_in_crop, crop_box):
    """
    Creates a hand mask from landmarks and replaces background with white.
    This matches the training data which had clean white backgrounds.
    """
    h, w = crop_bgr.shape[:2]
    if h == 0 or w == 0:
        return crop_bgr

    # Method: use skin color segmentation in YCrCb space
    # This is fast and works well for hand isolation
    ycrcb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2YCrCb)
    # Skin color range in YCrCb
    lower = np.array([0,   133, 77],  dtype=np.uint8)
    upper = np.array([255, 173, 127], dtype=np.uint8)
    skin_mask = cv2.inRange(ycrcb, lower, upper)

    # Clean up mask with morphological ops
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_DILATE, kernel, iterations=1)
    skin_mask = cv2.GaussianBlur(skin_mask, (5, 5), 0)
    _, skin_mask = cv2.threshold(skin_mask, 128, 255, cv2.THRESH_BINARY)

    # Replace background with white
    white_bg = np.ones_like(crop_bgr) * 255
    mask_3ch = cv2.cvtColor(skin_mask, cv2.COLOR_GRAY2BGR) / 255.0
    result = (crop_bgr * mask_3ch + white_bg * (1 - mask_3ch)).astype(np.uint8)
    return result

# ── Hand bounding box extraction ──────────────────────────────────────────────
def get_hand_crop(frame_rgb, hand_landmarks, pad_frac=0.25):
    h, w = frame_rgb.shape[:2]
    xs = [lm.x * w for lm in hand_landmarks.landmark]
    ys = [lm.y * h for lm in hand_landmarks.landmark]
    pad_x = int((max(xs) - min(xs)) * pad_frac)
    pad_y = int((max(ys) - min(ys)) * pad_frac)
    x1 = max(0, int(min(xs)) - pad_x)
    y1 = max(0, int(min(ys)) - pad_y)
    x2 = min(w, int(max(xs)) + pad_x)
    y2 = min(h, int(max(ys)) + pad_y)
    return frame_rgb[y1:y2, x1:x2], (x1, y1, x2, y2)

# ── Prediction ────────────────────────────────────────────────────────────────
@torch.no_grad()
def predict(model, crop_bgr, device, class_names):
    # Apply white background to match training distribution
    crop_wb = apply_white_background(crop_bgr, None, None)
    crop_rgb = cv2.cvtColor(crop_wb, cv2.COLOR_BGR2RGB)
    tensor = infer_tf(crop_rgb).unsqueeze(0).to(device)
    logits = model(tensor)
    probs  = torch.softmax(logits, dim=1).squeeze()
    top3   = torch.topk(probs, 3)
    return [(class_names[i], probs[i].item()) for i in top3.indices]

# ── HUD ───────────────────────────────────────────────────────────────────────
COLORS = {
    "box":   (0, 220, 120),
    "bg":    (20,  20,  20),
    "top1":  (50, 220, 100),
    "top23": (180, 180, 180),
    "warn":  (0, 100, 255),
}

def draw_hud(frame, predictions, bbox, fps, frozen, show_debug, debug_crop):
    h, w = frame.shape[:2]
    overlay = frame.copy()

    if bbox:
        x1, y1, x2, y2 = bbox
        cv2.rectangle(frame, (x1, y1), (x2, y2), COLORS["box"], 2)

    label, conf = predictions[0]
    font = cv2.FONT_HERSHEY_SIMPLEX

    panel_h = 130
    cv2.rectangle(overlay, (0, h - panel_h), (w, h), COLORS["bg"], -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

    cv2.putText(frame, f"{label}  {conf*100:.1f}%", (20, h - panel_h + 55),
                font, 1.8, COLORS["top1"], 3, cv2.LINE_AA)

    for i, (lbl, c) in enumerate(predictions[1:], 1):
        cv2.putText(frame, f"{i+1}. {lbl}  {c*100:.1f}%",
                    (20, h - panel_h + 75 + i * 28),
                    font, 0.65, COLORS["top23"], 1, cv2.LINE_AA)

    cv2.putText(frame, f"FPS: {fps:.1f}", (w - 130, 30), font, 0.7, (200,200,200), 1, cv2.LINE_AA)
    if frozen:
        cv2.putText(frame, "FROZEN", (w - 130, 60), font, 0.7, COLORS["warn"], 2, cv2.LINE_AA)

    cv2.putText(frame, "Q=Quit  S=Save  SPACE=Freeze  D=Debug",
                (10, 22), font, 0.5, (160,160,160), 1, cv2.LINE_AA)

    # Debug window: show what the model actually sees (white bg crop)
    if show_debug and debug_crop is not None:
        dh, dw = debug_crop.shape[:2]
        scale = 200 / max(dh, dw)
        debug_resized = cv2.resize(debug_crop, (int(dw*scale), int(dh*scale)))
        dh2, dw2 = debug_resized.shape[:2]
        frame[10:10+dh2, w-10-dw2:w-10] = debug_resized
        cv2.putText(frame, "Model sees:", (w-10-dw2, 8), font, 0.4, (200,200,200), 1)

    return frame

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", default="./model_output")
    parser.add_argument("--camera",    type=int, default=1)
    parser.add_argument("--conf",      type=float, default=0.3)
    args = parser.parse_args()

    device = get_device()
    print(f"🖥  Device: {device}")

    with open(os.path.join(args.model_dir, "class_names.json")) as f:
        class_names = json.load(f)

    model_path = os.path.join(args.model_dir, "best_model.pth")
    ckpt  = torch.load(model_path, map_location=device)
    model = build_model(len(class_names)).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"✅ Model loaded (val_acc={ckpt.get('val_acc', '?'):.4f})")

    hands = None
    if MEDIAPIPE_OK:
        mp_hands = mp.solutions.hands
        mp_draw  = mp.solutions.drawing_utils
        hands    = mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.6,
            min_tracking_confidence=0.5,
        )
        print("🖐  MediaPipe Hands ready")

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print(f"❌ Cannot open camera {args.camera}")
        sys.exit(1)

    # Warm up camera (fixes black frame issue on macOS)
    print("⏳ Warming up camera...")
    for _ in range(30):
        cap.read()
    print("🎥 Camera ready — press Q or ESC to quit\n")

    fps_timer   = time.time()
    fps         = 0.0
    frozen      = False
    frozen_frame = None
    predictions = [("?", 0.0), ("?", 0.0), ("?", 0.0)]
    bbox        = None
    save_count  = 0
    show_debug  = True   # press D to toggle
    debug_crop  = None

    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ Failed to grab frame")
            break

        frame = cv2.flip(frame, 1)

        if not frozen:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            crop_bgr  = frame
            bbox      = None
            debug_crop = None

            if hands:
                results = hands.process(frame_rgb)
                if results.multi_hand_landmarks:
                    lm = results.multi_hand_landmarks[0]
                    crop_rgb, bbox = get_hand_crop(frame_rgb, lm)
                    if crop_rgb.size > 0:
                        crop_bgr   = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2BGR)
                        # Show white-bg version in debug window
                        debug_crop = apply_white_background(crop_bgr, None, None)
                    mp_draw.draw_landmarks(frame, lm, mp_hands.HAND_CONNECTIONS)

            predictions = predict(model, crop_bgr, device, class_names)

        now = time.time()
        fps = 0.9 * fps + 0.1 * (1.0 / max(now - fps_timer, 1e-6))
        fps_timer = now

        display = frozen_frame.copy() if frozen and frozen_frame is not None else frame
        draw_hud(display, predictions, bbox, fps, frozen, show_debug, debug_crop)
        cv2.imshow("ASL Sign Language Recognition", display)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            break
        elif key == ord("s"):
            name = f"capture_{save_count:04d}_{predictions[0][0]}.jpg"
            cv2.imwrite(name, display)
            print(f"💾 Saved: {name}")
            save_count += 1
        elif key == ord(" "):
            frozen = not frozen
            frozen_frame = frame.copy() if frozen else None
        elif key == ord("d"):
            show_debug = not show_debug

    cap.release()
    cv2.destroyAllWindows()
    if hands:
        hands.close()
    print("👋 Demo closed.")

if __name__ == "__main__":
    main()