
import os
import cv2 as cv
import mediapipe as mp
import numpy as np
import torch
import torch.nn as nn
import collections
from collections import Counter
import time

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

#  hyperparams 
MODEL_PATH    = "landmark_asl_best.pth"
ENCODER_PATH  = "label_encoder.npy"
CONFIDENCE_TH = 0.55        # minimum confidence to display a prediction
SMOOTH_WINDOW = 7           

# model 
class LandmarkASL(nn.Module):
    def __init__(self, num_classes=26):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(63, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.1),

            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        return self.net(x)


def normalize_landmarks(landmarks, w, h):
    pts    = np.array([[lm.x * w, lm.y * h, lm.z * w] for lm in landmarks],
                      dtype=np.float32)
    wrist  = pts[0].copy()
    pts   -= wrist
    scale  = np.linalg.norm(pts.max(axis=0) - pts.min(axis=0)) + 1e-6
    pts   /= scale
    return pts.flatten()


#Smoothing 
class PredictionSmoother:
    def __init__(self, window=7):
        self.history = collections.deque(maxlen=window)

    def update(self, label, confidence):
        self.history.append((label, confidence))

    def get(self):
        if not self.history:
            return None, 0.0
        # majority vote
        votes  = Counter(l for l, _ in self.history)
        winner = votes.most_common(1)[0][0]
        avg_conf = np.mean([c for l, c in self.history if l == winner])
        return winner, avg_conf


# ── Overlay helpers ───────────────────────────────────────────────────────────
def draw_confidence_bars(frame, probabilities, class_names, top_k=5, x=10, y=10):
    """Draws a small top-K confidence chart in the corner."""
    top_idx  = np.argsort(probabilities)[::-1][:top_k]
    bar_h    = 18
    bar_maxw = 160
    for rank, idx in enumerate(top_idx):
        prob  = probabilities[idx]
        label = class_names[idx]
        yy    = y + rank * (bar_h + 4)
        # background
        cv.rectangle(frame, (x, yy), (x + bar_maxw, yy + bar_h), (30, 30, 30), -1)
        # fill
        fill_color = (0, int(200 * prob), int(80 * prob))
        cv.rectangle(frame, (x, yy), (x + int(bar_maxw * prob), yy + bar_h),
                     fill_color, -1)
        cv.putText(frame, f"{label}: {prob*100:.0f}%",
                   (x + 4, yy + 13), cv.FONT_HERSHEY_SIMPLEX, 0.45,
                   (255, 255, 255), 1)


def draw_prediction_badge(frame, label, confidence, x, y):
    """Big letter badge with confidence."""
    badge_w, badge_h = 110, 80
    bx, by = max(0, x - 10), max(0, y - badge_h - 10)
    # semi-transparent background
    overlay = frame.copy()
    cv.rectangle(overlay, (bx, by), (bx + badge_w, by + badge_h),
                 (15, 15, 15), -1)
    cv.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
    # letter
    cv.putText(frame, label, (bx + 15, by + 60),
               cv.FONT_HERSHEY_DUPLEX, 2.5, (0, 255, 120), 3)
    # confidence bar under badge
    bar_w = badge_w - 10
    cv.rectangle(frame, (bx + 5, by + badge_h + 4),
                 (bx + 5 + bar_w, by + badge_h + 14), (40, 40, 40), -1)
    cv.rectangle(frame, (bx + 5, by + badge_h + 4),
                 (bx + 5 + int(bar_w * confidence), by + badge_h + 14),
                 (0, 220, 100), -1)
    cv.putText(frame, f"{confidence*100:.0f}%",
               (bx + 5, by + badge_h + 26),
               cv.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 180), 1)


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    # load model        

    class_names = np.load(ENCODER_PATH, allow_pickle=True).tolist()
    num_classes  = len(class_names)
    print(f"Loaded {num_classes} classes: {class_names}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = LandmarkASL(num_classes=num_classes).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()
    print(f"Model loaded on {device}")

    # mediapipe
    mp_hands = mp.solutions.hands
    mp_draw  = mp.solutions.drawing_utils
    mp_style = mp.solutions.drawing_styles
    hands    = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.6
    )

    smoother       = PredictionSmoother(window=SMOOTH_WINDOW)
    show_skeleton  = True
    show_conf_bars = True
    fps_times      = collections.deque(maxlen=30)

    cap = cv.VideoCapture(0)
    print("Starting — press 'q' to quit, 's' skeleton, 'c' confidence bars")

    while cap.isOpened():
        ok, frame = cap.read()
        if not ok:
            break

        t0 = time.time()
        img = cv.flip(frame, 1)
        h = img.shape[0]
        w = img.shape[1]
        img_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        results = hands.process(img_rgb)

        label_display = None
        conf_display  = 0.0

        if results.multi_hand_landmarks:
            lms = results.multi_hand_landmarks[0]

            # draw skeleton
            if show_skeleton:
                mp_draw.draw_landmarks(
                    img, lms, mp_hands.HAND_CONNECTIONS,
                    mp_style.get_default_hand_landmarks_style(),
                    mp_style.get_default_hand_connections_style()
                )

            # extract + normalize features
            features = normalize_landmarks(lms.landmark, w, h)
            tensor   = torch.from_numpy(features).unsqueeze(0).to(device)

            with torch.no_grad():
                logits = model(tensor)
                probs  = torch.nn.functional.softmax(logits, dim=1)[0].cpu().numpy()

            idx        = int(np.argmax(probs))
            confidence = float(probs[idx])
            label      = class_names[idx]

            smoother.update(label, confidence)
            smooth_label, smooth_conf = smoother.get()

            if smooth_conf >= CONFIDENCE_TH:
                label_display = smooth_label
                conf_display  = smooth_conf

            # bounding box for badge placement
            xs = [int(lm.x * w) for lm in lms.landmark]
            ys = [int(lm.y * h) for lm in lms.landmark]
            x1, y1 = max(0, min(xs) - 30), max(0, min(ys) - 30)
            x2, y2 = min(w, max(xs) + 30), min(h, max(ys) + 30)
            cv.rectangle(img, (x1, y1), (x2, y2), (0, 255, 80), 2)

            if label_display:
                draw_prediction_badge(img, label_display, conf_display, x1, y1)

            if show_conf_bars:
                draw_confidence_bars(img, probs, class_names,
                                     top_k=5, x=w - 180, y=10)

        else:
            smoother.history.clear()
            cv.putText(img, "No hand detected", (w // 2 - 120, h // 2),
                       cv.FONT_HERSHEY_SIMPLEX, 0.9, (80, 80, 80), 2)
        
        cv.imshow("ASL Landmark Recognition", img)
        key = cv.waitKey(1) & 0xFF

        if   key == ord('q'): break
        elif key == ord('s'): show_skeleton  = not show_skeleton
        elif key == ord('c'): show_conf_bars = not show_conf_bars

    cap.release()
    cv.destroyAllWindows()


if __name__ == "__main__":
    main()