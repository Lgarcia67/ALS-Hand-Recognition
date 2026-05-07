import os
import cv2 as cv
import mediapipe as mp
import numpy as np
import csv
import time

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

SAMPLES_PER_CLASS   = 150        
OUTPUT_CSV          = "landmark_data.csv"
MIN_DETECTION_CONF  = 0.7
MIN_TRACKING_CONF   = 0.6

#mediapipe setup
mp_hands = mp.solutions.hands
mp_draw  = mp.solutions.drawing_utils
hands    = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=MIN_DETECTION_CONF,
    min_tracking_confidence=MIN_TRACKING_CONF
)

CLASS_NAMES = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")

def normalize_landmarks(landmarks, w, h):
    """
    Returns 63 floats (21 × xyz) normalized relative to the wrist (landmark 0)
    and scaled by the hand's bounding-box diagonal so distance-to-camera
    doesn't matter.
    """
    pts = np.array([[lm.x * w, lm.y * h, lm.z * w] for lm in landmarks])
    # Center on wrist
    wrist = pts[0].copy()
    pts  -= wrist
    # Scale by bounding-box diagonal
    scale = np.linalg.norm(pts.max(axis=0) - pts.min(axis=0)) + 1e-6
    pts  /= scale
    return pts.flatten().tolist()


def load_existing(path):
    """Returns list of already-saved rows and a count dict."""
    rows   = []
    counts = {c: 0 for c in CLASS_NAMES}
    if not os.path.exists(path):
        return rows, counts
    with open(path, "r") as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) == 64:          # 63 features + 1 label
                rows.append(row)
                label = row[-1].upper()
                if label in counts:
                    counts[label] += 1
    return rows, counts


def save_all(path, rows):
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(rows)


def draw_ui(frame, active_letter, counts, status_msg, recording, record_progress):
    h, w = frame.shape[:2]
    overlay = frame.copy()

    # ── dark sidebar ──────────────────────────────────────────────────────────
    cv.rectangle(overlay, (0, 0), (220, h), (15, 15, 20), -1)
    cv.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

    # title
    cv.putText(frame, "ASL Collector", (10, 30),
               cv.FONT_HERSHEY_DUPLEX, 0.7, (200, 200, 255), 1)
    cv.putText(frame, f"Target: {SAMPLES_PER_CLASS}/letter", (10, 52),
               cv.FONT_HERSHEY_SIMPLEX, 0.42, (150, 150, 150), 1)

    # letter grid
    for i, c in enumerate(CLASS_NAMES):
        col = i % 6
        row = i // 6
        x   = 12 + col * 34
        y   = 80 + row * 34
        pct = min(counts[c] / SAMPLES_PER_CLASS, 1.0)
        # background tile
        color = (0, int(180 * pct), int(60 * pct)) if pct < 1.0 else (0, 200, 80)
        cv.rectangle(frame, (x - 2, y - 18), (x + 24, y + 4), color, -1)
        text_color = (255, 255, 255) if c == active_letter else (200, 200, 200)
        cv.putText(frame, c, (x, y),
                   cv.FONT_HERSHEY_SIMPLEX, 0.55, text_color, 1)
        # count
        cv.putText(frame, str(counts[c]), (x, y + 14),
                   cv.FONT_HERSHEY_SIMPLEX, 0.3, (180, 180, 180), 1)

    # active letter big display
    if active_letter:
        cv.putText(frame, active_letter, (w - 100, 80),
                   cv.FONT_HERSHEY_DUPLEX, 3.0,
                   (0, 255, 120) if recording else (100, 100, 200), 4)

    # recording progress bar
    if recording and active_letter:
        bar_x, bar_y, bar_w, bar_h = w - 160, h - 50, 140, 18
        cv.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h),
                     (50, 50, 50), -1)
        fill = int(bar_w * record_progress)
        cv.rectangle(frame, (bar_x, bar_y), (bar_x + fill, bar_y + bar_h),
                     (0, 220, 100), -1)
        cv.putText(frame, "Recording...", (bar_x, bar_y - 6),
                   cv.FONT_HERSHEY_SIMPLEX, 0.45, (0, 220, 100), 1)

    # status message
    cv.putText(frame, status_msg, (230, h - 15),
               cv.FONT_HERSHEY_SIMPLEX, 0.5, (180, 220, 255), 1)

    # controls hint
    hints = ["[a-z] start letter", "[u] undo last", "[q] quit & save"]
    for i, hint in enumerate(hints):
        cv.putText(frame, hint, (230, h - 15 - (len(hints) - 1 - i) * 22),
                   cv.FONT_HERSHEY_SIMPLEX, 0.4, (100, 100, 100), 1)

    return frame


def main():
    saved_rows, counts = load_existing(OUTPUT_CSV)
    print(f"Loaded {len(saved_rows)} existing samples from '{OUTPUT_CSV}'")

    cap          = cv.VideoCapture(0)
    active_letter = None
    recording    = False
    record_start = 0.0
    status_msg   = "Press a letter key to begin"
    session_rows  = list(saved_rows)          # will grow as we record

    while cap.isOpened():
        ok, frame = cap.read()
        if not ok:
            break
        img     = cv.flip(frame, 1)
        h, w, _ = img.shape
        img_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        results = hands.process(img_rgb)

        record_progress = 0.0

        # ── landmark processing ───────────────────────────────────────────────
        hand_detected = False
        if results.multi_hand_landmarks:
            hand_detected = True
            lms = results.multi_hand_landmarks[0]
            mp_draw.draw_landmarks(img, lms, mp_hands.HAND_CONNECTIONS)

            if recording and active_letter:
                elapsed          = time.time() - record_start
                record_progress  = min(elapsed / 0.5, 1.0)   # 0.5 s capture window

                if record_progress >= 1.0 and counts[active_letter] < SAMPLES_PER_CLASS:
                    features = normalize_landmarks(lms.landmark, w, h)
                    row      = features + [active_letter]
                    session_rows.append(row)
                    counts[active_letter] += 1
                    save_all(OUTPUT_CSV, session_rows)
                    remaining = SAMPLES_PER_CLASS - counts[active_letter]
                    status_msg = (f"Saved {counts[active_letter]}/{SAMPLES_PER_CLASS} "
                                  f"for '{active_letter}'  —  {remaining} to go")
                    recording    = False
                    record_start = 0.0

                    if counts[active_letter] >= SAMPLES_PER_CLASS:
                        status_msg = f"'{active_letter}' complete! Pick next letter."
                        active_letter = None

        else:
            if recording:
                recording    = False
                record_start = 0.0
                status_msg   = "Hand lost — hold steady and press key again"

        # ── draw UI ───────────────────────────────────────────────────────────
        img = draw_ui(img, active_letter, counts, status_msg,
                      recording, record_progress)

        cv.imshow("ASL Landmark Collector", img)

        key = cv.waitKey(1) & 0xFF

        if key == ord('q'):
            break

        elif key != 255:
            ch = chr(key).upper()
            if ch in CLASS_NAMES:
                if counts[ch] >= SAMPLES_PER_CLASS:
                    status_msg = f"'{ch}' already has {SAMPLES_PER_CLASS} samples!"
                elif not hand_detected:
                    status_msg = "No hand detected — show your hand first"
                else:
                    active_letter = ch
                    recording     = True
                    record_start  = time.time()
                    status_msg    = f"Hold '{ch}' sign steady..."

    cap.release()
    cv.destroyAllWindows()
    total = sum(counts.values())
    print(f"\nDone! {total} total samples saved to '{OUTPUT_CSV}'")
    for c in CLASS_NAMES:
        bar = "█" * counts[c] + "░" * max(0, SAMPLES_PER_CLASS - counts[c])
        print(f"  {c}: {bar}  {counts[c]}/{SAMPLES_PER_CLASS}")


if __name__ == "__main__":
    main()