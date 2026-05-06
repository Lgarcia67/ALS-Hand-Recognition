import os
import cv2 as cv
import mediapipe as mp
import numpy as np
import torch
from model import model 
from resnet import ResNetASL  
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cnn = ResNetASL()
#cnn.model_architecture()
cnn.model.load_state_dict(torch.load('resnet_asl_best_collab.pth', map_location=device))
cnn.model.to(device)
cnn.model.eval()

class_names = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']

#Setup MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

cap = cv.VideoCapture(0)

while cap.isOpened():
    success, frame = cap.read()
    if not success: break

    img = cv.flip(frame, 1)
    h, w, c = img.shape
    img_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Box from landmarks
            x_max, y_max = 0, 0
            x_min, y_min = w, h
            for lm in hand_landmarks.landmark:
                x, y = int(lm.x * w), int(lm.y * h)
                x_max, y_max = max(x, x_max), max(y, y_max)
                x_min, y_min = min(x, x_min), min(y, y_min)

            # crop
            y1, y2 = max(0, y_min-20), min(h, y_max+40)
            x1, x2 = max(0, x_min-40), min(w, x_max+40)
            hand_crop = img_rgb[y1:y2, x1:x2]
            
            if hand_crop.size != 0:
                input_img = cv.resize(hand_crop, (224, 224))  # fix typo
                input_img = input_img.astype(np.float32) / 255.0

                mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
                std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
                input_img = (input_img - mean) / std

                input_img = np.transpose(input_img, (2, 0, 1))
                input_tensor = torch.from_numpy(input_img).unsqueeze(0).to(device)

            with torch.no_grad():
                output = cnn.model(input_tensor)
                probabilities = torch.nn.functional.softmax(output, dim=1)[0]
                idx = torch.argmax(probabilities).item()
                label = class_names[idx]
                confidence = probabilities[idx].item()

                # display box and prediction
                cv.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv.putText(img, f"{label} ({confidence*100:.1f}%)", (x1, y1-10), 
                           cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv.imshow("ASL Recognition", img)
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv.destroyAllWindows()

