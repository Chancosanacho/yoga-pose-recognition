# main_realtime.py

import cv2
import json
import numpy as np
import tensorflow as tf
from collections import deque
import mediapipe as mp
import os

from evaluate_pose import evaluate_pose
from pose_feedback import generate_pose_feedback


# --------------------------------------------------
# Configuration
# --------------------------------------------------

MODEL_PATH = "C:/Users/nacho/OneDrive - UPV/TELECO/ERASMUS/1erQUATRI/PSBio/PS_YOGA/model/modelo_yoga.h5"
LABEL_MAP_PATH = "C:/Users/nacho/OneDrive - UPV/TELECO/ERASMUS/1erQUATRI/PSBio/PS_YOGA/label_map.json"
ICON_PATH = "C:/Users/nacho/OneDrive - UPV/TELECO/ERASMUS/1erQUATRI/PSBio/PS_YOGA/assets/yoga_icon.png"

CONFIDENCE_THRESHOLD = 0.75
MAX_FRAMES = 100
MIN_FRAMES_FOR_PRED = 15

FEEDBACK_HOLD_FRAMES = 20
PANEL_WIDTH = 420

FONT = cv2.FONT_HERSHEY_DUPLEX


# --------------------------------------------------
# Load model and labels
# --------------------------------------------------

model = tf.keras.models.load_model(MODEL_PATH)

with open(LABEL_MAP_PATH, "r") as f:
    label_map = json.load(f)

label_map = {k: int(v) for k, v in label_map.items()}
inv_label_map = {v: k for k, v in label_map.items()}


# --------------------------------------------------
# MediaPipe Pose
# --------------------------------------------------

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(
    static_image_mode=False,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)


# --------------------------------------------------
# Load decorative image
# --------------------------------------------------

icon = None
if os.path.exists(ICON_PATH):
    icon = cv2.imread(ICON_PATH)
    icon = cv2.resize(icon, (120, 120))


# --------------------------------------------------
# Temporal buffers
# --------------------------------------------------

keypoint_buffer = deque(maxlen=MAX_FRAMES)
last_feedback = []
feedback_counter = 0


# --------------------------------------------------
# Camera capture
# --------------------------------------------------

cap = cv2.VideoCapture(0)
print("Press 'q' to exit")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_display = frame.copy()

    # ---------------------------------------------
    # Keypoints
    # ---------------------------------------------

    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = pose.process(image_rgb)

    if result.pose_landmarks:
        keypoints = np.array(
            [[lm.x, lm.y, lm.z] for lm in result.pose_landmarks.landmark]
        ).flatten()
    else:
        keypoints = np.zeros(33 * 3)

    keypoint_buffer.append(keypoints)

    predicted_label = "Detecting pose..."
    feedback_lines = []
    status_text = "Waiting for pose"
    status_color = (180, 180, 180)

    # ---------------------------------------------
    # Classification
    # ---------------------------------------------

    if len(keypoint_buffer) >= MIN_FRAMES_FOR_PRED:

        seq = np.array(keypoint_buffer)

        if len(seq) < MAX_FRAMES:
            seq = np.pad(seq, ((0, MAX_FRAMES - len(seq)), (0, 0)))
        else:
            seq = seq[-MAX_FRAMES:]

        pelvis_index = mp_pose.PoseLandmark.LEFT_HIP.value
        pelvis = seq[:, pelvis_index*3:pelvis_index*3+3]
        seq = seq - pelvis.repeat(33, axis=1)
        seq = seq / (np.max(np.abs(seq)) + 1e-6)

        seq = seq.reshape(1, MAX_FRAMES, 33, 3)

        preds = model.predict(seq, verbose=0)
        class_id = np.argmax(preds)
        confidence = preds[0][class_id]

        if confidence > CONFIDENCE_THRESHOLD:
            pose_name = inv_label_map[class_id]
            predicted_label = f"{pose_name} ({confidence:.2f})"
            status_text = "Pose recognized"
            status_color = (0, 200, 0)

            try:
                if feedback_counter == 0:
                    eval_result = evaluate_pose(frame, pose_name)
                    last_feedback = generate_pose_feedback(eval_result)
                    feedback_counter = FEEDBACK_HOLD_FRAMES
                else:
                    feedback_counter -= 1

                feedback_lines = last_feedback

            except Exception:
                feedback_lines = ["Analyzing posture..."]

    # ---------------------------------------------
    # Graphical user interface (UI)
    # ---------------------------------------------

    h, w, _ = frame_display.shape
    ui = np.zeros((h, w + PANEL_WIDTH, 3), dtype=np.uint8)
    ui[:, :w] = frame_display
    ui[:, w:] = (35, 35, 35)

    # Title
    cv2.putText(ui, "Yoga Pose Correction", (w + 20, 40),
                FONT, 0.85, (245, 245, 245), 2)

    # Icon
    if icon is not None:
        ui[70:190, w + 150:w + 270] = icon

    # Detected pose
    cv2.putText(ui, "Detected Pose", (w + 20, 230),
                FONT, 0.7, (180, 180, 180), 1)

    cv2.putText(ui, predicted_label, (w + 20, 265),
                FONT, 0.75, (0, 255, 180), 2)

    # Corrections
    cv2.putText(ui, "Corrections", (w + 20, 320),
                FONT, 0.7, (180, 180, 180), 1)

    y = 360
    for line in feedback_lines:
        cv2.putText(ui, f"- {line}", (w + 20, y),
                    FONT, 0.65, (255, 220, 120), 1)
        y += 30

    # Status
    cv2.putText(ui, status_text, (w + 20, h - 30),
                FONT, 0.6, status_color, 1)

    cv2.imshow("Yoga Pose Correction", ui)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


cap.release()
pose.close()
cv2.destroyAllWindows()