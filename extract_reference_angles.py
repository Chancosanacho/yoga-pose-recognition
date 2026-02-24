# extract_reference_angles_oriented.py

import cv2
import os
import json
import numpy as np
from pose_angles import image_to_angles, extract_keypoints_from_image
import mediapipe as mp

mp_pose = mp.solutions.pose

REFERENCE_DIR = "C:/Users/nacho/OneDrive - UPV/TELECO/ERASMUS/1erQUATRI/PSBio/PS_YOGA/references/classes/"
OUTPUT_FILE = "C:/Users/nacho/OneDrive - UPV/TELECO/ERASMUS/1erQUATRI/PSBio/PS_YOGA/references/reference_angles/reference_angles.json"

# Poses that depend on lateral orientation or supporting leg
POSES_ORIENTED = ["Vrikshasana", "Trikasana"]

reference_data = {}

for pose_name in os.listdir(REFERENCE_DIR):
    pose_folder = os.path.join(REFERENCE_DIR, pose_name)
    if not os.path.isdir(pose_folder):
        continue

    # Temporary dictionary for sub-classes according to orientation
    sub_classes = {}

    for img_file in os.listdir(pose_folder):
        img_path = os.path.join(pose_folder, img_file)
        img = cv2.imread(img_path)
        if img is None:
            print(f"Invalid or not found image: {img_path}")
            continue  # skip this image
        angles = image_to_angles(img)
        if angles is None:
            continue

        keypoints = extract_keypoints_from_image(img)
        if keypoints is None:
            continue

        # Determine sub-class according to orientation
        if pose_name in POSES_ORIENTED:
            if pose_name == "Vrikshasana":
                # Check which foot is higher (smaller y coordinate)
                left_ankle_y = keypoints[mp_pose.PoseLandmark.LEFT_ANKLE.value][1]
                right_ankle_y = keypoints[mp_pose.PoseLandmark.RIGHT_ANKLE.value][1]
                if left_ankle_y < right_ankle_y:
                    sub_class = f"{pose_name}_left"
                else:
                    sub_class = f"{pose_name}_right"

            elif pose_name == "Trikasana":
                # Check which hand is higher (smaller y coordinate)
                left_wrist_y = keypoints[mp_pose.PoseLandmark.LEFT_WRIST.value][1]
                right_wrist_y = keypoints[mp_pose.PoseLandmark.RIGHT_WRIST.value][1]
                if left_wrist_y < right_wrist_y:
                    sub_class = f"{pose_name}_left"
                else:
                    sub_class = f"{pose_name}_right"

        else:
            sub_class = pose_name

        if sub_class not in sub_classes:
            sub_classes[sub_class] = []

        sub_classes[sub_class].append(angles)

    # Compute mean and variance per sub-class
    for sub_class, angle_list in sub_classes.items():
        reference_data[sub_class] = {
            k: {
                "mean": float(np.mean([a[k] for a in angle_list])),
                "var": float(np.var([a[k] for a in angle_list]))
            }
            for k in angle_list[0].keys()
        }

# Save final JSON
os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
with open(OUTPUT_FILE, "w") as f:
    json.dump(reference_data, f, indent=2)

print(f"Reference angles saved in {OUTPUT_FILE}")