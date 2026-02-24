# evaluate_pose.py

import json
import math
import cv2
from pose_angles import image_to_angles, extract_keypoints_from_image
import mediapipe as mp

mp_pose = mp.solutions.pose

# ---------------------------------------------
# Configuration
# ---------------------------------------------

REFERENCE_FILE = "C:/Users/nacho/OneDrive - UPV/TELECO/ERASMUS/1erQUATRI/PSBio/PS_YOGA/references/reference_angles/reference_angles.json"
K_SIGMA = 2.0  # tolerance level (2σ recommended)

POSES_ORIENTED = ["Vrikshasana", "Trikasana"]


# ---------------------------------------------
# Orientation utilities
# ---------------------------------------------

def detect_vrikhasana_side(keypoints):
    """
    Determines the raised leg:
    the highest knee (smaller y in MediaPipe)
    """
    lm = mp_pose.PoseLandmark

    left_knee_y = keypoints[lm.LEFT_KNEE.value][1]
    right_knee_y = keypoints[lm.RIGHT_KNEE.value][1]

    if left_knee_y < right_knee_y:
        return "Vrikshasana_left"
    else:
        return "Vrikshasana_right"


def detect_trikasana_side(keypoints):
    """
    Determines bending side:
    the lowest hand (larger y)
    """
    lm = mp_pose.PoseLandmark

    left_wrist_y = keypoints[lm.LEFT_WRIST.value][1]
    right_wrist_y = keypoints[lm.RIGHT_WRIST.value][1]

    if left_wrist_y > right_wrist_y:
        return "Trikasana_left"
    else:
        return "Trikasana_right"


def detect_pose_orientation(pose_name, image_bgr):
    """
    Returns the correct sub-class name
    """
    keypoints = extract_keypoints_from_image(image_bgr)
    if keypoints is None:
        return None

    if pose_name == "Vrikshasana":
        return detect_vrikhasana_side(keypoints)

    if pose_name == "Trikasana":
        return detect_trikasana_side(keypoints)

    return pose_name


# ---------------------------------------------
# Comparison with references
# ---------------------------------------------

def compare_angles(user_angles, reference_angles):
    """
    Returns normalized error per angle
    """
    results = {}

    for angle_name, ref in reference_angles.items():
        mean = ref["mean"]
        var = ref["var"]
        std = math.sqrt(var) if var > 1e-6 else 1e-3

        user_value = user_angles[angle_name]
        error = abs(user_value - mean)
        normalized_error = error / std

        results[angle_name] = {
            "user": user_value,
            "mean": mean,
            "std": std,
            "error": error,
            "normalized_error": normalized_error,
            "ok": normalized_error <= K_SIGMA,
        }

    return results


# ---------------------------------------------
# Full evaluation
# ---------------------------------------------

def evaluate_pose(image_bgr, pose_name):
    with open(REFERENCE_FILE, "r") as f:
        reference_db = json.load(f)

    # Extract user keypoints
    keypoints = extract_keypoints_from_image(image_bgr)
    if keypoints is None:
        print("No keypoints detected")
        return None

    # Detect real orientation
    pose_key = detect_pose_orientation(pose_name, image_bgr)
    print(f"Predicted pose: {pose_name}, detected pose_key: {pose_key}")

    if pose_key is None or pose_key not in reference_db:
        print(f"No reference available for {pose_key}")
        return None

    # Extract angles
    user_angles = image_to_angles(image_bgr)
    if user_angles is None:
        print("Could not compute angles")
        return None

    reference_angles = reference_db[pose_key]
    comparison = compare_angles(user_angles, reference_angles)

    return {
        "pose": pose_key,
        "comparison": comparison
    }



# ---------------------------------------------
# Example usage
# ---------------------------------------------
"""
if __name__ == "__main__":
    result = evaluate_pose(
        image_path="example_user_image.jpg",
        pose_name="vrikhasana"
    )

    for k, v in result["comparison"].items():
        status = "OK" if v["ok"] else "BAD"
        print(f"{k:20s} → {status} (err_norm = {v['normalized_error']:.2f})")
"""