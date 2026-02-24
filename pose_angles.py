# pose_angles.py

import cv2
import numpy as np
import mediapipe as mp
import math

mp_pose = mp.solutions.pose


# --------------------------------------------------
# Mathematical utilities
# --------------------------------------------------

def angle_between_points(a, b, c):
    """
    Computes angle ABC (in degrees)
    a, b, c: np.array([x,y,z])
    """
    ba = a - b
    bc = c - b

    cosine_angle = np.dot(ba, bc) / (
        np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6
    )
    angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
    return np.degrees(angle)


def inclination_from_vertical(a, b):
    """
    Inclination of segment AB with respect to the vertical
    (0° = completely vertical)
    """
    vector = a - b
    vertical = np.array([0, -1, 0])
    cosine = np.dot(vector, vertical) / (
        np.linalg.norm(vector) * np.linalg.norm(vertical) + 1e-6
    )
    return np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))


# --------------------------------------------------
# Keypoint extraction from image
# --------------------------------------------------

def extract_keypoints_from_image(image_bgr):
    """
    Returns an array (33,3) with the keypoints
    """
    pose = mp_pose.Pose(static_image_mode=True)
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    
    result = pose.process(image_rgb)
    pose.close()

    if not result.pose_landmarks:
        return None

    keypoints = np.array(
        [[lm.x, lm.y, lm.z] for lm in result.pose_landmarks.landmark]
    )
    return keypoints


# --------------------------------------------------
# Biomechanical angle computation
# --------------------------------------------------

def compute_angles(keypoints):
    """
    Input: keypoints (33,3)
    Output: dictionary of angles
    """

    lm = mp_pose.PoseLandmark

    angles = {}

    # --- Arms ---
    angles["left_elbow"] = angle_between_points(
        keypoints[lm.LEFT_SHOULDER.value],
        keypoints[lm.LEFT_ELBOW.value],
        keypoints[lm.LEFT_WRIST.value],
    )

    angles["right_elbow"] = angle_between_points(
        keypoints[lm.RIGHT_SHOULDER.value],
        keypoints[lm.RIGHT_ELBOW.value],
        keypoints[lm.RIGHT_WRIST.value],
    )

    angles["left_shoulder"] = angle_between_points(
        keypoints[lm.LEFT_ELBOW.value],
        keypoints[lm.LEFT_SHOULDER.value],
        keypoints[lm.LEFT_HIP.value],
    )

    angles["right_shoulder"] = angle_between_points(
        keypoints[lm.RIGHT_ELBOW.value],
        keypoints[lm.RIGHT_SHOULDER.value],
        keypoints[lm.RIGHT_HIP.value],
    )

    # --- Legs ---
    angles["left_knee"] = angle_between_points(
        keypoints[lm.LEFT_HIP.value],
        keypoints[lm.LEFT_KNEE.value],
        keypoints[lm.LEFT_ANKLE.value],
    )

    angles["right_knee"] = angle_between_points(
        keypoints[lm.RIGHT_HIP.value],
        keypoints[lm.RIGHT_KNEE.value],
        keypoints[lm.RIGHT_ANKLE.value],
    )

    angles["left_hip"] = angle_between_points(
        keypoints[lm.LEFT_SHOULDER.value],
        keypoints[lm.LEFT_HIP.value],
        keypoints[lm.LEFT_KNEE.value],
    )

    angles["right_hip"] = angle_between_points(
        keypoints[lm.RIGHT_SHOULDER.value],
        keypoints[lm.RIGHT_HIP.value],
        keypoints[lm.RIGHT_KNEE.value],
    )

    # --- Spine ---
    shoulder_center = (
        keypoints[lm.LEFT_SHOULDER.value] +
        keypoints[lm.RIGHT_SHOULDER.value]
    ) / 2

    hip_center = (
        keypoints[lm.LEFT_HIP.value] +
        keypoints[lm.RIGHT_HIP.value]
    ) / 2

    angles["spine_inclination"] = inclination_from_vertical(
        shoulder_center, hip_center
    )

    # --- Alignments ---
    angles["hip_alignment"] = abs(
        keypoints[lm.LEFT_HIP.value][1] -
        keypoints[lm.RIGHT_HIP.value][1]
    )

    angles["ankle_alignment"] = abs(
        keypoints[lm.LEFT_ANKLE.value][1] -
        keypoints[lm.RIGHT_ANKLE.value][1]
    )

    return angles


# --------------------------------------------------
# Full pipeline (image → angles)
# --------------------------------------------------

def image_to_angles(image_bgr):
    keypoints = extract_keypoints_from_image(image_bgr)
    if keypoints is None:
        return None
    return compute_angles(keypoints)