# pose_feedback.py

"""
Converts the output from evaluate_pose.py into textual feedback for the user
"""

# --------------------------------------------------
# Feedback rules per pose and angle
# --------------------------------------------------

POSE_FEEDBACK_RULES = {

    "Tadasana": {
        "spine_inclination": {
            "too_large": "Straighten your back",
        },
        "hip_alignment": {
            "too_large": "Align your hips better",
        },
        "ankle_alignment": {
            "too_large": "Keep your feet parallel",
        },
        "left_elbow": {
            "too_small": "Extend your left arm more",
        },
        "right_elbow": {
            "too_small": "Extend your right arm more",
        }
    },

    "Bhujasana": {
        "left_elbow": {
            "too_small": "Extend your left elbow more",
            "too_large": "Bend your left elbow a little less",
        },
        "right_elbow": {
            "too_small": "Extend your right elbow more",
            "too_large": "Bend your right elbow a little less",
        },
        "spine_inclination": {
            "too_small": "Lift your chest more",
            "too_large": "Don't arch your back too much",
        }
    },

    "Vrikshasana_left": {
        "right_knee": {
            "too_large": "Bend the raised leg more",
            "too_small": "Don't stretch the raised leg too much",
        },
        "spine_inclination": {
            "too_large": "Keep your torso straighter",
        }
    },

    "Vrikshasana_right": {
        "left_knee": {
            "too_large": "Bend the raised leg more",
            "too_small": "Don't stretch the raised leg too much",
        },
        "spine_inclination": {
            "too_large": "Keep your torso straighter",
        }
    },

    "Trikasana_left": {
        "spine_inclination": {
            "too_small": "Lean more to the side",
            "too_large": "Reduce the lateral tilt slightly",
        },
        "left_knee": {
            "too_small": "Extend your front knee more",
        },
        "right_knee": {
            "too_small": "Extend your back knee more",
        },
        "left_elbow": {
            "too_small": "Extend your lower arm more",
        },
        "right_elbow": {
            "too_small": "Extend your upper arm more",
        }
    },

    "Trikasana_right": {
        "spine_inclination": {
            "too_small": "Lean more to the side",
            "too_large": "Reduce the lateral tilt slightly",
        },
        "right_knee": {
            "too_small": "Extend your front knee more",
        },
        "left_knee": {
            "too_small": "Extend your back knee more",
        },
        "right_elbow": {
            "too_small": "Extend your lower arm more",
        },
        "left_elbow": {
            "too_small": "Extend your upper arm more",
        }
    },

    "Padamasana": {
        "left_knee": {
            "too_large": "Relax your left leg",
            "too_small": "Open your left leg a little more",
        },
        "right_knee": {
            "too_large": "Relax your right leg",
            "too_small": "Open your right leg a little more",
        },
        "spine_inclination": {
            "too_large": "Straighten your back",
            "too_small": "Keep your torso slightly upright",
        }
    }
}


# --------------------------------------------------
# Feedback generation
# --------------------------------------------------

def generate_pose_feedback(evaluation_result, max_messages=3):
    """
    Converts the output of evaluate_pose into textual feedback
    """

    pose_name = evaluation_result["pose"]
    comparison = evaluation_result["comparison"]

    feedback_messages = []

    if pose_name not in POSE_FEEDBACK_RULES:
        return ["Pose detected, but no specific correction rules available"]

    rules = POSE_FEEDBACK_RULES[pose_name]

    sorted_angles = sorted(
        comparison.items(),
        key=lambda x: x[1]["normalized_error"],
        reverse=True
    )

    for angle_name, data in sorted_angles:
        if data["ok"]:
            continue

        if angle_name not in rules:
            continue

        mean = data["mean"]
        user_value = data["user"]

        error_type = "too_small" if user_value < mean else "too_large"

        angle_rules = rules[angle_name]

        if error_type in angle_rules:
            feedback_messages.append(angle_rules[error_type])

        if len(feedback_messages) >= max_messages:
            break

    if not feedback_messages:
        feedback_messages.append("Great posture, keep it up")

    return feedback_messages
