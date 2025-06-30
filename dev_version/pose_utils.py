"""
Pose Detection Utilities
Contains functions for pose detection, skeleton tracking, and pose analysis
"""

import cv2
import mediapipe as mp
import numpy as np
from config import *

# MediaPipe Initialization
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

def get_bbox_from_landmarks(landmarks, image_width, image_height):
    """Calculate skeleton bounding box from MediaPipe keypoints."""
    if not landmarks:
        return [0, 0, 0, 0] # Return an invalid bbox
    
    x_coords = [lm.x * image_width for lm in landmarks.landmark if lm.visibility > 0.5]
    y_coords = [lm.y * image_height for lm in landmarks.landmark if lm.visibility > 0.5]
    
    if not x_coords or not y_coords: # If not enough visible keypoints
        return [0, 0, 0, 0]

    min_x, max_x = int(min(x_coords)), int(max(x_coords))
    min_y, max_y = int(min(y_coords)), int(max(y_coords))
    
    # Add some padding to ensure the entire body is included
    min_x = max(0, min_x - BBOX_PADDING)
    min_y = max(0, min_y - BBOX_PADDING)
    max_x = min(image_width - 1, max_x + BBOX_PADDING)
    max_y = min(image_height - 1, max_y + BBOX_PADDING)

    return [min_x, min_y, max_x - min_x, max_y - min_y]

def is_hands_on_hips(landmarks):
    """
    Determine if MediaPipe keypoints represent "hands on hips" action.
    This is a simplified version and may need adjustment and optimization based on actual results.
    """
    # Keypoint indices: MediaPipe PoseLandmark
    left_wrist = landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST]
    right_wrist = landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST]
    left_hip = landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP]
    right_hip = landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP]
    left_elbow = landmarks.landmark[mp_pose.PoseLandmark.LEFT_ELBOW]
    right_elbow = landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ELBOW]
    left_shoulder = landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER]
    right_shoulder = landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER]

    # Check wrist visibility to ensure keypoint data is valid
    if not (left_wrist.visibility > KEYPOINT_VISIBILITY_THRESHOLD and 
            right_wrist.visibility > KEYPOINT_VISIBILITY_THRESHOLD and
            left_hip.visibility > KEYPOINT_VISIBILITY_THRESHOLD and 
            right_hip.visibility > KEYPOINT_VISIBILITY_THRESHOLD and
            left_elbow.visibility > KEYPOINT_VISIBILITY_THRESHOLD and 
            right_elbow.visibility > KEYPOINT_VISIBILITY_THRESHOLD):
        return False

    # Check if wrists are near hips (using relative coordinates)
    left_hand_on_hip = (abs(left_wrist.y - left_hip.y) < WRIST_HIP_Y_THRESHOLD_REL and 
                       abs(left_wrist.x - left_hip.x) < WRIST_HIP_X_THRESHOLD_REL)

    right_hand_on_hip = (abs(right_wrist.y - right_hip.y) < WRIST_HIP_Y_THRESHOLD_REL and 
                        abs(right_wrist.x - right_hip.x) < WRIST_HIP_X_THRESHOLD_REL)

    # Check elbow angles: shoulder-elbow-wrist angle to determine if elbows are bent outward
    def calculate_angle(a, b, c):
        a_coords = np.array([a.x, a.y])
        b_coords = np.array([b.x, b.y])
        c_coords = np.array([c.x, c.y])
        
        ba = a_coords - b_coords
        bc = c_coords - b_coords
        
        cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
        angle = np.degrees(np.arccos(cosine_angle))
        return angle

    left_elbow_angle = calculate_angle(left_shoulder, left_elbow, left_wrist)
    right_elbow_angle = calculate_angle(right_shoulder, right_elbow, right_wrist)

    left_elbow_ok = ELBOW_ANGLE_THRESHOLD_MIN < left_elbow_angle < ELBOW_ANGLE_THRESHOLD_MAX
    right_elbow_ok = ELBOW_ANGLE_THRESHOLD_MIN < right_elbow_angle < ELBOW_ANGLE_THRESHOLD_MAX
    
    # At least one hand satisfies hands-on-hips condition
    return (left_hand_on_hip and left_elbow_ok) or (right_hand_on_hip and right_elbow_ok)

def calculate_iou(boxA, boxB):
    """Calculate IoU (Intersection over Union) of two bounding boxes"""
    # box: [x, y, w, h]
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[0] + boxA[2], boxB[0] + boxB[2])
    yB = min(boxA[1] + boxA[3], boxB[1] + boxB[3])

    interArea = max(0, xB - xA) * max(0, yB - yA)

    boxAArea = boxA[2] * boxA[3]
    boxBArea = boxB[2] * boxB[3]

    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou

def draw_skeleton(image, landmarks, color=(245, 117, 66)):
    """Draw skeleton landmarks on image"""
    mp_drawing.draw_landmarks(
        image, 
        landmarks, 
        mp_pose.POSE_CONNECTIONS,
        mp_drawing.DrawingSpec(color=color, thickness=2, circle_radius=2),
        mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
    ) 