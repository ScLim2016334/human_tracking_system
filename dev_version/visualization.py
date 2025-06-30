"""
Visualization Module
Handles all drawing and display operations
"""

import cv2
import time
from config import *
from state_manager import state_manager
from pose_utils import draw_skeleton

def draw_face_recording_ui(image_bgr, face_locations):
    """Draw UI elements for face recording step"""
    # Display face recording status
    cv2.putText(image_bgr, "STEP 1: Record Master Face (Close to camera)", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(image_bgr, f"Recorded faces: {len(state_manager.get_master_face_encodings())}", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(image_bgr, f"Face Tolerance: {FACE_RECOGNITION_TOLERANCE}", (10, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(image_bgr, "Press 's' to record face, 'n' to proceed", (10, 120),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1, cv2.LINE_AA)
    
    # Draw detected faces
    for (top, right, bottom, left) in face_locations:
        cv2.rectangle(image_bgr, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.putText(image_bgr, "Found Face (Press 's')", (left, top - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)

def draw_skeleton_recording_ui(image_bgr):
    """Draw UI elements for skeleton recording step"""
    # Display skeleton recording status
    cv2.putText(image_bgr, "STEP 2: Record Master Skeleton (Full body visible)", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2, cv2.LINE_AA)
    cv2.putText(image_bgr, f"Recorded faces: {len(state_manager.get_master_face_encodings())}", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 1, cv2.LINE_AA)
    cv2.putText(image_bgr, "Make 'hands on hips' gesture", (10, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 1, cv2.LINE_AA)

def draw_normal_tracking_ui(image_bgr, occlusion_detected):
    """Draw UI elements for normal tracking mode"""
    # Display current master status and face recognition info
    master_status_text = "Master Status: Not Set"
    master_id = state_manager.get_master_id()
    tracked_skeletons = state_manager.get_tracked_skeletons()
    
    if master_id != -1:
        if master_id in tracked_skeletons and tracked_skeletons[master_id]['is_master']:
            master_status_text = f"Master Status: Tracking (ID: {master_id})"
        else:
            if state_manager.is_occlusion_detected():
                master_status_text = f"Master Status: Occluded (IoU frames: {state_manager.get_occlusion_iou_frames()})"
            elif state_manager.is_occlusion_recovery_mode():
                recovery_time = state_manager.get_occlusion_recovery_start_time()
                if recovery_time:
                    elapsed = int(time.time() - recovery_time)
                    master_status_text = f"Master Status: Recovery Mode ({elapsed}s elapsed)"
                else:
                    master_status_text = "Master Status: Recovery Mode"
            else:
                master_status_text = "Master Status: Attempting Re-identification..."
    
    cv2.putText(image_bgr, "STEP 3: Normal Tracking Mode", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(image_bgr, master_status_text, (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(image_bgr, f"Face Tolerance: {FACE_RECOGNITION_TOLERANCE}", (10, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(image_bgr, f"Recorded faces: {len(state_manager.get_master_face_encodings())}", (10, 120),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1, cv2.LINE_AA)
    
    # Display occlusion information
    if state_manager.is_occlusion_detected():
        cv2.putText(image_bgr, f"IoU Threshold: {OCCLUSION_IOU_THRESHOLD}", (10, 150),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 1, cv2.LINE_AA)
        cv2.putText(image_bgr, f"Stop Time: {OCCLUSION_STOP_TIME}s", (10, 170),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 1, cv2.LINE_AA)

def draw_skeletons(image_bgr, step_mode="normal"):
    """Draw all tracked skeletons with appropriate colors and labels"""
    tracked_skeletons = state_manager.get_tracked_skeletons()
    master_id = state_manager.get_master_id()
    
    for s_id, s_data in tracked_skeletons.items():
        if step_mode == "face_recording":
            # Gray for unprocessed skeletons during face recording
            color = (128, 128, 128)
            label = f"ID: {s_id} (Not Master Yet)"
        else:
            # Normal color scheme
            if s_id == master_id and s_data['is_master']:
                color = (0, 0, 255)  # Master: red
                label = f"Master ID: {s_id}"
            else:
                color = (0, 255, 0)  # Others: green
                label = f"ID: {s_id}"
        
        # Draw skeleton
        draw_skeleton(image_bgr, s_data['last_pose'].pose_landmarks)

        # Draw bounding box and ID label
        x, y, w, h = s_data['bbox']
        cv2.rectangle(image_bgr, (x, y), (x + w, y + h), color, 2)
        cv2.putText(image_bgr, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        # Draw IoU information if in normal tracking mode and occlusion detected
        if (step_mode == "normal" and state_manager.is_occlusion_detected() and 
            master_id != -1 and master_id in tracked_skeletons):
            master_bbox = tracked_skeletons[master_id]['bbox']
            if s_id != master_id:  # Only show IoU for non-master skeletons
                iou = calculate_iou(master_bbox, s_data['bbox'])
                iou_text = f"IoU: {iou:.3f}"
                cv2.putText(image_bgr, iou_text, (x, y + h + 20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1, cv2.LINE_AA)

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