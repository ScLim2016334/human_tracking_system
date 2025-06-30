"""
Human Tracking System - Refactored Main Module
Integrates all modules for human tracking with face recognition and IoU-based occlusion handling
"""

import cv2
import mediapipe as mp
import numpy as np

# Import our modules
from config import *
from state_manager import state_manager
from pose_utils import draw_skeleton
from skeleton_tracker import update_skeleton_ids, detect_iou_occlusion, handle_occlusion_recovery
from master_calibration import record_master_faces, calibrate_master_skeleton, attempt_master_reidentification
from face_recognition_utils import detect_faces_in_image
from visualization import draw_face_recording_ui, draw_skeleton_recording_ui, draw_normal_tracking_ui, draw_skeletons

# MediaPipe Initialization
mp_pose = mp.solutions.pose

def main():
    """Main function for human tracking system"""
    
    cap = cv2.VideoCapture(0) # 0 represents default camera
    if not cap.isOpened():
        print("Error: Cannot open camera. Please check device connection or permissions.")
        return

    # Try to read one frame to get image dimensions
    ret, frame = cap.read()
    if not ret:
        print("Error: Cannot read frame from camera.")
        cap.release()
        return
    image_height, image_width, _ = frame.shape
    print(f"Camera resolution: {image_width}x{image_height}")

    print("\n--- Human Following Algorithm with Face Recognition ---")
    print("=== STEP 1: Record Master Face (Close to camera) ===")
    print("1. Move close to camera and press 's' multiple times to record your face from different angles.")
    print("2. Press 'r' to reset face records if needed.")
    print("3. Press 'n' to proceed to next step when face recording is complete.")

    # Initialize MediaPipe Pose model
    with mp_pose.Pose(min_detection_confidence=MIN_DETECTION_CONFIDENCE, 
                     min_tracking_confidence=MIN_TRACKING_CONFIDENCE) as pose:
        frame_count = 0
        step2_introduced = False  # Flag to track if step 2 has been introduced
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1
            
            # Convert image from BGR to RGB (MediaPipe requires RGB format)
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image_rgb.flags.writeable = False # Improve performance, mark image as read-only
            
            # MediaPipe pose detection
            results = pose.process(image_rgb)
            
            image_rgb.flags.writeable = True # Re-mark image as writable for drawing
            image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

            current_skeletons_in_frame_data = [] # Store all skeleton data detected in current frame
            if results.pose_landmarks:
                current_skeletons_in_frame_data.append((results.pose_landmarks, results))

            # --- 1. Update skeleton IDs (core tracking logic) ---
            update_skeleton_ids(current_skeletons_in_frame_data, frame_count, image_width, image_height)
            
            # --- 2. IoU-based occlusion detection and recovery (only in normal tracking mode) ---
            occlusion_detected = False
            if state_manager.is_master_skeleton_recorded() and state_manager.get_master_id() != -1:
                # Detect occlusion using IoU
                occlusion_detected = detect_iou_occlusion(current_skeletons_in_frame_data, frame_count, image_width, image_height)
                
                # Handle recovery if in recovery mode
                if state_manager.is_occlusion_recovery_mode():
                    handle_occlusion_recovery(current_skeletons_in_frame_data, frame_count, image_width, image_height)
            
            # --- 3. Different logic based on recording stage ---
            if not state_manager.is_master_face_recorded():
                # STEP 1: Record master face (close to camera)
                # Detect faces in entire image for face recording
                face_locations, face_encodings = detect_faces_in_image(image_rgb)
                
                # Draw UI and faces
                draw_face_recording_ui(image_bgr, face_locations)
                draw_skeletons(image_bgr, step_mode="face_recording")
                
            elif not state_manager.is_master_skeleton_recorded():
                # STEP 2: Record master skeleton (move away from camera)
                if not step2_introduced:
                    print("\n=== STEP 2: Record Master Skeleton (Move away from camera) ===")
                    print("1. Move away from camera so your full body is visible.")
                    print("2. Make 'hands on hips' gesture to calibrate as master.")
                    step2_introduced = True
                
                # Draw UI
                draw_skeleton_recording_ui(image_bgr)
                
                # Try to calibrate master skeleton
                calibrate_master_skeleton(image_rgb, frame_count)
                
                # Draw skeletons
                draw_skeletons(image_bgr)
                
            else:
                # STEP 3: Normal tracking with both face and skeleton recognition
                # --- Master re-identification (only when master not currently in view and not occluded) ---
                is_master_present_this_frame = False
                master_id = state_manager.get_master_id()
                tracked_skeletons = state_manager.get_tracked_skeletons()
                
                if master_id != -1 and master_id in tracked_skeletons:
                    is_master_present_this_frame = tracked_skeletons[master_id]['is_master']

                # Only perform face re-identification when master skeleton is not tracked and not occluded
                if (master_id != -1 and not is_master_present_this_frame and 
                    state_manager.get_master_face_encodings() and not occlusion_detected and
                    not state_manager.is_occlusion_recovery_mode()):
                    attempt_master_reidentification(image_rgb, image_width, image_height)

                # Draw UI and skeletons
                draw_normal_tracking_ui(image_bgr, occlusion_detected)
                draw_skeletons(image_bgr)

            cv2.imshow('Human Following Algorithm Demo', image_bgr)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('s'): # Press 's' to record master face information
                if not state_manager.is_master_face_recorded():
                    record_master_faces(image_rgb)
                        
            elif key == ord('n'): # Press 'n' to proceed to next step
                if not state_manager.is_master_face_recorded() and len(state_manager.get_master_face_encodings()) > 0:
                    state_manager.set_master_face_recorded(True)
                    print(f"\n=== Proceeding to Step 2: Skeleton Recording ===")
                    print(f"Face recording complete! {len(state_manager.get_master_face_encodings())} faces recorded.")
                    print("Now move away from camera and make 'hands on hips' gesture.")
                elif state_manager.is_master_face_recorded() and not state_manager.is_master_skeleton_recorded():
                    print("Please complete skeleton recording first by making 'hands on hips' gesture.")
                else:
                    print("Already in normal tracking mode.")
                    
            elif key == ord('r'): # Press 'r' to reset records
                state_manager.reset_all()
                step2_introduced = False  # Reset step 2 introduction flag
                print("--- All recorded information and master status has been reset ---")
                print("=== Back to Step 1: Face Recording ===")
            elif key == ord('q'): # Press 'q' to quit
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main() 