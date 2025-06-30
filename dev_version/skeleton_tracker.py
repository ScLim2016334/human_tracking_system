"""
Skeleton Tracker Module
Handles skeleton ID assignment, tracking, and IoU-based occlusion detection
"""

import numpy as np
import time
from config import *
from state_manager import state_manager
from pose_utils import get_bbox_from_landmarks, calculate_iou
from voice_announcer import voice_announcer

def update_skeleton_ids(current_skeletons_data, frame_idx, image_width, image_height):
    """
    Update and maintain skeleton IDs.
    Try to match IDs by comparing current frame skeletons with previous frame skeletons' positions, and remove skeletons not seen for a long time.
    """
    tracked_skeletons = state_manager.get_tracked_skeletons()
    next_id = state_manager.get_next_id()
    master_id = state_manager.get_master_id()

    new_tracked_skeletons = {}
    matched_current_indices = set()

    for current_idx, (landmarks, pose_results) in enumerate(current_skeletons_data):
        # Calculate current skeleton center point (for matching)
        current_center = np.mean([[lm.x, lm.y] for lm in landmarks.landmark], axis=0) * np.array([image_width, image_height])
        current_bbox = get_bbox_from_landmarks(landmarks, image_width, image_height)

        best_match_id = -1
        min_dist = float('inf')

        # Try to match existing IDs
        for s_id, s_data in tracked_skeletons.items():
            last_center = np.mean([[lm.x, lm.y] for lm in s_data['last_pose'].pose_landmarks.landmark], axis=0) * np.array([image_width, image_height])
            dist = np.linalg.norm(current_center - last_center)
            
            if dist < min_dist and dist < MAX_DIST_THRESHOLD:
                min_dist = dist
                best_match_id = s_id
        
        # If match successful and this ID hasn't been matched by other skeletons in this frame
        if best_match_id != -1 and best_match_id not in new_tracked_skeletons:
            new_tracked_skeletons[best_match_id] = {
                'last_pose': pose_results,
                'bbox': current_bbox,
                'last_seen_frame': frame_idx,
                'is_master': tracked_skeletons[best_match_id]['is_master'],
            }
            matched_current_indices.add(current_idx)
        else:
            # No match, assign new ID
            new_tracked_skeletons[next_id] = {
                'last_pose': pose_results,
                'bbox': current_bbox,
                'last_seen_frame': frame_idx,
                'is_master': False, # New skeleton is not master by default
            }
            state_manager.increment_next_id()
            matched_current_indices.add(current_idx)

    # Remove skeletons not seen for a long time
    temp_tracked_skeletons = {}
    for s_id, s_data in new_tracked_skeletons.items():
        if frame_idx - s_data['last_seen_frame'] <= MAX_MISS_FRAMES:
            temp_tracked_skeletons[s_id] = s_data
        else:
            # If master disappears, reset MASTER_ID
            if s_id == master_id:
                state_manager.set_master_id(-1)
                print(f"Master ID {s_id} skeleton not seen for a long time, resetting master status.")
    
    state_manager.set_tracked_skeletons(temp_tracked_skeletons)

def detect_iou_occlusion(current_skeletons_data, frame_idx, image_width, image_height):
    """
    Detect occlusion using IoU between master and other skeletons.
    Returns occlusion status and handles voice announcements.
    """
    tracked_skeletons = state_manager.get_tracked_skeletons()
    master_id = state_manager.get_master_id()
    
    if master_id == -1 or master_id not in tracked_skeletons:
        return False
    
    # Get master skeleton
    master_skeleton = tracked_skeletons[master_id]
    master_bbox = master_skeleton['bbox']
    
    # Check IoU with all other skeletons
    max_iou = 0.0
    for s_id, s_data in tracked_skeletons.items():
        if s_id != master_id:  # Skip master itself
            other_bbox = s_data['bbox']
            iou = calculate_iou(master_bbox, other_bbox)
            max_iou = max(max_iou, iou)
    
    # Check if IoU exceeds threshold
    if max_iou > OCCLUSION_IOU_THRESHOLD:
        state_manager.increment_occlusion_iou_frames()
        
        # If IoU has been high for consecutive frames, confirm occlusion
        if state_manager.get_occlusion_iou_frames() >= OCCLUSION_DETECTION_FRAMES:
            if not state_manager.is_occlusion_detected():
                state_manager.set_occlusion_detected(True)
                print(f"🚨 Occlusion detected! Max IoU: {max_iou:.3f}")
            
            # Handle stop announcement
            if not state_manager.is_occlusion_stop_announced():
                voice_announcer.announce_stop_request(OCCLUSION_STOP_TIME)
                state_manager.set_occlusion_stop_announced(True)
            
            return True
    else:
        # Reset IoU frame counter if no occlusion
        state_manager.reset_occlusion_iou_frames()
        
        # Check if occlusion just ended
        if state_manager.is_occlusion_detected():
            state_manager.set_occlusion_detected(False)
            print("✅ Occlusion ended, entering recovery mode")
    
    return state_manager.is_occlusion_detected()

def handle_occlusion_recovery(current_skeletons_data, frame_idx, image_width, image_height):
    """
    Handle recovery after occlusion ends.
    Returns True if master is successfully recovered, False otherwise.
    """
    if not state_manager.is_occlusion_recovery_mode():
        return False
    
    recovery_start_time = state_manager.get_occlusion_recovery_start_time()
    if recovery_start_time is None:
        return False
    
    # Wait for the stop time to pass
    elapsed_time = time.time() - recovery_start_time
    if elapsed_time < OCCLUSION_STOP_TIME:
        return False
    
    # Now analyze the scene
    tracked_skeletons = state_manager.get_tracked_skeletons()
    
    # Count skeletons within tracking distance
    skeletons_in_range = []
    for s_id, s_data in tracked_skeletons.items():
        bbox = s_data['bbox']
        center = [bbox[0] + bbox[2] // 2, bbox[1] + bbox[3] // 2]
        
        # Calculate distance from image center (assuming robot is at center)
        image_center = [image_width // 2, image_height // 2]
        distance = np.sqrt((center[0] - image_center[0])**2 + (center[1] - image_center[1])**2)
        
        if distance <= DISTANCE_THRESHOLD_FOR_TRACKING:
            skeletons_in_range.append(s_id)
    
    print(f"🔍 Recovery analysis: {len(skeletons_in_range)} skeletons in range")
    
    if len(skeletons_in_range) == 0:
        # No skeletons visible
        voice_announcer.announce_cant_see()
        print("❌ No skeletons visible, asking master to come in front")
        return False
    
    elif len(skeletons_in_range) == 1:
        # Only one skeleton - likely the master
        master_candidate_id = skeletons_in_range[0]
        state_manager.set_master_id(master_candidate_id)
        tracked_skeletons[master_candidate_id]['is_master'] = True
        state_manager.set_tracked_skeletons(tracked_skeletons)
        state_manager.set_occlusion_recovery_mode(False)
        print(f"✅ Master recovered: ID {master_candidate_id} (single skeleton)")
        return True
    
    else:
        # Multiple skeletons - need face recognition
        voice_announcer.announce_look_request()
        print("👥 Multiple skeletons detected, requesting master to look at robot")
        
        # Try to identify master by face recognition
        # Note: We need the image data for face recognition, but we only have skeleton data
        # For now, we'll use a simplified approach
        from master_calibration import attempt_master_reidentification_from_skeletons
        if attempt_master_reidentification_from_skeletons(current_skeletons_data, image_width, image_height):
            state_manager.set_occlusion_recovery_mode(False)
            print("✅ Master recovered through face recognition")
            return True
        else:
            print("❌ Face recognition failed, continuing recovery mode")
            return False

def get_skeletons_in_tracking_range(tracked_skeletons, image_width, image_height):
    """
    Get all skeletons within the tracking distance threshold.
    Returns list of skeleton IDs.
    """
    skeletons_in_range = []
    image_center = [image_width // 2, image_height // 2]
    
    for s_id, s_data in tracked_skeletons.items():
        bbox = s_data['bbox']
        center = [bbox[0] + bbox[2] // 2, bbox[1] + bbox[3] // 2]
        
        distance = np.sqrt((center[0] - image_center[0])**2 + (center[1] - image_center[1])**2)
        
        if distance <= DISTANCE_THRESHOLD_FOR_TRACKING:
            skeletons_in_range.append(s_id)
    
    return skeletons_in_range

def bind_face_to_skeleton(face_bbox, face_center, tracked_skeletons, image_width, image_height):
    """
    Try to bind a detected face to a skeleton based on spatial relationship.
    Returns (best_skeleton_id, best_score) or (-1, -1) if no suitable match found.
    """
    best_skeleton_id = -1
    best_skeleton_score = -1
    
    for s_id, s_data in tracked_skeletons.items():
        if not s_data['is_master']: # Ensure not already marked master
            skeleton_bbox = s_data['bbox']
            skeleton_center = [skeleton_bbox[0] + skeleton_bbox[2] // 2, 
                              skeleton_bbox[1] + skeleton_bbox[3] // 2]
            
            # Calculate multiple matching criteria
            iou = calculate_iou(face_bbox, skeleton_bbox)
            
            # Calculate distance between face center and skeleton center
            center_distance = np.sqrt((face_center[0] - skeleton_center[0])**2 + 
                                    (face_center[1] - skeleton_center[1])**2)
            
            # Calculate relative distance (normalized by image size)
            relative_distance = center_distance / np.sqrt(image_width**2 + image_height**2)
            
            # Combined score: prioritize IoU but also consider center distance
            # For distant faces, IoU might be low but center distance should be reasonable
            if iou > REID_IOU_THRESHOLD:  # Very low IoU threshold for distant faces
                score = iou * 0.7 + (1.0 - relative_distance) * 0.3
            else:
                # If IoU is too low, only consider center distance
                score = (1.0 - relative_distance) * 0.5
            
            if score > best_skeleton_score:
                best_skeleton_score = score
                best_skeleton_id = s_id
    
    return best_skeleton_id, best_skeleton_score

# Legacy function for compatibility
def detect_occlusion(current_skeletons_data, frame_idx, image_width, image_height):
    """
    Legacy occlusion detection function - now uses IoU-based detection.
    Returns True if occlusion is detected, False otherwise.
    """
    return detect_iou_occlusion(current_skeletons_data, frame_idx, image_width, image_height) 