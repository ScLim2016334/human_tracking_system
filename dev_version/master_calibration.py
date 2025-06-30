"""
Master Calibration Module
Handles master face recording and skeleton calibration
"""

from config import *
from state_manager import state_manager
from pose_utils import is_hands_on_hips
from face_recognition_utils import detect_faces_in_image, recognize_master_face

def record_master_faces(image_rgb):
    """
    Record master face encodings from the current image.
    Returns the number of faces recorded.
    """
    face_locations, face_encodings = detect_faces_in_image(image_rgb)
    
    total_recorded = 0
    for face_encoding in face_encodings:
        state_manager.add_master_face_encoding(face_encoding)
        total_recorded += 1
        print(f"--- Recorded {len(state_manager.get_master_face_encodings())}th master face information ---")
        print(f"Face encoding: {face_encoding[:5]}...") # Print first few digits of encoding
    
    if total_recorded > 0:
        print(f"Currently recorded {len(state_manager.get_master_face_encodings())} face information")
    else:
        print("No face detected, cannot record face information. Please ensure you are facing the camera.")
    
    return total_recorded

def calibrate_master_skeleton(image_rgb, frame_count):
    """
    Try to calibrate master skeleton through hands-on-hips gesture and face recognition.
    Returns True if calibration is successful, False otherwise.
    """
    tracked_skeletons = state_manager.get_tracked_skeletons()
    
    for s_id, s_data in tracked_skeletons.items():
        # Only consider skeletons just detected in current frame
        if s_data['last_seen_frame'] == frame_count:
            landmarks_for_action = s_data['last_pose'].pose_landmarks
            if is_hands_on_hips(landmarks_for_action):
                state_manager.add_to_master_pose_buffer(True)
            else:
                state_manager.add_to_master_pose_buffer(False)
            
            # If hands-on-hips action for consecutive frames
            pose_buffer = state_manager.get_master_pose_buffer()
            if sum(pose_buffer) == MASTER_HISTORY_THRESHOLD:
                # Check if this person's face matches any recorded master faces
                # Detect faces in entire image (not just skeleton bbox)
                face_locations, face_encodings = detect_faces_in_image(image_rgb)
                
                if face_encodings:
                    # Check if any face in the image matches master
                    master_face_encodings = state_manager.get_master_face_encodings()
                    is_master_face, best_distance = recognize_master_face(face_encodings, master_face_encodings)
                    
                    if is_master_face:
                        # Set this skeleton as master
                        state_manager.set_master_id(s_id)
                        tracked_skeletons[s_id]['is_master'] = True
                        state_manager.set_tracked_skeletons(tracked_skeletons)
                        state_manager.set_master_skeleton_recorded(True)
                        
                        # Initialize last known position
                        master_bbox = tracked_skeletons[s_id]['bbox']
                        master_center = [master_bbox[0] + master_bbox[2] // 2, 
                                        master_bbox[1] + master_bbox[3] // 2]
                        state_manager.set_master_last_known_position(master_center)
                        
                        print(f"--- Master skeleton {s_id} calibrated through hands-on-hips action and face recognition! ---")
                        print(f"Face match distance: {best_distance:.3f}")
                        state_manager.clear_master_pose_buffer() # Clear cache to avoid repeated triggering
                        return True
                    else:
                        print("Warning: Hands-on-hips action detected but face doesn't match recorded master faces.")
                        print(f"Best face distance: {best_distance:.3f} (threshold: {FACE_RECOGNITION_TOLERANCE})")
                        state_manager.clear_master_pose_buffer() # Reset cache if face doesn't match
                else:
                    print("Warning: Hands-on-hips action detected but no face found in the image.")
                    state_manager.clear_master_pose_buffer() # Reset cache if no face found
    
    return False

def attempt_master_reidentification(image_rgb, image_width, image_height):
    """
    Attempt to re-identify master through face recognition when skeleton is lost.
    Returns True if re-identification is successful, False otherwise.
    """
    master_id = state_manager.get_master_id()
    master_face_encodings = state_manager.get_master_face_encodings()
    tracked_skeletons = state_manager.get_tracked_skeletons()
    
    if master_id == -1 or not master_face_encodings:
        return False
    
    # Check if master is currently tracked
    is_master_present = False
    if master_id in tracked_skeletons:
        is_master_present = tracked_skeletons[master_id]['is_master']
    
    if is_master_present:
        return False  # Master is already tracked
    
    # Detect faces in entire image for re-identification
    face_locations, face_encodings = detect_faces_in_image(image_rgb)
    
    if not face_encodings:
        return False
    
    # Find the best matching master face in the image
    best_master_face_idx, best_master_distance = find_best_master_face_match(
        face_locations, face_encodings, master_face_encodings
    )
    
    if best_master_face_idx == -1:
        return False
    
    # Found matching face, try to bind it to a skeleton
    top, right, bottom, left = face_locations[best_master_face_idx]
    face_bbox = [left, top, right - left, bottom - top]
    face_center = [left + (right - left) // 2, top + (bottom - top) // 2]
    
    # Try to bind face with a skeleton
    from skeleton_tracker import bind_face_to_skeleton
    best_skeleton_id, best_skeleton_score = bind_face_to_skeleton(
        face_bbox, face_center, tracked_skeletons, image_width, image_height
    )
    
    # If we found a suitable skeleton, bind it
    if best_skeleton_id != -1 and best_skeleton_score > REID_SCORE_THRESHOLD:
        state_manager.set_master_id(best_skeleton_id) # Re-set master ID
        tracked_skeletons[best_skeleton_id]['is_master'] = True
        state_manager.set_tracked_skeletons(tracked_skeletons)
        
        # Update last known position
        master_bbox = tracked_skeletons[best_skeleton_id]['bbox']
        master_center = [master_bbox[0] + master_bbox[2] // 2, 
                        master_bbox[1] + master_bbox[3] // 2]
        state_manager.set_master_last_known_position(master_center)
        
        print(f"--- Master {best_skeleton_id} re-identification successful! ---")
        print(f"Face match distance: {best_master_distance:.3f}")
        print(f"Skeleton binding score: {best_skeleton_score:.3f}")
        return True
    else:
        print(f"Master face detected (distance: {best_master_distance:.3f}) but no suitable skeleton found.")
        print(f"Best skeleton score: {best_skeleton_score:.3f}")
        return False

def attempt_master_reidentification_from_skeletons(current_skeletons_data, image_width, image_height):
    """
    Attempt to re-identify master through face recognition from skeleton data.
    This function is specifically for occlusion recovery scenarios.
    Returns True if re-identification is successful, False otherwise.
    """
    master_face_encodings = state_manager.get_master_face_encodings()
    tracked_skeletons = state_manager.get_tracked_skeletons()
    
    if not master_face_encodings:
        return False
    
    # Detect faces in entire image for re-identification
    # We need to convert the skeleton data back to image format for face detection
    # For now, we'll use the current tracked skeletons and try to find faces near them
    
    # Get skeletons in tracking range
    from skeleton_tracker import get_skeletons_in_tracking_range
    skeletons_in_range = get_skeletons_in_tracking_range(tracked_skeletons, image_width, image_height)
    
    if len(skeletons_in_range) < 2:
        return False  # Need at least 2 skeletons for this to make sense
    
    # For each skeleton in range, try to detect faces in their bounding box
    for s_id in skeletons_in_range:
        s_data = tracked_skeletons[s_id]
        bbox = s_data['bbox']
        
        # Create a dummy image region for face detection (this is a simplified approach)
        # In a real implementation, you would need the actual image data
        # For now, we'll use the existing face detection on the full image
        
        # Try to find faces near this skeleton
        from face_recognition_utils import get_face_encodings_from_bbox
        # Note: This is a placeholder - in reality you'd need the actual image data
        # face_encodings = get_face_encodings_from_bbox(image_rgb, bbox)
        
        # For now, we'll use a simplified approach: check if any face in the image
        # is close to this skeleton's bounding box
        # This is a workaround - in a real implementation you'd need the image data
        
        # Simulate face detection near skeleton
        # This is a placeholder implementation
        return False
    
    return False

def find_best_master_face_match(face_locations, face_encodings, master_face_encodings):
    """
    Find the best matching master face in the image.
    Returns (best_face_idx, best_distance) or (-1, float('inf')) if no match found.
    """
    best_master_face_idx = -1
    best_master_distance = float('inf')
    
    for i, face_encoding in enumerate(face_encodings):
        # Check if this face matches master
        is_master_face, distance = recognize_master_face([face_encoding], master_face_encodings)
        
        if is_master_face and distance < best_master_distance:
            best_master_distance = distance
            best_master_face_idx = i
    
    return best_master_face_idx, best_master_distance 