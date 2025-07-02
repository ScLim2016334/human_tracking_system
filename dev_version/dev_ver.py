import cv2
import mediapipe as mp
import numpy as np
from ultralytics import YOLO # For general object detection (people, faces)
import pyttsx3 # For text-to-speech
import face_recognition # For face recognition and re-identification
import time # For precise timing in occlusion
import threading

# --- MediaPipe Initialization ---
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# --- Text-to-Speech Initialization ---
engine = pyttsx3.init()
# Optional: Adjust speech rate and volume
engine.setProperty('rate', 150) # Speed of speech
engine.setProperty('volume', 0.9) # Volume (0.0 to 1.0)

def announce(text):
    """Speaks the given text in a non-blocking way to prevent UI freezing."""
    print(f"ANNOUNCEMENT: {text}")
    
    def speak():
        try:
            engine.say(text)
            engine.runAndWait()
        except Exception as e:
            print(f"TTS Error: {e}")
    
    # Run TTS in a separate thread to avoid blocking
    tts_thread = threading.Thread(target=speak)
    tts_thread.daemon = True
    tts_thread.start()

def beep():
    """Generates a simple beep sound."""
    # This is a placeholder. For actual sound, you might need a library like 'winsound' on Windows
    # or 'simpleaudio'/'pydub' for cross-platform. For now, it will just print a message.
    print("BEEP!")
    # Example for Windows:
    # import winsound
    # winsound.Beep(1000, 200) # Frequency, Duration
    announce("Beep!") # Announce the beep for demonstration

# --- Global Variables and Configuration Parameters ---
# Skeleton tracker related (for MediaPipe poses)
tracked_skeletons = {}  # {skeleton_id: {'last_pose': [landmarks], 'bbox': [x,y,w,h], 'last_seen_frame': frame_idx, 'is_master': False}}
next_id = 0             # Next available skeleton ID
MAX_DIST_THRESHOLD = 80 # Skeleton center point distance threshold for ID matching (pixels, adjust based on actual conditions)
MAX_MISS_FRAMES = 15    # Remove ID after skeleton disappears for how many frames, prevent ID accumulation

# Master calibration related
MASTER_ID = -1          # Current master skeleton ID (-1 means not set)
MASTER_FACE_ENCODINGS = [] # Store multiple master face encodings for better recognition
MASTER_POSE_BUFFER = [] # Store master calibration action history frame results
MASTER_HISTORY_THRESHOLD = 5 # At least 5 consecutive frames of calibration action to be considered successful
MASTER_FACE_RECORDED = False # Whether master face has been recorded
MASTER_SKELETON_RECORDED = False # Whether master skeleton has been recorded

# Occlusion handling related (IoU-based)
OCCLUSION_IOU_THRESHOLD = 0.4 # IoU threshold to consider an occlusion (lowered for more sensitive detection)
OCCLUSION_STOP_SECONDS = 3 # Time in seconds to stop and wait after occlusion (reduced for faster debugging)
WAIT_AFTER_NO_DETECTION_SECONDS = 3 # Time to wait when no one is detected before trying face recognition
OCCLUSION_START_TIME = None # Timestamp when occlusion started
WAIT_START_TIME = None # Timestamp when waiting period started (for no detection scenario)

# Original occlusion handling for tracking (from human_tracking.py)
MASTER_LAST_KNOWN_POSITION = None
MASTER_OCCLUSION_FRAMES = 0
MAX_OCCLUSION_FRAMES = 30
OCCLUSION_DISTANCE_THRESHOLD = 150

# Detection thresholds
YOLO_PERSON_CONFIDENCE_THRESHOLD = 0.5 # Minimum confidence for YOLO to detect a person
FACE_RECOGNITION_TOLERANCE = 0.3 # Face recognition tolerance, smaller value = stricter
MIN_PERSON_AREA_THRESHOLD = 0.01 # Minimum relative area (bbox_area / image_area) for a person to be considered significant

# State variables
current_state = "STEP1_FACE_RECORDING"  # STEP1_FACE_RECORDING, STEP2_SKELETON_RECORDING, STEP3_TRACKING, OCCLUSION_WAIT, RECOVERY_WAIT
occlusion_detected = False # This can be local to STEP3 now or unused.
recovery_mode = ""  # "SINGLE_PERSON", "MULTIPLE_PEOPLE", "NO_PEOPLE"

# STEP1 performance optimization variables
STEP1_FACE_DETECTION_INTERVAL = 3   # Only detect faces every N frames in STEP1
step1_last_face_detection_frame = 0
step1_cached_face_data = {'locations': [], 'encodings': []}
STEP1_CAPTURE_INTERVAL = 3 # Seconds between face captures
STEP1_LAST_CAPTURE_TIME = 0
STEP1_CAPTURE_COUNT = 0
STEP1_TOTAL_CAPTURES = 3 # Left, Front, Right

# STEP3 master direction detection variables
master_last_center_x = None   # Track master's last horizontal position
master_disappeared_direction = None   # "LEFT", "RIGHT", or None

# --- Helper Functions ---

def get_bbox_from_landmarks(landmarks, image_width, image_height):
    """Calculate skeleton bounding box from MediaPipe keypoints."""
    if not landmarks:
        return [0, 0, 0, 0] # Return an invalid bbox
    x_coords = [lm.x * image_width for lm in landmarks.landmark if lm.visibility > 0.5]
    y_coords = [lm.y * image_height for lm in landmarks.landmark if lm.visibility > 0.5]

    if not x_coords or not y_coords:
        return [0, 0, 0, 0]

    min_x, max_x = int(min(x_coords)), int(max(x_coords))
    min_y, max_y = int(min(y_coords)), int(max(y_coords))

    # Add padding
    padding = 20
    min_x = max(0, min_x - padding)
    min_y = max(0, min_y - padding)
    max_x = min(image_width - 1, max_x + padding)
    max_y = min(image_height - 1, max_y + padding)

    return [min_x, min_y, max_x - min_x, max_y - min_y]

def get_person_detections(yolo_model, image, confidence_threshold=0.3):
    """Get YOLO person detections with specified confidence threshold."""
    yolo_results = yolo_model(image, verbose=False, conf=confidence_threshold)
    person_detections = []
    
    for r in yolo_results:
        for box in r.boxes:
            if yolo_model.names[int(box.cls[0])] == 'person':
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                person_detections.append({
                    'bbox': [x1, y1, x2 - x1, y2 - y1],
                    'conf': float(box.conf[0])
                })
    
    return person_detections

def filter_significant_persons(person_detections, image_width, image_height, min_area_threshold):
    """Filter person detections to only include those with significant size."""
    significant_persons = []
    image_area = image_width * image_height
    
    for p_det in person_detections:
        bbox = p_det['bbox']
        person_area = bbox[2] * bbox[3]
        if person_area / image_area > min_area_threshold:
            significant_persons.append(p_det)
    
    return significant_persons

def update_skeleton_ids(current_skeletons_data, frame_idx, image_width, image_height):
    """Update and maintain skeleton IDs."""
    global next_id, tracked_skeletons, MASTER_ID

    new_tracked_skeletons = {}
    # matched_current_indices = set() # This variable is not used after setting. Can remove.

    for current_idx, (landmarks, pose_results) in enumerate(current_skeletons_data):
        current_center = np.mean([[lm.x, lm.y] for lm in landmarks.landmark], axis=0) * np.array([image_width, image_height])
        current_bbox = get_bbox_from_landmarks(landmarks, image_width, image_height)

        best_match_id = -1
        min_dist = float('inf')

        for s_id, s_data in tracked_skeletons.items():
            last_center = np.mean([[lm.x, lm.y] for lm in s_data['last_pose'].pose_landmarks.landmark], axis=0) * np.array([image_width, image_height])
            dist = np.linalg.norm(current_center - last_center)
            
            if dist < min_dist and dist < MAX_DIST_THRESHOLD:
                min_dist = dist
                best_match_id = s_id
        
        if best_match_id != -1 and best_match_id not in new_tracked_skeletons:
            # Re-confirm master if a match is found near the last known position.
            # This logic was part of old `detect_occlusion` and caused issues.
            # Let's trust the ID tracking for master status for now.
            # The OCCLUSION_DISTANCE_THRESHOLD check here is for preventing re-assigning
            # a master ID to a seemingly 'new' person too far from where master was last seen.
            if best_match_id == MASTER_ID:
                if MASTER_LAST_KNOWN_POSITION is not None:
                    distance_to_last_known = np.linalg.norm(current_center - np.array(MASTER_LAST_KNOWN_POSITION))
                    if distance_to_last_known > OCCLUSION_DISTANCE_THRESHOLD:
                        print(f"Potential new person too far from master's last position: {distance_to_last_known:.1f}px. Not re-assigning master.")
                        best_match_id = -1 # Treat as a new person, not the returning master
                    # else: # Master reappeared close by. No need to print this frequently.
                        # print(f"Master reappeared! Distance from last position: {distance_to_last_known:.1f}px")
            
            if best_match_id != -1:
                new_tracked_skeletons[best_match_id] = {
                    'last_pose': pose_results,
                    'bbox': current_bbox,
                    'last_seen_frame': frame_idx,
                    'is_master': tracked_skeletons[best_match_id]['is_master'],
                }
                # matched_current_indices.add(current_idx) # Not used
            else: # No good match found, assign new ID
                new_tracked_skeletons[next_id] = {
                    'last_pose': pose_results,
                    'bbox': current_bbox,
                    'last_seen_frame': frame_idx,
                    'is_master': False,
                }
                next_id += 1
                # matched_current_indices.add(current_idx) # Not used
        else: # No good match found or match already taken, assign new ID
            new_tracked_skeletons[next_id] = {
                'last_pose': pose_results,
                'bbox': current_bbox,
                'last_seen_frame': frame_idx,
                'is_master': False,
            }
            next_id += 1
            # matched_current_indices.add(current_idx) # Not used

    temp_tracked_skeletons = {}
    for s_id, s_data in tracked_skeletons.items(): # Keep old tracked skeletons that are still within MAX_MISS_FRAMES
        if frame_idx - s_data['last_seen_frame'] <= MAX_MISS_FRAMES:
            temp_tracked_skeletons[s_id] = s_data

    # Add newly found/matched skeletons
    for s_id, s_data in new_tracked_skeletons.items():
        temp_tracked_skeletons[s_id] = s_data # Overwrite if existing, add if new

    # Check if master ID was in old tracked_skeletons but not in new_tracked_skeletons (meaning it wasn't seen this frame)
    # And if it has now exceeded MAX_MISS_FRAMES.
    if MASTER_ID != -1 and MASTER_ID in tracked_skeletons and MASTER_ID not in new_tracked_skeletons:
        if frame_idx - tracked_skeletons[MASTER_ID]['last_seen_frame'] > MAX_MISS_FRAMES:
            print(f"Master ID {MASTER_ID} skeleton not seen for {frame_idx - tracked_skeletons[MASTER_ID]['last_seen_frame']} frames, resetting master status.")
            # Do NOT reset MASTER_ID here immediately, let main loop handle full loss/disappearance
            # Just ensure 'is_master' is False if it's truly lost.
            if MASTER_ID in temp_tracked_skeletons: # If it made it past MAX_MISS_FRAMES cleanup
                temp_tracked_skeletons[MASTER_ID]['is_master'] = False # Mark as no longer master if lost beyond threshold

    tracked_skeletons = temp_tracked_skeletons


def is_hands_on_hips(landmarks):
    """Determine if MediaPipe keypoints represent 'hands on hips' action."""
    left_wrist = landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST]
    right_wrist = landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST]
    left_hip = landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP]
    right_hip = landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP]
    left_elbow = landmarks.landmark[mp_pose.PoseLandmark.LEFT_ELBOW]
    right_elbow = landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ELBOW]
    left_shoulder = landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER]
    right_shoulder = landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER]

    if not (left_wrist.visibility > 0.7 and right_wrist.visibility > 0.7 and
            left_hip.visibility > 0.7 and right_hip.visibility > 0.7 and
            left_elbow.visibility > 0.7 and right_elbow.visibility > 0.7):
        return False

    wrist_hip_y_threshold_rel = 0.08 
    wrist_hip_x_threshold_rel = 0.05 

    left_hand_on_hip = abs(left_wrist.y - left_hip.y) < wrist_hip_y_threshold_rel and \
                        abs(left_wrist.x - left_hip.x) < wrist_hip_x_threshold_rel

    right_hand_on_hip = abs(right_wrist.y - right_hip.y) < wrist_hip_y_threshold_rel and \
                         abs(right_wrist.x - right_hip.x) < wrist_hip_x_threshold_rel

    def calculate_angle(a, b, c):
        a_coords = np.array([a.x, a.y])
        b_coords = np.array([b.x, b.y])
        c_coords = np.array([c.x, c.y])
        
        ba = a_coords - b_coords
        bc = c_coords - b_coords
        
        cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
        cosine_angle = np.clip(cosine_angle, -1.0, 1.0)
        angle = np.degrees(np.arccos(cosine_angle))
        return angle

    left_elbow_angle = calculate_angle(left_shoulder, left_elbow, left_wrist)
    right_elbow_angle = calculate_angle(right_shoulder, right_elbow, right_wrist)

    elbow_angle_threshold_min = 60
    elbow_angle_threshold_max = 120

    left_elbow_ok = elbow_angle_threshold_min < left_elbow_angle < elbow_angle_threshold_max
    right_elbow_ok = elbow_angle_threshold_min < right_elbow_angle < elbow_angle_threshold_max
    
    return (left_hand_on_hip and left_elbow_ok) or (right_hand_on_hip and right_elbow_ok)

def calculate_iou(boxA, boxB):
    """Calculate IoU (Intersection over Union) of two bounding boxes"""
    if boxA[2] <= 0 or boxA[3] <= 0 or boxB[2] <= 0 or boxB[3] <= 0:
        return 0.0

    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[0] + boxA[2], boxB[0] + boxB[2])
    yB = min(boxA[1] + boxA[3], boxB[1] + boxB[3])

    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = boxA[2] * boxA[3]
    boxBArea = boxB[2] * boxB[3]
    unionArea = float(boxAArea + boxBArea - interArea)
    
    if unionArea == 0:
        return 0.0

    return interArea / unionArea

def recognize_master_face(face_encodings):
    """Recognize if any face encodings match the master."""
    if not MASTER_FACE_ENCODINGS:
        return False, float('inf')

    best_distance = float('inf')
    is_master = False

    for face_encoding in face_encodings:
        distances = []
        matches = []

        for master_encoding in MASTER_FACE_ENCODINGS:
            distance = face_recognition.face_distance([master_encoding], face_encoding)[0]
            distances.append(distance)
            matches.append(distance <= FACE_RECOGNITION_TOLERANCE)

        min_distance = min(distances)
        if any(matches) and min_distance < best_distance:
            best_distance = min_distance
            is_master = True

    return is_master, best_distance

def detect_occlusion(current_skeletons_data, frame_idx, image_width, image_height):
    """
    Manages MASTER_LAST_KNOWN_POSITION and MASTER_OCCLUSION_FRAMES.
    Returns True if master is NOT tracked by MediaPipe but a skeleton is nearby.
    Resets MASTER_LAST_KNOWN_POSITION to None if master is lost for too long.
    """
    global MASTER_LAST_KNOWN_POSITION, MASTER_OCCLUSION_FRAMES, MASTER_ID
    
    if MASTER_ID == -1:
        return False # No master to track for occlusion logic
    
    master_currently_tracked_by_mediapipe = (MASTER_ID in tracked_skeletons and tracked_skeletons[MASTER_ID]['is_master'])
    
    if master_currently_tracked_by_mediapipe:
        master_current_position = [tracked_skeletons[MASTER_ID]['bbox'][0] + tracked_skeletons[MASTER_ID]['bbox'][2] // 2, 
                                   tracked_skeletons[MASTER_ID]['bbox'][1] + tracked_skeletons[MASTER_ID]['bbox'][3] // 2]
        MASTER_LAST_KNOWN_POSITION = master_current_position
        MASTER_OCCLUSION_FRAMES = 0
        return False # Master is visible, not "occluded" by this definition
    else:
        # Master is not currently tracked by MediaPipe.
        MASTER_OCCLUSION_FRAMES += 1
        
        if MASTER_OCCLUSION_FRAMES > MAX_OCCLUSION_FRAMES:
            # Master lost for too long, reset last known position.
            if MASTER_LAST_KNOWN_POSITION is not None:
                print(f"Master lost for {MASTER_OCCLUSION_FRAMES} frames, resetting last known position.")
            MASTER_LAST_KNOWN_POSITION = None
            MASTER_OCCLUSION_FRAMES = 0 # Reset frame count for next detection
            return False # Master is considered fully lost, not just occluded by another nearby person.
        
        # If not fully lost, check if another skeleton is nearby, implying potential occlusion.
        if MASTER_LAST_KNOWN_POSITION is not None:
            for landmarks, pose_results in current_skeletons_data:
                current_center = np.mean([[lm.x, lm.y] for lm in landmarks.landmark], axis=0) * np.array([image_width, image_height])
                distance_to_master = np.linalg.norm(current_center - np.array(MASTER_LAST_KNOWN_POSITION))
                
                if distance_to_master < OCCLUSION_DISTANCE_THRESHOLD:
                    # Another skeleton is close to where the master was. This is a potential occlusion.
                    print(f"Master not seen, but potential occluder detected nearby! Distance: {distance_to_master:.1f}px")
                    return True # Indicate potential short-term occlusion by another person
        
        return False # Master not tracked, no nearby skeleton, and not yet over MAX_OCCLUSION_FRAMES (just momentarily lost or left frame)


def detect_master_disappearance_direction(image_width):
    """Detect which direction the master disappeared from (LEFT or RIGHT) based on last known position."""
    global master_last_center_x, master_disappeared_direction
    
    if master_last_center_x is None:
        return None
    
    # Define screen zones for direction detection
    screen_center = image_width / 2
    left_zone_threshold = screen_center * 0.3  # Closer to edge for disappearance
    right_zone_threshold = screen_center * 1.7  # Closer to edge for disappearance
    
    if master_last_center_x < left_zone_threshold:
        master_disappeared_direction = "LEFT"
        return "LEFT"
    elif master_last_center_x > right_zone_threshold:
        master_disappeared_direction = "RIGHT"
        return "RIGHT"
    else:
        # Master was in the center area, hard to say direction definitively.
        # Could imply they moved directly away or towards camera, or fell.
        master_disappeared_direction = "CENTER"
        return "CENTER"

def update_master_position_tracking(master_bbox):
    """Update master's position for direction tracking (STEP3 only)."""
    global master_last_center_x
    
    if master_bbox and len(master_bbox) >= 4:
        # Calculate center x position of master's bounding box
        master_last_center_x = master_bbox[0] + master_bbox[2] / 2

# --- Main Program Logic ---
def main():
    global MASTER_ID, MASTER_FACE_ENCODINGS, MASTER_POSE_BUFFER, next_id, tracked_skeletons
    global MASTER_FACE_RECORDED, MASTER_SKELETON_RECORDED, MASTER_LAST_KNOWN_POSITION, MASTER_OCCLUSION_FRAMES
    global current_state, OCCLUSION_START_TIME, WAIT_START_TIME, occlusion_detected, recovery_mode
    global master_last_center_x, master_disappeared_direction
    global STEP1_LAST_CAPTURE_TIME, STEP1_CAPTURE_COUNT, STEP1_TOTAL_CAPTURES

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Cannot open camera.")
        announce("Error: Cannot open camera.")
        return

    ret, frame = cap.read()
    if not ret:
        print("Error: Cannot read frame from camera.")
        cap.release()
        return
    
    image_height, image_width, _ = frame.shape
    print(f"Camera resolution: {image_width}x{image_height}")

    print("\n--- Human Following Algorithm with Enhanced IoU-based Occlusion Detection ---")
    announce("Starting human following system.")
    print("=== STEP 1: Record Master Face ===")
    announce("I will now capture your face. Please look to your left. Then straight. Then to your right. I will beep before each capture.")
    print("1. Please look to your left, then straight, then to your right when prompted by the system.")
    print("2. Listen for the beep before each capture.")
    print("3. Press 'r' to reset face records if needed.")

    # YOLO model will be loaded only when needed in Step 3
    yolo_model = None

    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        frame_count = 0
        step2_introduced = False
        
        # FPS monitoring for performance validation
        fps_start_time = time.time()
        fps_frame_count = 0
        current_fps = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1
            
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image_rgb.flags.writeable = False
            
            results = pose.process(image_rgb)
            
            image_rgb.flags.writeable = True
            image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

            current_skeletons_in_frame_data = []
            if results.pose_landmarks:
                current_skeletons_in_frame_data.append((results.pose_landmarks, results))

            # Update skeleton IDs (MediaPipe tracking)
            update_skeleton_ids(current_skeletons_in_frame_data, frame_count, image_width, image_height)
            
            # --- State Machine Logic ---
            if current_state == "STEP1_FACE_RECORDING":
                # STEP 1: Record master face (close to camera) - Optimized and Automated
                global step1_last_face_detection_frame, step1_cached_face_data
                
                if frame_count - step1_last_face_detection_frame >= STEP1_FACE_DETECTION_INTERVAL:
                    face_locations = face_recognition.face_locations(image_rgb)
                    face_encodings = face_recognition.face_encodings(image_rgb, face_locations)
                    step1_cached_face_data = {
                        'locations': face_locations,
                        'encodings': face_encodings
                    }
                    step1_last_face_detection_frame = frame_count

                current_time = time.time()
                if STEP1_CAPTURE_COUNT < STEP1_TOTAL_CAPTURES:
                    if current_time - STEP1_LAST_CAPTURE_TIME >= STEP1_CAPTURE_INTERVAL:
                        beep()
                        face_encodings_to_record = step1_cached_face_data['encodings']
                        
                        total_recorded_this_shot = 0
                        for face_encoding in face_encodings_to_record:
                            MASTER_FACE_ENCODINGS.append(face_encoding)
                            total_recorded_this_shot += 1
                            print(f"--- Recorded {len(MASTER_FACE_ENCODINGS)}th master face information ---")
                        
                        if total_recorded_this_shot > 0:
                            STEP1_CAPTURE_COUNT += 1
                            print(f"Captured {total_recorded_this_shot} face(s) for shot {STEP1_CAPTURE_COUNT}/{STEP1_TOTAL_CAPTURES}")
                            if STEP1_CAPTURE_COUNT == 1:
                                announce("Please turn your head to face forward.")
                            elif STEP1_CAPTURE_COUNT == 2:
                                announce("Please turn your head to face your right.")
                            elif STEP1_CAPTURE_COUNT == 3:
                                announce("All face captures complete.")
                        else:
                            print("No face detected for current capture. Please ensure you are facing the camera.")
                            announce("No face detected. Please try again.")

                        STEP1_LAST_CAPTURE_TIME = current_time
                
                if STEP1_CAPTURE_COUNT >= STEP1_TOTAL_CAPTURES and not MASTER_FACE_RECORDED:
                    if len(MASTER_FACE_ENCODINGS) > 0:
                        current_state = "STEP2_SKELETON_RECORDING"
                        MASTER_FACE_RECORDED = True
                        print(f"\n=== Proceeding to Step 2: Skeleton Recording ===")
                        print(f"Face recording complete! {len(MASTER_FACE_ENCODINGS)} faces recorded.")
                        print("Now move away from camera and make yourself visible.")
                        announce("Face recording complete. Now move away and make yourself visible to be selected as master.")
                    else:
                        print("No faces were successfully recorded. Please reset and try again.")
                        announce("No faces were recorded. Please reset the system to try again.")
                        STEP1_CAPTURE_COUNT = 0 # Allow retrying if no faces were recorded

                cv2.putText(image_bgr, "STEP 1: Record Master Face", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)
                cv2.putText(image_bgr, f"Recorded faces: {len(MASTER_FACE_ENCODINGS)}", (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 1, cv2.LINE_AA)
                cv2.putText(image_bgr, f"Captures remaining: {STEP1_TOTAL_CAPTURES - STEP1_CAPTURE_COUNT}", (10, 90),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 1, cv2.LINE_AA)
                cv2.putText(image_bgr, f"Next capture in: {max(0, STEP1_CAPTURE_INTERVAL - (current_time - STEP1_LAST_CAPTURE_TIME)):.1f}s", (10, 120),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1, cv2.LINE_AA)
                
                if STEP1_CAPTURE_COUNT == 0:
                    instruction_text = "Look Left for next capture"
                elif STEP1_CAPTURE_COUNT == 1:
                    instruction_text = "Look Forward for next capture"
                elif STEP1_CAPTURE_COUNT == 2:
                    instruction_text = "Look Right for next capture"
                else:
                    instruction_text = "Captures complete. Moving to next step soon."
                cv2.putText(image_bgr, instruction_text, (10, 150),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 100, 0), 2, cv2.LINE_AA)
                
                for (top, right, bottom, left) in step1_cached_face_data['locations']:
                    cv2.rectangle(image_bgr, (left, top), (right, bottom), (0, 255, 0), 2)
                    cv2.putText(image_bgr, "Detected Face", (left, top - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)
                
                if tracked_skeletons:
                    for s_id, s_data in tracked_skeletons.items():
                        color = (128, 128, 128)
                        label = f"ID: {s_id} (Not Master Yet)"
                        
                        mp_drawing.draw_landmarks(image_bgr, s_data['last_pose'].pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                                 mp_drawing.DrawingSpec(color=(128,128,128), thickness=2, circle_radius=2),
                                                 mp_drawing.DrawingSpec(color=(128,128,128), thickness=2, circle_radius=2))

                        x, y, w, h = s_data['bbox']
                        cv2.rectangle(image_bgr, (x, y), (x + w, y + h), color, 2)
                        cv2.putText(image_bgr, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

            elif current_state == "STEP2_SKELETON_RECORDING":
                # STEP 2: Record master skeleton - Automatically identify closest skeleton
                if not step2_introduced:
                    print("\n=== STEP 2: Select Master Skeleton (Automatic) ===")
                    print("1. Ensure only the person to be tracked is clearly visible in the camera.")
                    print("2. The system will automatically select the closest skeleton as the master.")
                    step2_introduced = True
                
                cv2.putText(image_bgr, "STEP 2: Selecting Master Skeleton (Closest Person)", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2, cv2.LINE_AA)
                cv2.putText(image_bgr, f"Recorded faces: {len(MASTER_FACE_ENCODINGS)}", (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 1, cv2.LINE_AA)
                cv2.putText(image_bgr, "Ensure only master is visible.", (10, 90),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 1, cv2.LINE_AA)
                
                if current_skeletons_in_frame_data:
                    closest_skeleton_id = -1
                    max_bbox_area = 0

                    for s_id, s_data in tracked_skeletons.items():
                        if s_data['last_seen_frame'] == frame_count:
                            bbox = s_data['bbox']
                            bbox_area = bbox[2] * bbox[3]
                            
                            if bbox_area > max_bbox_area:
                                max_bbox_area = bbox_area
                                closest_skeleton_id = s_id

                    if closest_skeleton_id != -1:
                        if not MASTER_SKELETON_RECORDED:
                            MASTER_ID = closest_skeleton_id
                            tracked_skeletons[MASTER_ID]['is_master'] = True
                            MASTER_SKELETON_RECORDED = True
                            
                            master_bbox = tracked_skeletons[MASTER_ID]['bbox']
                            MASTER_LAST_KNOWN_POSITION = [master_bbox[0] + master_bbox[2] // 2, 
                                                          master_bbox[1] + master_bbox[3] // 2]
                            
                            current_state = "STEP3_TRACKING"
                            announce("Master skeleton selected! Starting enhanced tracking mode.")
                            print(f"--- Master skeleton {MASTER_ID} automatically selected! ---")
                        else:
                            if MASTER_ID not in tracked_skeletons or not tracked_skeletons[MASTER_ID]['is_master']:
                                MASTER_ID = closest_skeleton_id
                                tracked_skeletons[MASTER_ID]['is_master'] = True
                                master_bbox = tracked_skeletons[MASTER_ID]['bbox']
                                MASTER_LAST_KNOWN_POSITION = [master_bbox[0] + master_bbox[2] // 2, 
                                                              master_bbox[1] + master_bbox[3] // 2]
                                announce("Master re-selected. Resuming tracking.")
                                print(f"--- Master skeleton re-selected to {MASTER_ID} ---")
                            current_state = "STEP3_TRACKING"

                for s_id, s_data in tracked_skeletons.items():
                    color = (0, 255, 0)
                    label = f"ID: {s_id}"
                    
                    if s_id == MASTER_ID and s_data['is_master']:
                        color = (0, 0, 255)
                        label = f"Master ID: {s_id}"
                    
                    mp_drawing.draw_landmarks(image_bgr, s_data['last_pose'].pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                             mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2),
                                             mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2))

                    x, y, w, h = s_data['bbox']
                    cv2.rectangle(image_bgr, (x, y), (x + w, y + h), color, 2)
                    cv2.putText(image_bgr, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

            elif current_state == "STEP3_TRACKING":
                if yolo_model is None:
                    try:
                        print("Loading YOLO model for enhanced tracking mode...")
                        yolo_model = YOLO('yolov8n.pt')
                        print("YOLO model loaded successfully!")
                    except Exception as e:
                        print(f"Error loading YOLO model: {e}")
                        announce("Error: Could not load object detection model.")
                        current_state = "STEP2_SKELETON_RECORDING"
                        continue

                cv2.putText(image_bgr, "STEP 3: Enhanced Tracking (MediaPipe + YOLO)", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

                # --- Core logic for occlusion vs. disappearance prioritization ---

                # 1. Update MediaPipe's internal master loss counters and last known position
                # This function now purely manages MASTER_LAST_KNOWN_POSITION and MASTER_OCCLUSION_FRAMES
                # and returns True if master is lost but another skeleton is nearby.
                is_master_mediapipe_lost_and_another_skeleton_nearby = detect_occlusion(current_skeletons_in_frame_data, frame_count, image_width, image_height)

                # 2. Get YOLO person detections
                person_detections = get_person_detections(yolo_model, image_bgr, YOLO_PERSON_CONFIDENCE_THRESHOLD)
                significant_persons = filter_significant_persons(person_detections, image_width, image_height, MIN_PERSON_AREA_THRESHOLD)

                # 3. Determine if master skeleton is currently tracked by MediaPipe
                master_is_tracked_by_mediapipe_this_frame = (MASTER_ID != -1 and MASTER_ID in tracked_skeletons and tracked_skeletons[MASTER_ID]['is_master'])

                # Store previous master visibility for disappearance detection
                previous_master_was_tracked = (master_last_center_x is not None)

                # --- Scenario A: Master skeleton IS currently tracked by MediaPipe ---
                if master_is_tracked_by_mediapipe_this_frame:
                    master_skel_bbox = tracked_skeletons[MASTER_ID]['bbox']
                    update_master_position_tracking(master_skel_bbox) # Update master's x position
                    master_disappeared_direction = None # Master is visible, so reset any disappearance flag

                    # Check for IoU-based occlusion with *other* people while master is visible
                    max_iou_with_other_person = 0.0
                    master_yolo_bbox_for_iou_check = None
                    
                    # Try to find the YOLO bbox that strongly corresponds to the master skeleton
                    best_iou_to_master_skel_bbox = 0.0
                    for yolo_p_det in significant_persons:
                        iou_val = calculate_iou(master_skel_bbox, yolo_p_det['bbox'])
                        if iou_val > best_iou_to_master_skel_bbox and iou_val > 0.3: # Require a decent overlap
                            best_iou_to_master_skel_bbox = iou_val
                            master_yolo_bbox_for_iou_check = yolo_p_det['bbox']

                    if master_yolo_bbox_for_iou_check:
                        # Now, check for occlusion by other people against this master's YOLO bbox
                        for p_det_other in significant_persons:
                            # Ensure we are not comparing the master's YOLO bbox to itself
                            if p_det_other['bbox'] != master_yolo_bbox_for_iou_check:
                                iou_val_with_other = calculate_iou(master_yolo_bbox_for_iou_check, p_det_other['bbox'])
                                max_iou_with_other_person = max(max_iou_with_other_person, iou_val_with_other)
                                if iou_val_with_other > OCCLUSION_IOU_THRESHOLD:
                                    # High IoU detected while master is visible -> immediate occlusion
                                    current_state = "OCCLUSION_WAIT"
                                    OCCLUSION_START_TIME = time.time()
                                    announce(f"Occlusion detected! Master please stop for {OCCLUSION_STOP_SECONDS} seconds.")
                                    print(f"IoU-based occlusion detected (IoU: {max_iou_with_other_person:.2f})!")
                                    break # Exit inner loop, occlusion confirmed
                        
                        cv2.putText(image_bgr, f"Max IoU w/Others: {max_iou_with_other_person:.2f}", (10, 90),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                    else:
                        cv2.putText(image_bgr, "Master YOLO not confirmed", (10, 90),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 255), 1)

                    cv2.putText(image_bgr, f"Master Status: Tracking (ID: {MASTER_ID})", (10, 60),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                    if master_last_center_x is not None:
                        cv2.putText(image_bgr, f"Master X: {master_last_center_x:.0f}", (10, 110),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

                # --- Scenario B: Master skeleton is NOT currently tracked by MediaPipe ---
                else: # Master is lost by MediaPipe
                    # If master ID was never set or fully lost previously.
                    if MASTER_ID == -1 or MASTER_LAST_KNOWN_POSITION is None:
                        cv2.putText(image_bgr, "Master Status: Lost (Not Calibrated)", (10, 60),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                        master_disappeared_direction = None
                    else: # Master was previously tracked (MASTER_ID != -1 and MASTER_LAST_KNOWN_POSITION is not None)
                          # but is now lost by MediaPipe in this frame.

                        # Prioritize true disappearance if MASTER_OCCLUSION_FRAMES exceeds threshold.
                        # This implies a long-term loss, not just a brief occlusion.
                        if MASTER_OCCLUSION_FRAMES >= MAX_OCCLUSION_FRAMES:
                            if previous_master_was_tracked and master_disappeared_direction is None: # Only announce direction once after first real loss
                                direction = detect_master_disappearance_direction(image_width)
                                if direction:
                                    print(f"*** 主人骨架从视野{direction}边消失 (长时间丢失) ***")
                                    if direction == "LEFT":
                                        announce("Master disappeared from the left side")
                                    elif direction == "RIGHT":
                                        announce("Master disappeared from the right side")
                                    else:
                                        announce("Master disappeared from the center")
                                # Consider resetting MASTER_ID here if it's a permanent loss
                                # MASTER_ID = -1 # uncomment if a true permanent reset is desired
                            
                            cv2.putText(image_bgr, "Master Status: Lost (Disappeared)", (10, 60),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                            if master_disappeared_direction:
                                cv2.putText(image_bgr, f"Disappeared from: {master_disappeared_direction}", (10, 90),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 100, 100), 1)
                        else:
                            # Master is temporarily lost (short-term, within MAX_OCCLUSION_FRAMES).
                            # This is where re-identification attempt makes sense.
                            cv2.putText(image_bgr, "Master Status: Lost (Searching)", (10, 60),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 165, 0), 2)
                            if master_disappeared_direction: # If already set from a previous, quick disappearance.
                                cv2.putText(image_bgr, f"Last Direction: {master_disappeared_direction}", (10, 90),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 100, 100), 1)
                            
                            # Re-identification attempt: If master is not currently tracked by MediaPipe,
                            # and we are not in OCCLUSION_WAIT state (which is for IoU occlusions).
                            # The `is_master_mediapipe_lost_and_another_skeleton_nearby` (from detect_occlusion)
                            # is a softer "occlusion" signal, which should *not* prevent face re-id,
                            # as the face might be momentarily visible for re-id.
                            if current_state == "STEP3_TRACKING" and MASTER_FACE_ENCODINGS:
                                face_locations = face_recognition.face_locations(image_rgb)
                                face_encodings = face_recognition.face_encodings(image_rgb, face_locations)

                                if face_encodings:
                                    best_master_face_idx = -1
                                    best_master_distance = float('inf')
                                    
                                    for i, face_encoding in enumerate(face_encodings):
                                        is_master_face, distance = recognize_master_face([face_encoding])
                                        
                                        if is_master_face and distance < best_master_distance:
                                            best_master_distance = distance
                                            best_master_face_idx = i
                                    
                                    if best_master_face_idx != -1:
                                        top, right, bottom, left = face_locations[best_master_face_idx]
                                        face_bbox_reid = [left, top, right - left, bottom - top]
                                        face_center = [left + (right - left) // 2, top + (bottom - top) // 2]

                                        best_skeleton_id = -1
                                        best_skeleton_score = -1
                                        
                                        # Find the skeleton that best matches this re-identified master face
                                        for s_id, s_data in tracked_skeletons.items():
                                            # We need to make sure this is not the current MASTER_ID if MASTER_ID is just temporarily lost
                                            # and we are trying to re-associate. Or perhaps if is_master is false for some reason.
                                            # Let's consider all non-master skeletons for re-association.
                                            # Or specifically, if current MASTER_ID is lost (is_master_tracked_by_mediapipe_this_frame is false)
                                            # and this s_id is not already marked as master (s_data['is_master'] is False).
                                            if not s_data['is_master'] or (s_id == MASTER_ID and not master_is_tracked_by_mediapipe_this_frame):
                                                skeleton_bbox_reid = s_data['bbox']
                                                skeleton_center = [skeleton_bbox_reid[0] + skeleton_bbox_reid[2] // 2, 
                                                                    skeleton_bbox_reid[1] + skeleton_bbox_reid[3] // 2]
                                                
                                                iou = calculate_iou(face_bbox_reid, skeleton_bbox_reid)
                                                center_distance = np.sqrt((face_center[0] - skeleton_center[0])**2 + 
                                                                            (face_center[1] - skeleton_center[1])**2)
                                                relative_distance = center_distance / np.sqrt(image_width**2 + image_height**2)
                                                
                                                if iou > 0.05:
                                                    score = iou * 0.7 + (1.0 - relative_distance) * 0.3
                                                else:
                                                    score = (1.0 - relative_distance) * 0.5
                                                
                                                if score > best_skeleton_score:
                                                    best_skeleton_score = score
                                                    best_skeleton_id = s_id
                                        
                                        if best_skeleton_id != -1 and best_skeleton_score > 0.1:
                                            # Re-assign master if a good match is found
                                            MASTER_ID = best_skeleton_id
                                            tracked_skeletons[MASTER_ID]['is_master'] = True
                                            master_bbox = tracked_skeletons[MASTER_ID]['bbox']
                                            MASTER_LAST_KNOWN_POSITION = [master_bbox[0] + master_bbox[2] // 2, 
                                                                          master_bbox[1] + master_bbox[3] // 2]
                                            MASTER_OCCLUSION_FRAMES = 0 # Reset occlusion frames on re-id
                                            print(f"--- Master {MASTER_ID} re-identification successful! ---")
                                            print(f"Face match distance: {best_master_distance:.3f}")
                                            print(f"Skeleton binding score: {best_skeleton_score:.3f}")
                                            announce("Master re-identified.")
                                            master_disappeared_direction = None # Reset direction once re-identified


                # Draw MediaPipe skeletons
                for s_id, s_data in tracked_skeletons.items():
                    color = (0, 255, 0)
                    label = f"ID: {s_id}"
                    if s_data['is_master']:
                        color = (0, 0, 255)
                        label = f"Master ID: {s_id}"
                    
                    mp_drawing.draw_landmarks(image_bgr, s_data['last_pose'].pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                             mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2),
                                             mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2))
                    
                    x, y, w, h = s_data['bbox']
                    cv2.rectangle(image_bgr, (x, y), (x + w, y + h), color, 2)
                    cv2.putText(image_bgr, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

                # Draw YOLO person detections with color coding
                for p_det in person_detections:
                    x, y, w, h = p_det['bbox']
                    color = (0, 255, 0)  # Green for others
                    label = f"Person: {p_det['conf']:.2f}"
                    
                    # If master is currently tracked by MediaPipe, try to identify its YOLO bbox
                    if master_is_tracked_by_mediapipe_this_frame and MASTER_ID in tracked_skeletons:
                        master_skel_bbox = tracked_skeletons[MASTER_ID]['bbox']
                        if calculate_iou(p_det['bbox'], master_skel_bbox) > 0.3: # If YOLO bbox significantly overlaps with master's skeleton bbox
                            color = (0, 0, 255)  # Red for master
                            label = f"Master: {p_det['conf']:.2f}"
                    
                    cv2.rectangle(image_bgr, (x, y), (x + w, y + h), color, 2)
                    cv2.putText(image_bgr, label, (x, y + h + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

            elif current_state == "OCCLUSION_WAIT":
                # IoU-based occlusion wait period
                elapsed_time = time.time() - OCCLUSION_START_TIME
                remaining_time = max(0, OCCLUSION_STOP_SECONDS - elapsed_time)
                
                cv2.putText(image_bgr, f"IoU Occlusion Wait: {remaining_time:.1f}s", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2)
                
                if remaining_time <= 0:
                    current_state = "STEP3_TRACKING"
                    announce("Occlusion period ended. Resuming tracking.")
                    print("IoU-based occlusion period ended.")
                
                # Still draw skeletons/detections during wait
                # Get YOLO person detections for drawing
                person_detections = get_person_detections(yolo_model, image_bgr, YOLO_PERSON_CONFIDENCE_THRESHOLD)
                
                # Draw MediaPipe skeletons
                for s_id, s_data in tracked_skeletons.items():
                    color = (0, 255, 0)
                    label = f"ID: {s_id}"
                    if s_id == MASTER_ID: # Even in wait state, keep track of master if it's there
                        color = (0, 0, 255)
                        label = f"Master ID: {s_id} (WAIT)"
                    mp_drawing.draw_landmarks(image_bgr, s_data['last_pose'].pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                             mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2),
                                             mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2))
                    x, y, w, h = s_data['bbox']
                    cv2.rectangle(image_bgr, (x, y), (x + w, y + h), color, 2)
                    cv2.putText(image_bgr, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

                # Draw YOLO detections
                for p_det in person_detections:
                    x, y, w, h = p_det['bbox']
                    color = (0, 255, 0)
                    label = f"Person: {p_det['conf']:.2f}"
                    if MASTER_ID != -1 and MASTER_ID in tracked_skeletons: # Check if master skeleton is still technically tracked (even if occluded)
                        master_skel_bbox = tracked_skeletons[MASTER_ID]['bbox']
                        if calculate_iou(p_det['bbox'], master_skel_bbox) > 0.3:
                            color = (0, 0, 255)
                            label = f"Master (Occluded): {p_det['conf']:.2f}"
                    cv2.rectangle(image_bgr, (x, y), (x + w, y + h), color, 2)
                    cv2.putText(image_bgr, label, (x, y + h + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)



            # FPS monitoring for performance validation
            fps_frame_count += 1
            if time.time() - fps_start_time >= 1.0:
                current_fps = fps_frame_count / (time.time() - fps_start_time)
                fps_start_time = time.time()
                fps_frame_count = 0

            # Display FPS on all states
            cv2.putText(image_bgr, f"FPS: {current_fps:.1f}", (image_width - 120, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

            cv2.imshow('Human Following Algorithm', image_bgr)

            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('r'):
                MASTER_FACE_ENCODINGS = []
                MASTER_ID = -1
                MASTER_POSE_BUFFER = []
                MASTER_FACE_RECORDED = False
                MASTER_SKELETON_RECORDED = False
                MASTER_LAST_KNOWN_POSITION = None
                MASTER_OCCLUSION_FRAMES = 0
                tracked_skeletons = {}
                current_state = "STEP1_FACE_RECORDING"
                OCCLUSION_START_TIME = None
                WAIT_START_TIME = None
                occlusion_detected = False # Reset this too
                recovery_mode = ""
                step2_introduced = False
                yolo_model = None
                step1_last_face_detection_frame = 0
                step1_cached_face_data = {'locations': [], 'encodings': []}
                STEP1_LAST_CAPTURE_TIME = 0
                STEP1_CAPTURE_COUNT = 0
                master_last_center_x = None
                master_disappeared_direction = None
                print("--- All recorded information and master status has been reset ---")
                print("=== Back to Step 1: Face Recording ===")
                announce("System reset. Please record your face again.")
                announce("I will now capture your face. Please look to your left. Then straight. Then to your right. I will beep before each capture.")

            elif key == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()
    announce("System shutting down. Goodbye!")
    engine.stop()

if __name__ == '__main__':
    main()