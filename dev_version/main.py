import cv2
import mediapipe as mp
import numpy as np
from ultralytics import YOLO # For general object detection (people, faces)
import pyttsx3 # For text-to-speech
import face_recognition # For face recognition and re-identification
import time # For precise timing in occlusion

# --- MediaPipe Initialization ---
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# --- Text-to-Speech Initialization ---
engine = pyttsx3.init()
# Optional: Adjust speech rate and volume
engine.setProperty('rate', 150) # Speed of speech
engine.setProperty('volume', 0.9) # Volume (0.0 to 1.0)

def announce(text):
    """Speaks the given text."""
    print(f"ANNOUNCEMENT: {text}")
    engine.say(text)
    engine.runAndWait()

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
OCCLUSION_IOU_THRESHOLD = 0.25 # IoU threshold to consider an occlusion
OCCLUSION_STOP_SECONDS = 5 # Time in seconds to stop and wait after occlusion
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
FACE_RECOGNITION_TOLERANCE = 0.4 # Face recognition tolerance, smaller value = stricter
MIN_PERSON_AREA_THRESHOLD = 0.01 # Minimum relative area (bbox_area / image_area) for a person to be considered significant

# State variables
current_state = "STEP1_FACE_RECORDING"  # STEP1_FACE_RECORDING, STEP2_SKELETON_RECORDING, STEP3_TRACKING, OCCLUSION_WAIT, RECOVERY_WAIT
occlusion_detected = False
recovery_mode = ""  # "SINGLE_PERSON", "MULTIPLE_PEOPLE", "NO_PEOPLE"

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
    matched_current_indices = set()

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
            if best_match_id == MASTER_ID:
                if MASTER_LAST_KNOWN_POSITION is not None:
                    distance_to_last_known = np.linalg.norm(current_center - np.array(MASTER_LAST_KNOWN_POSITION))
                    if distance_to_last_known > OCCLUSION_DISTANCE_THRESHOLD:
                        print(f"Potential occluder detected! Distance from master's last position: {distance_to_last_known:.1f}px")
                        best_match_id = -1
                    else:
                        print(f"Master reappeared! Distance from last position: {distance_to_last_known:.1f}px")
            
            if best_match_id != -1:
                new_tracked_skeletons[best_match_id] = {
                    'last_pose': pose_results,
                    'bbox': current_bbox,
                    'last_seen_frame': frame_idx,
                    'is_master': tracked_skeletons[best_match_id]['is_master'],
                }
                matched_current_indices.add(current_idx)
            else:
                new_tracked_skeletons[next_id] = {
                    'last_pose': pose_results,
                    'bbox': current_bbox,
                    'last_seen_frame': frame_idx,
                    'is_master': False,
                }
                next_id += 1
                matched_current_indices.add(current_idx)
        else:
            new_tracked_skeletons[next_id] = {
                'last_pose': pose_results,
                'bbox': current_bbox,
                'last_seen_frame': frame_idx,
                'is_master': False,
            }
            next_id += 1
            matched_current_indices.add(current_idx)

    temp_tracked_skeletons = {}
    for s_id, s_data in new_tracked_skeletons.items():
        if frame_idx - s_data['last_seen_frame'] <= MAX_MISS_FRAMES:
            temp_tracked_skeletons[s_id] = s_data
        else:
            if s_id == MASTER_ID:
                MASTER_ID = -1 
                print(f"Master ID {s_id} skeleton not seen for a long time, resetting master status.")
    
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
    """Detect potential occlusion of master by other people."""
    global MASTER_LAST_KNOWN_POSITION, MASTER_OCCLUSION_FRAMES, MASTER_ID
    
    if MASTER_ID == -1 or MASTER_LAST_KNOWN_POSITION is None:
        return False
    
    master_currently_tracked = False
    master_current_position = None
    
    for s_id, s_data in tracked_skeletons.items():
        if s_id == MASTER_ID and s_data['is_master']:
            master_currently_tracked = True
            master_current_position = [s_data['bbox'][0] + s_data['bbox'][2] // 2, 
                                     s_data['bbox'][1] + s_data['bbox'][3] // 2]
            break
    
    if master_currently_tracked:
        MASTER_LAST_KNOWN_POSITION = master_current_position
        MASTER_OCCLUSION_FRAMES = 0
        return False
    else:
        MASTER_OCCLUSION_FRAMES += 1
        
        for landmarks, pose_results in current_skeletons_data:
            current_center = np.mean([[lm.x, lm.y] for lm in landmarks.landmark], axis=0) * np.array([image_width, image_height])
            distance_to_master = np.linalg.norm(current_center - np.array(MASTER_LAST_KNOWN_POSITION))
            
            if distance_to_master < OCCLUSION_DISTANCE_THRESHOLD:
                print(f"Potential occlusion detected! Distance to master: {distance_to_master:.1f}px")
                return True
        
        if MASTER_OCCLUSION_FRAMES > MAX_OCCLUSION_FRAMES:
            print(f"Master lost after {MASTER_OCCLUSION_FRAMES} frames of occlusion")
            MASTER_LAST_KNOWN_POSITION = None
            MASTER_OCCLUSION_FRAMES = 0
            return False
        
        return False

# --- Main Program Logic ---
def main():
    global MASTER_ID, MASTER_FACE_ENCODINGS, MASTER_POSE_BUFFER, next_id, tracked_skeletons
    global MASTER_FACE_RECORDED, MASTER_SKELETON_RECORDED, MASTER_LAST_KNOWN_POSITION, MASTER_OCCLUSION_FRAMES
    global current_state, OCCLUSION_START_TIME, WAIT_START_TIME, occlusion_detected, recovery_mode

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
    print("=== STEP 1: Record Master Face (Close to camera) ===")
    print("1. Move close to camera and press 's' multiple times to record your face from different angles.")
    print("2. Press 'r' to reset face records if needed.")
    print("3. Press 'n' to proceed to next step when face recording is complete.")

    # YOLO model will be loaded only when needed in Step 3
    yolo_model = None

    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        frame_count = 0
        step2_introduced = False
        
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

            # Update skeleton IDs
            update_skeleton_ids(current_skeletons_in_frame_data, frame_count, image_width, image_height)
            
            # Occlusion detection (only in normal tracking mode)
            occlusion_detected_original = False
            if MASTER_SKELETON_RECORDED and MASTER_ID != -1:
                occlusion_detected_original = detect_occlusion(current_skeletons_in_frame_data, frame_count, image_width, image_height)

            # Different logic based on current state
            if current_state == "STEP1_FACE_RECORDING":
                # STEP 1: Record master face (close to camera) - EXACT same as human_tracking.py
                face_locations = face_recognition.face_locations(image_rgb)
                face_encodings = face_recognition.face_encodings(image_rgb, face_locations)
                
                cv2.putText(image_bgr, "STEP 1: Record Master Face (Close to camera)", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)
                cv2.putText(image_bgr, f"Recorded faces: {len(MASTER_FACE_ENCODINGS)}", (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 1, cv2.LINE_AA)
                cv2.putText(image_bgr, f"Face Tolerance: {FACE_RECOGNITION_TOLERANCE}", (10, 90),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 1, cv2.LINE_AA)
                cv2.putText(image_bgr, "Press 's' to record face, 'n' to proceed", (10, 120),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1, cv2.LINE_AA)
                
                for (top, right, bottom, left) in face_locations:
                    cv2.rectangle(image_bgr, (left, top), (right, bottom), (0, 255, 0), 2)
                    cv2.putText(image_bgr, "Found Face (Press 's')", (left, top - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)
                
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
                # STEP 2: Record master skeleton - EXACT same as human_tracking.py
                if not step2_introduced:
                    print("\n=== STEP 2: Record Master Skeleton (Move away from camera) ===")
                    print("1. Move away from camera so your full body is visible.")
                    print("2. Make 'hands on hips' gesture to calibrate as master.")
                    step2_introduced = True
                
                cv2.putText(image_bgr, "STEP 2: Record Master Skeleton (Full body visible)", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2, cv2.LINE_AA)
                cv2.putText(image_bgr, f"Recorded faces: {len(MASTER_FACE_ENCODINGS)}", (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 1, cv2.LINE_AA)
                cv2.putText(image_bgr, "Make 'hands on hips' gesture", (10, 90),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 1, cv2.LINE_AA)
                
                for s_id, s_data in tracked_skeletons.items():
                    if s_data['last_seen_frame'] == frame_count:
                        landmarks_for_action = s_data['last_pose'].pose_landmarks
                        if is_hands_on_hips(landmarks_for_action):
                            MASTER_POSE_BUFFER.append(True)
                        else:
                            MASTER_POSE_BUFFER.append(False)
                        
                        if len(MASTER_POSE_BUFFER) > MASTER_HISTORY_THRESHOLD:
                            MASTER_POSE_BUFFER.pop(0)
                        
                        if sum(MASTER_POSE_BUFFER) == MASTER_HISTORY_THRESHOLD:
                            face_locations = face_recognition.face_locations(image_rgb)
                            face_encodings = face_recognition.face_encodings(image_rgb, face_locations)
                            
                            if face_encodings:
                                is_master_face, best_distance = recognize_master_face(face_encodings)
                                
                                if is_master_face:
                                    MASTER_ID = s_id
                                    tracked_skeletons[MASTER_ID]['is_master'] = True
                                    MASTER_SKELETON_RECORDED = True
                                    master_bbox = tracked_skeletons[MASTER_ID]['bbox']
                                    MASTER_LAST_KNOWN_POSITION = [master_bbox[0] + master_bbox[2] // 2, 
                                                                master_bbox[1] + master_bbox[3] // 2]
                                    current_state = "STEP3_TRACKING"
                                    announce("Master skeleton calibrated! Starting enhanced tracking mode.")
                                    print(f"--- Master skeleton {MASTER_ID} calibrated through hands-on-hips action and face recognition! ---")
                                    print(f"Face match distance: {best_distance:.3f}")
                                    MASTER_POSE_BUFFER = []
                                    break
                                else:
                                    print("Warning: Hands-on-hips action detected but face doesn't match recorded master faces.")
                                    print(f"Best face distance: {best_distance:.3f} (threshold: {FACE_RECOGNITION_TOLERANCE})")
                                    MASTER_POSE_BUFFER = []
                            else:
                                print("Warning: Hands-on-hips action detected but no face found in the image.")
                                MASTER_POSE_BUFFER = []
                
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
                    cv2.putText(image_bgr, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

            elif current_state == "STEP3_TRACKING":
                # STEP 3: Enhanced tracking with MediaPipe + YOLO parallel processing
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

                # Parallel processing: MediaPipe + YOLO
                person_detections = get_person_detections(yolo_model, image_bgr, YOLO_PERSON_CONFIDENCE_THRESHOLD)
                significant_persons = filter_significant_persons(person_detections, image_width, image_height, MIN_PERSON_AREA_THRESHOLD)

                # Check if master is still being tracked
                master_in_view = (MASTER_ID != -1 and MASTER_ID in tracked_skeletons and tracked_skeletons[MASTER_ID]['is_master'])
                
                if master_in_view:
                    # Find master's YOLO bounding box
                    master_yolo_bbox = None
                    master_skel_bbox = tracked_skeletons[MASTER_ID]['bbox']
                    
                    best_iou = 0
                    for p_det in significant_persons:
                        iou = calculate_iou(p_det['bbox'], master_skel_bbox)
                        if iou > best_iou and iou > 0.3:
                            best_iou = iou
                            master_yolo_bbox = p_det['bbox']
                    
                    if master_yolo_bbox:
                        # Check for IoU-based occlusion with other people
                        max_iou = 0
                        occlusion_this_frame = False
                        
                        for p_det in significant_persons:
                            if p_det['bbox'] != master_yolo_bbox:
                                iou = calculate_iou(master_yolo_bbox, p_det['bbox'])
                                max_iou = max(max_iou, iou)
                                if iou > OCCLUSION_IOU_THRESHOLD:
                                    occlusion_this_frame = True
                                    break
                        
                        if occlusion_this_frame:
                            current_state = "OCCLUSION_WAIT"
                            OCCLUSION_START_TIME = time.time()
                            announce(f"Occlusion detected! Master please stop for {OCCLUSION_STOP_SECONDS} seconds.")
                            print(f"IoU-based occlusion detected (IoU: {max_iou:.2f})!")
                        
                        cv2.putText(image_bgr, f"Master Status: Tracking (ID: {MASTER_ID})", (10, 60),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                        cv2.putText(image_bgr, f"Max IoU: {max_iou:.2f}", (10, 90),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                    else:
                        cv2.putText(image_bgr, f"Master Status: Skeleton tracked, YOLO lost", (10, 60),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 165, 0), 2)
                else:
                    cv2.putText(image_bgr, "Master Status: Lost", (10, 60),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

                # Master re-identification (from human_tracking.py logic)
                is_master_present_this_frame = False
                if MASTER_ID != -1 and MASTER_ID in tracked_skeletons:
                    is_master_present_this_frame = tracked_skeletons[MASTER_ID]['is_master']

                if MASTER_ID != -1 and not is_master_present_this_frame and MASTER_FACE_ENCODINGS and not occlusion_detected_original:
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
                            
                            for s_id, s_data in tracked_skeletons.items():
                                if not s_data['is_master']:
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
                                MASTER_ID = best_skeleton_id
                                tracked_skeletons[MASTER_ID]['is_master'] = True
                                master_bbox = tracked_skeletons[MASTER_ID]['bbox']
                                MASTER_LAST_KNOWN_POSITION = [master_bbox[0] + master_bbox[2] // 2, 
                                                            master_bbox[1] + master_bbox[3] // 2]
                                print(f"--- Master {MASTER_ID} re-identification successful! ---")
                                print(f"Face match distance: {best_master_distance:.3f}")
                                print(f"Skeleton binding score: {best_skeleton_score:.3f}")

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
                    
                    if master_in_view:
                        master_skel_bbox = tracked_skeletons[MASTER_ID]['bbox']
                        if calculate_iou(p_det['bbox'], master_skel_bbox) > 0.3:
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

            cv2.imshow('Human Following Algorithm', image_bgr)

            # Handle key presses - EXACT same as human_tracking.py
            key = cv2.waitKey(1) & 0xFF
            if key == ord('s') and current_state == "STEP1_FACE_RECORDING":
                face_locations = face_recognition.face_locations(image_rgb)
                face_encodings = face_recognition.face_encodings(image_rgb, face_locations)
                
                total_recorded = 0
                for face_encoding in face_encodings:
                    MASTER_FACE_ENCODINGS.append(face_encoding)
                    total_recorded += 1
                    print(f"--- Recorded {len(MASTER_FACE_ENCODINGS)}th master face information ---")
                
                if total_recorded > 0:
                    print(f"Currently recorded {len(MASTER_FACE_ENCODINGS)} face information")
                    announce(f"Recorded {total_recorded} face samples.")
                else:
                    print("No face detected, cannot record face information. Please ensure you are facing the camera.")
                    announce("No face detected. Please face the camera.")

            elif key == ord('n') and current_state == "STEP1_FACE_RECORDING":
                if len(MASTER_FACE_ENCODINGS) > 0:
                    current_state = "STEP2_SKELETON_RECORDING"
                    MASTER_FACE_RECORDED = True
                    print(f"\n=== Proceeding to Step 2: Skeleton Recording ===")
                    print(f"Face recording complete! {len(MASTER_FACE_ENCODINGS)} faces recorded.")
                    print("Now move away from camera and make 'hands on hips' gesture.")
                    announce("Face recording complete. Now move away and make 'hands on hips' gesture.")
                else:
                    print("Please record at least one face first.")
                    announce("Please record your face first.")

            elif key == ord('r'):
                # Reset everything
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
                occlusion_detected = False
                recovery_mode = ""
                step2_introduced = False
                yolo_model = None
                print("--- All recorded information and master status has been reset ---")
                print("=== Back to Step 1: Face Recording ===")
                announce("System reset. Please record your face again.")

            elif key == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()
    announce("System shutting down. Goodbye!")
    engine.stop()

if __name__ == '__main__':
    main()