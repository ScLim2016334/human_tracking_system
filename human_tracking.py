import cv2
import mediapipe as mp
import numpy as np
import face_recognition # For face recognition and re-identification
import pyttsx3 # For text-to-speech
import time
import threading

# --- MediaPipe Initialization ---
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# --- Global Variables and Configuration Parameters ---
# Skeleton tracker related
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

# New occlusion handling related (IoU-based)
OCCLUSION_IOU_THRESHOLD = 0.3 # IoU threshold to detect occlusion
OCCLUSION_STOP_TIME = 3 # Time in seconds to ask master to stop
OCCLUSION_DETECTION_FRAMES = 5 # Number of consecutive frames to confirm occlusion
occlusion_detection_buffer = [] # Buffer to store recent occlusion detection results
is_in_occlusion_mode = False # Whether currently in occlusion handling mode
occlusion_start_time = None # When occlusion mode started
last_announcement_time = 0 # To prevent too frequent announcements
ANNOUNCEMENT_COOLDOWN = 2 # Minimum seconds between announcements

# Face recognition related
FACE_RECOGNITION_TOLERANCE = 0.4 # Face recognition tolerance, smaller value = stricter (0.6 is common default)

# Text-to-speech engine
tts_engine = None
def init_tts():
    """Initialize text-to-speech engine"""
    global tts_engine
    try:
        tts_engine = pyttsx3.init()
        tts_engine.setProperty('rate', 150)  # Speed of speech
        tts_engine.setProperty('volume', 0.9)  # Volume level
        print("Text-to-speech engine initialized successfully")
    except Exception as e:
        print(f"Failed to initialize text-to-speech engine: {e}")
        tts_engine = None

def announce_message(message):
    """Announce message using text-to-speech"""
    global last_announcement_time
    current_time = time.time()
    
    # Prevent too frequent announcements
    if current_time - last_announcement_time < ANNOUNCEMENT_COOLDOWN:
        return
    
    last_announcement_time = current_time
    print(f"ANNOUNCEMENT: {message}")
    
    if tts_engine:
        try:
            # Run TTS in a separate thread to avoid blocking
            def speak():
                tts_engine.say(message)
                tts_engine.runAndWait()
            
            thread = threading.Thread(target=speak)
            thread.daemon = True
            thread.start()
        except Exception as e:
            print(f"Failed to announce message: {e}")

# --- Helper Functions ---

def get_bbox_from_landmarks(landmarks, image_width, image_height):
    """Calculate skeleton bounding box from MediaPipe keypoints."""
    if not landmarks:
        return [0, 0, 0, 0] # Return an invalid bbox
    x_coords = [lm.x * image_width for lm in landmarks.landmark if lm.visibility > 0.5] # Only consider keypoints with high visibility
    y_coords = [lm.y * image_height for lm in landmarks.landmark if lm.visibility > 0.5]
    
    if not x_coords or not y_coords: # If not enough visible keypoints
        return [0, 0, 0, 0]

    min_x, max_x = int(min(x_coords)), int(max(x_coords))
    min_y, max_y = int(min(y_coords)), int(max(y_coords))
    
    # Add some padding to ensure the entire body is included
    padding = 20 
    min_x = max(0, min_x - padding)
    min_y = max(0, min_y - padding)
    max_x = min(image_width - 1, max_x + padding)
    max_y = min(image_height - 1, max_y + padding)

    return [min_x, min_y, max_x - min_x, max_y - min_y]

def update_skeleton_ids(current_skeletons_data, frame_idx, image_width, image_height):
    """
    Update and maintain skeleton IDs.
    Try to match IDs by comparing current frame skeletons with previous frame skeletons' positions, and remove skeletons not seen for a long time.
    """
    global next_id, tracked_skeletons, MASTER_ID

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
            next_id += 1
            matched_current_indices.add(current_idx)

    # Remove skeletons not seen for a long time
    temp_tracked_skeletons = {}
    for s_id, s_data in new_tracked_skeletons.items():
        if frame_idx - s_data['last_seen_frame'] <= MAX_MISS_FRAMES:
            temp_tracked_skeletons[s_id] = s_data
        else:
            # If master disappears, reset MASTER_ID
            if s_id == MASTER_ID:
                MASTER_ID = -1 
                print(f"Master ID {s_id} skeleton not seen for a long time, resetting master status.")
    
    tracked_skeletons = temp_tracked_skeletons

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
    if not (left_wrist.visibility > 0.7 and right_wrist.visibility > 0.7 and
            left_hip.visibility > 0.7 and right_hip.visibility > 0.7 and
            left_elbow.visibility > 0.7 and right_elbow.visibility > 0.7):
        return False

    # Check if wrists are near hips (using relative coordinates)
    # Assume wrists are within 0.1 range of hip Y coordinate, and X coordinate within 0.05 range of hip X coordinate (considering lateral offset)
    wrist_hip_y_threshold_rel = 0.08 
    wrist_hip_x_threshold_rel = 0.05 

    left_hand_on_hip = abs(left_wrist.y - left_hip.y) < wrist_hip_y_threshold_rel and \
                       abs(left_wrist.x - left_hip.x) < wrist_hip_x_threshold_rel

    right_hand_on_hip = abs(right_wrist.y - right_hip.y) < wrist_hip_y_threshold_rel and \
                        abs(right_wrist.x - right_hip.x) < wrist_hip_x_threshold_rel

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

    elbow_angle_threshold_min = 60 # Minimum elbow opening angle (avoid completely straight arms)
    elbow_angle_threshold_max = 120 # Maximum elbow opening angle (avoid over-bent arms)

    left_elbow_ok = elbow_angle_threshold_min < left_elbow_angle < elbow_angle_threshold_max
    right_elbow_ok = elbow_angle_threshold_min < right_elbow_angle < elbow_angle_threshold_max
    
    # At least one hand satisfies hands-on-hips condition
    return (left_hand_on_hip and left_elbow_ok) or (right_hand_on_hip and right_elbow_ok)

def get_face_encodings_from_bbox(image, bbox):
    """
    Extract face encodings from image and bounding box.
    bbox format: [x, y, w, h]
    Returns list of face encodings found in the region
    """
    x, y, w, h = bbox
    # Ensure crop region is valid
    if w <= 0 or h <= 0 or x < 0 or y < 0 or x + w > image.shape[1] or y + h > image.shape[0]:
        return []

    # face_recognition library expects (top, right, bottom, left) format
    face_location = (y, x + w, y + h, x)
    face_encodings = face_recognition.face_encodings(image, [face_location])
    return face_encodings

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

def recognize_master_face(face_encodings):
    """
    Recognize if any of the given face encodings match the master.
    Returns (is_master, best_distance) tuple.
    """
    if not MASTER_FACE_ENCODINGS:
        return False, float('inf')
    
    best_distance = float('inf')
    is_master = False
    
    for face_encoding in face_encodings:
        # Compare with all recorded master face encodings
        distances = []
        matches = []
        
        for master_encoding in MASTER_FACE_ENCODINGS:
            # Calculate face distance
            distance = face_recognition.face_distance([master_encoding], face_encoding)[0]
            distances.append(distance)
            matches.append(distance <= FACE_RECOGNITION_TOLERANCE)
        
        # Find best match for this face
        min_distance = min(distances)
        if any(matches) and min_distance < best_distance:
            best_distance = min_distance
            is_master = True
    
    return is_master, best_distance

def detect_occlusion_by_iou(current_skeletons_data, frame_idx, image_width, image_height):
    """
    Detect occlusion using IoU between master and other skeletons.
    Returns True if occlusion is detected, False otherwise.
    """
    global occlusion_detection_buffer, is_in_occlusion_mode, occlusion_start_time, MASTER_ID
    
    if MASTER_ID == -1 or MASTER_ID not in tracked_skeletons:
        return False
    
    # Get master skeleton bbox
    master_bbox = tracked_skeletons[MASTER_ID]['bbox']
    
    # Check IoU with all other skeletons in current frame
    max_iou = 0
    for landmarks, pose_results in current_skeletons_data:
        current_bbox = get_bbox_from_landmarks(landmarks, image_width, image_height)
        iou = calculate_iou(master_bbox, current_bbox)
        max_iou = max(max_iou, iou)
    
    # Update occlusion detection buffer
    occlusion_detected = max_iou > OCCLUSION_IOU_THRESHOLD
    occlusion_detection_buffer.append(occlusion_detected)
    
    # Keep only recent frames
    if len(occlusion_detection_buffer) > OCCLUSION_DETECTION_FRAMES:
        occlusion_detection_buffer.pop(0)
    
    # Confirm occlusion if detected in consecutive frames
    if len(occlusion_detection_buffer) == OCCLUSION_DETECTION_FRAMES and all(occlusion_detection_buffer):
        if not is_in_occlusion_mode:
            is_in_occlusion_mode = True
            occlusion_start_time = time.time()
            announce_message(f"Master please stop for {OCCLUSION_STOP_TIME} seconds")
            print(f"Occlusion detected! IoU: {max_iou:.3f}, asking master to stop for {OCCLUSION_STOP_TIME} seconds")
        return True
    
    return False

# --- Main Program Logic ---
def main():
    global MASTER_ID, MASTER_FACE_ENCODINGS, MASTER_POSE_BUFFER, next_id, tracked_skeletons, FACE_RECOGNITION_TOLERANCE
    global MASTER_FACE_RECORDED, MASTER_SKELETON_RECORDED, is_in_occlusion_mode, occlusion_start_time

    # Initialize text-to-speech engine
    init_tts()

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
    # static_image_mode=False means video stream mode, enable_segmentation=False reduces computation
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
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
            
            # --- 2. Occlusion detection and recovery (only in normal tracking mode) ---
            occlusion_detected = False
            if MASTER_SKELETON_RECORDED and MASTER_ID != -1:
                occlusion_detected = detect_occlusion_by_iou(current_skeletons_in_frame_data, frame_count, image_width, image_height)
                
                # Handle occlusion recovery
                if is_in_occlusion_mode:
                    # Check if occlusion period is over
                    current_time = time.time()
                    if current_time - occlusion_start_time >= OCCLUSION_STOP_TIME:
                        # Try to recover master
                        skeleton_count = len(current_skeletons_in_frame_data)
                        
                        if skeleton_count == 0:
                            # No skeletons detected
                            announce_message("Master I can't see you, please come in front of me")
                            print("No skeletons detected after occlusion, asking master to come in front")
                            is_in_occlusion_mode = False
                            occlusion_detection_buffer.clear()
                            
                        elif skeleton_count == 1:
                            # Only one skeleton, assume it's master
                            landmarks, pose_results = current_skeletons_in_frame_data[0]
                            current_bbox = get_bbox_from_landmarks(landmarks, image_width, image_height)
                            
                            # Assign this skeleton as master
                            new_master_id = next_id
                            tracked_skeletons[new_master_id] = {
                                'last_pose': pose_results,
                                'bbox': current_bbox,
                                'last_seen_frame': frame_count,
                                'is_master': True,
                            }
                            MASTER_ID = new_master_id
                            next_id += 1
                            is_in_occlusion_mode = False
                            occlusion_detection_buffer.clear()
                            print(f"Single skeleton detected after occlusion, assigned as master (ID: {MASTER_ID})")
                            
                        else:
                            # Multiple skeletons, need face recognition
                            announce_message("Master please look at me")
                            print(f"Multiple skeletons detected after occlusion ({skeleton_count}), asking master to look at camera")
                            
                            # Detect faces in the image
                            face_locations = face_recognition.face_locations(image_rgb)
                            face_encodings = face_recognition.face_encodings(image_rgb, face_locations)
                            
                            if face_encodings:
                                # Find master face
                                best_master_face_idx = -1
                                best_master_distance = float('inf')
                                
                                for i, face_encoding in enumerate(face_encodings):
                                    is_master_face, distance = recognize_master_face([face_encoding])
                                    if is_master_face and distance < best_master_distance:
                                        best_master_distance = distance
                                        best_master_face_idx = i
                                
                                if best_master_face_idx != -1:
                                    # Found master face, find closest skeleton
                                    top, right, bottom, left = face_locations[best_master_face_idx]
                                    face_center = [left + (right - left) // 2, top + (bottom - top) // 2]
                                    
                                    best_skeleton_idx = -1
                                    min_distance = float('inf')
                                    
                                    for i, (landmarks, pose_results) in enumerate(current_skeletons_in_frame_data):
                                        current_bbox = get_bbox_from_landmarks(landmarks, image_width, image_height)
                                        skeleton_center = [current_bbox[0] + current_bbox[2] // 2, current_bbox[1] + current_bbox[3] // 2]
                                        distance = np.sqrt((face_center[0] - skeleton_center[0])**2 + (face_center[1] - skeleton_center[1])**2)
                                        
                                        if distance < min_distance:
                                            min_distance = distance
                                            best_skeleton_idx = i
                                    
                                    if best_skeleton_idx != -1:
                                        # Assign closest skeleton to master face as master
                                        landmarks, pose_results = current_skeletons_in_frame_data[best_skeleton_idx]
                                        current_bbox = get_bbox_from_landmarks(landmarks, image_width, image_height)
                                        
                                        new_master_id = next_id
                                        tracked_skeletons[new_master_id] = {
                                            'last_pose': pose_results,
                                            'bbox': current_bbox,
                                            'last_seen_frame': frame_count,
                                            'is_master': True,
                                        }
                                        MASTER_ID = new_master_id
                                        next_id += 1
                                        is_in_occlusion_mode = False
                                        occlusion_detection_buffer.clear()
                                        print(f"Master face recognized, assigned closest skeleton as master (ID: {MASTER_ID})")
                                    else:
                                        announce_message("Master I can't see you, please come in front of me")
                                        print("No suitable skeleton found for master face")
                                        is_in_occlusion_mode = False
                                        occlusion_detection_buffer.clear()
                                else:
                                    announce_message("Master I can't see you, please come in front of me")
                                    print("No master face recognized after occlusion")
                                    is_in_occlusion_mode = False
                                    occlusion_detection_buffer.clear()
                            else:
                                announce_message("Master I can't see you, please come in front of me")
                                print("No faces detected after occlusion")
                                is_in_occlusion_mode = False
                                occlusion_detection_buffer.clear()
            
            # --- 3. Different logic based on recording stage ---
            if not MASTER_FACE_RECORDED:
                # STEP 1: Record master face (close to camera)
                # Detect faces in entire image for face recording
                face_locations = face_recognition.face_locations(image_rgb)
                face_encodings = face_recognition.face_encodings(image_rgb, face_locations)
                
                # Display face recording status
                cv2.putText(image_bgr, "STEP 1: Record Master Face (Close to camera)", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)
                cv2.putText(image_bgr, f"Recorded faces: {len(MASTER_FACE_ENCODINGS)}", (10, 60),
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
                
                # Draw skeletons (if any) but don't process them yet
                for s_id, s_data in tracked_skeletons.items():
                    color = (128, 128, 128) # Gray for unprocessed skeletons
                    label = f"ID: {s_id} (Not Master Yet)"
                    
                    # Draw skeleton
                    mp_drawing.draw_landmarks(image_bgr, s_data['last_pose'].pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                             mp_drawing.DrawingSpec(color=(128,128,128), thickness=2, circle_radius=2),
                                             mp_drawing.DrawingSpec(color=(128,128,128), thickness=2, circle_radius=2))

                    # Draw bounding box and ID label
                    x, y, w, h = s_data['bbox']
                    cv2.rectangle(image_bgr, (x, y), (x + w, y + h), color, 2)
                    cv2.putText(image_bgr, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                
            elif not MASTER_SKELETON_RECORDED:
                # STEP 2: Record master skeleton (move away from camera)
                if not step2_introduced:
                    print("\n=== STEP 2: Record Master Skeleton (Move away from camera) ===")
                    print("1. Move away from camera so your full body is visible.")
                    print("2. Make 'hands on hips' gesture to calibrate as master.")
                    step2_introduced = True
                
                # Display skeleton recording status
                cv2.putText(image_bgr, "STEP 2: Record Master Skeleton (Full body visible)", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2, cv2.LINE_AA)
                cv2.putText(image_bgr, f"Recorded faces: {len(MASTER_FACE_ENCODINGS)}", (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 1, cv2.LINE_AA)
                cv2.putText(image_bgr, "Make 'hands on hips' gesture", (10, 90),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 1, cv2.LINE_AA)
                
                # Try to calibrate master skeleton
                for s_id, s_data in tracked_skeletons.items():
                    # Only consider skeletons just detected in current frame
                    if s_data['last_seen_frame'] == frame_count:
                        landmarks_for_action = s_data['last_pose'].pose_landmarks
                        if is_hands_on_hips(landmarks_for_action):
                            MASTER_POSE_BUFFER.append(True)
                        else:
                            MASTER_POSE_BUFFER.append(False)
                        
                        # Maintain buffer size
                        if len(MASTER_POSE_BUFFER) > MASTER_HISTORY_THRESHOLD:
                            MASTER_POSE_BUFFER.pop(0)
                        
                        # If hands-on-hips action for consecutive frames
                        if sum(MASTER_POSE_BUFFER) == MASTER_HISTORY_THRESHOLD:
                            # Check if this person's face matches any recorded master faces
                            # Detect faces in entire image (not just skeleton bbox)
                            face_locations = face_recognition.face_locations(image_rgb)
                            face_encodings = face_recognition.face_encodings(image_rgb, face_locations)
                            
                            if face_encodings:
                                # Check if any face in the image matches master
                                is_master_face, best_distance = recognize_master_face(face_encodings)
                                
                                if is_master_face:
                                    # Set this skeleton as master
                                    MASTER_ID = s_id
                                    tracked_skeletons[MASTER_ID]['is_master'] = True
                                    MASTER_SKELETON_RECORDED = True
                                    # Initialize last known position
                                    master_bbox = tracked_skeletons[MASTER_ID]['bbox']
                                    MASTER_LAST_KNOWN_POSITION = [master_bbox[0] + master_bbox[2] // 2, 
                                                                master_bbox[1] + master_bbox[3] // 2]
                                    print(f"--- Master skeleton {MASTER_ID} calibrated through hands-on-hips action and face recognition! ---")
                                    print(f"Face match distance: {best_distance:.3f}")
                                    MASTER_POSE_BUFFER = [] # Clear cache to avoid repeated triggering
                                    break # Exit after finding master to avoid trying to calibrate others
                                else:
                                    print("Warning: Hands-on-hips action detected but face doesn't match recorded master faces.")
                                    print(f"Best face distance: {best_distance:.3f} (threshold: {FACE_RECOGNITION_TOLERANCE})")
                                    MASTER_POSE_BUFFER = [] # Reset cache if face doesn't match
                            else:
                                print("Warning: Hands-on-hips action detected but no face found in the image.")
                                MASTER_POSE_BUFFER = [] # Reset cache if no face found
                
                # Draw all skeletons
                for s_id, s_data in tracked_skeletons.items():
                    color = (0, 255, 0) # Green for normal skeletons
                    label = f"ID: {s_id}"
                    
                    if s_data['is_master']:
                        color = (0, 0, 255) # Red for master
                        label = f"Master ID: {s_id}"
                    
                    # Draw skeleton
                    mp_drawing.draw_landmarks(image_bgr, s_data['last_pose'].pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                             mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2),
                                             mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2))

                    # Draw bounding box and ID label
                    x, y, w, h = s_data['bbox']
                    cv2.rectangle(image_bgr, (x, y), (x + w, y + h), color, 2)
                    cv2.putText(image_bgr, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                
            else:
                # STEP 3: Normal tracking with both face and skeleton recognition
                # --- Master re-identification (only when master not currently in view and not occluded) ---
                is_master_present_this_frame = False
                if MASTER_ID != -1 and MASTER_ID in tracked_skeletons:
                    is_master_present_this_frame = tracked_skeletons[MASTER_ID]['is_master']

                # Only perform face re-identification when master skeleton is not tracked and not occluded
                if MASTER_ID != -1 and not is_master_present_this_frame and MASTER_FACE_ENCODINGS and not occlusion_detected:
                    # Detect faces in entire image for re-identification
                    face_locations = face_recognition.face_locations(image_rgb)
                    face_encodings = face_recognition.face_encodings(image_rgb, face_locations)

                    if face_encodings:
                        reidentified_master_found = False
                        best_master_face_idx = -1
                        best_master_distance = float('inf')
                        
                        # First, find the best matching master face in the image
                        for i, face_encoding in enumerate(face_encodings):
                            # Check if this face matches master
                            is_master_face, distance = recognize_master_face([face_encoding])
                            
                            if is_master_face and distance < best_master_distance:
                                best_master_distance = distance
                                best_master_face_idx = i
                        
                        # If we found a master face, try to bind it to a skeleton
                        if best_master_face_idx != -1:
                            # Found matching face
                            top, right, bottom, left = face_locations[best_master_face_idx]
                            face_bbox_reid = [left, top, right - left, bottom - top]
                            face_center = [left + (right - left) // 2, top + (bottom - top) // 2]

                            # Try to bind face with some skeleton in current frame that is **not marked as master**
                            best_skeleton_id = -1
                            best_skeleton_score = -1
                            
                            for s_id, s_data in tracked_skeletons.items():
                                if not s_data['is_master']: # Ensure not already marked master
                                    skeleton_bbox_reid = s_data['bbox']
                                    skeleton_center = [skeleton_bbox_reid[0] + skeleton_bbox_reid[2] // 2, 
                                                      skeleton_bbox_reid[1] + skeleton_bbox_reid[3] // 2]
                                    
                                    # Calculate multiple matching criteria
                                    iou = calculate_iou(face_bbox_reid, skeleton_bbox_reid)
                                    
                                    # Calculate distance between face center and skeleton center
                                    center_distance = np.sqrt((face_center[0] - skeleton_center[0])**2 + 
                                                            (face_center[1] - skeleton_center[1])**2)
                                    
                                    # Calculate relative distance (normalized by image size)
                                    relative_distance = center_distance / np.sqrt(image_width**2 + image_height**2)
                                    
                                    # Combined score: prioritize IoU but also consider center distance
                                    # For distant faces, IoU might be low but center distance should be reasonable
                                    if iou > 0.05:  # Very low IoU threshold for distant faces
                                        score = iou * 0.7 + (1.0 - relative_distance) * 0.3
                                    else:
                                        # If IoU is too low, only consider center distance
                                        score = (1.0 - relative_distance) * 0.5
                                    
                                    if score > best_skeleton_score:
                                        best_skeleton_score = score
                                        best_skeleton_id = s_id
                            
                            # If we found a suitable skeleton, bind it
                            if best_skeleton_id != -1 and best_skeleton_score > 0.1:
                                MASTER_ID = best_skeleton_id # Re-set master ID
                                tracked_skeletons[MASTER_ID]['is_master'] = True
                                # Update last known position
                                master_bbox = tracked_skeletons[MASTER_ID]['bbox']
                                MASTER_LAST_KNOWN_POSITION = [master_bbox[0] + master_bbox[2] // 2, 
                                                            master_bbox[1] + master_bbox[3] // 2]
                                print(f"--- Master {MASTER_ID} re-identification successful! ---")
                                print(f"Face match distance: {best_master_distance:.3f}")
                                print(f"Skeleton binding score: {best_skeleton_score:.3f}")
                                reidentified_master_found = True
                            else:
                                print(f"Master face detected (distance: {best_master_distance:.3f}) but no suitable skeleton found.")
                                print(f"Best skeleton score: {best_skeleton_score:.3f}")

                # --- Draw and display results ---
                for s_id, s_data in tracked_skeletons.items():
                    color = (0, 255, 0) # Default: green
                    label = f"ID: {s_id}"
                    
                    if s_data['is_master']:
                        color = (0, 0, 255) # Master: red
                        label = f"Master ID: {s_id}"
                    
                    # Draw skeleton
                    mp_drawing.draw_landmarks(image_bgr, s_data['last_pose'].pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                             mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2),
                                             mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2))

                    # Draw bounding box and ID label
                    x, y, w, h = s_data['bbox']
                    cv2.rectangle(image_bgr, (x, y), (x + w, y + h), color, 2)
                    cv2.putText(image_bgr, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

                # Display current master status and face recognition info
                master_status_text = "Master Status: Not Set"
                if MASTER_ID != -1:
                    if MASTER_ID in tracked_skeletons and tracked_skeletons[MASTER_ID]['is_master']:
                        master_status_text = f"Master Status: Tracking (ID: {MASTER_ID})"
                    else:
                        if occlusion_detected:
                            master_status_text = f"Master Status: Occluded ({occlusion_detection_buffer.count(True)} frames)"
                        else:
                            master_status_text = "Master Status: Attempting Re-identification..."
                
                cv2.putText(image_bgr, "STEP 3: Normal Tracking Mode", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
                cv2.putText(image_bgr, master_status_text, (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
                cv2.putText(image_bgr, f"Face Tolerance: {FACE_RECOGNITION_TOLERANCE}", (10, 90),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1, cv2.LINE_AA)
                cv2.putText(image_bgr, f"Recorded faces: {len(MASTER_FACE_ENCODINGS)}", (10, 120),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1, cv2.LINE_AA)

            cv2.imshow('Human Following Algorithm Demo', image_bgr)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('s'): # Press 's' to record master face information
                if not MASTER_FACE_RECORDED:
                    # Record faces from entire image (close to camera)
                    face_locations = face_recognition.face_locations(image_rgb)
                    face_encodings = face_recognition.face_encodings(image_rgb, face_locations)
                    
                    total_recorded = 0
                    for face_encoding in face_encodings:
                        MASTER_FACE_ENCODINGS.append(face_encoding)
                        total_recorded += 1
                        print(f"--- Recorded {len(MASTER_FACE_ENCODINGS)}th master face information ---")
                        print(f"Face encoding: {face_encoding[:5]}...") # Print first few digits of encoding
                    
                    if total_recorded > 0:
                        print(f"Currently recorded {len(MASTER_FACE_ENCODINGS)} face information")
                    else:
                        print("No face detected, cannot record face information. Please ensure you are facing the camera.")
                        
            elif key == ord('n'): # Press 'n' to proceed to next step
                if not MASTER_FACE_RECORDED and len(MASTER_FACE_ENCODINGS) > 0:
                    MASTER_FACE_RECORDED = True
                    print(f"\n=== Proceeding to Step 2: Skeleton Recording ===")
                    print(f"Face recording complete! {len(MASTER_FACE_ENCODINGS)} faces recorded.")
                    print("Now move away from camera and make 'hands on hips' gesture.")
                elif MASTER_FACE_RECORDED and not MASTER_SKELETON_RECORDED:
                    print("Please complete skeleton recording first by making 'hands on hips' gesture.")
                else:
                    print("Already in normal tracking mode.")
                    
            elif key == ord('r'): # Press 'r' to reset records
                MASTER_FACE_ENCODINGS = []
                MASTER_ID = -1
                MASTER_POSE_BUFFER = []
                MASTER_FACE_RECORDED = False
                MASTER_SKELETON_RECORDED = False
                is_in_occlusion_mode = False
                occlusion_detection_buffer.clear()
                occlusion_start_time = None
                step2_introduced = False  # Reset step 2 introduction flag
                print("--- All recorded information and master status has been reset ---")
                print("=== Back to Step 1: Face Recording ===")
            elif key == ord('q'): # Press 'q' to quit
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()