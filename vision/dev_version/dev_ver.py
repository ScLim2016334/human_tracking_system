import cv2
import mediapipe as mp
import numpy as np
from ultralytics import YOLO # For general object detection (people, faces)
import pyttsx3 # For text-to-speech
import face_recognition # For face recognition and re-identification
import time # For precise timing in occlusion
import threading
import simpleaudio as sa # For the "beep" sound
import queue # For thread-safe communication

# --- MediaPipe Initialization ---
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# --- Text-to-Speech Initialization ---
engine = pyttsx3.init()
engine.setProperty('rate', 150) # Speed of speech
engine.setProperty('volume', 0.9) # Volume (0.0 to 1.0)

# --- Beep Sound Initialization ---
try:
    frequency = 440  # Hz
    duration = 0.2   # seconds
    samplerate = 44100 # Hz
    t = np.linspace(0, duration, int(samplerate * duration), False)
    audio_data = 0.5 * np.sin(2 * np.pi * frequency * t) # Amplitude 0.5
    beep_wave = audio_data.astype(np.float32)
    beep_wave_int16 = (beep_wave * 32767).astype(np.int16)
except Exception as e:
    print(f"Could not prepare beep sound: {e}. Beeps will be announced verbally.")
    beep_wave_int16 = None

def play_beep():
    """Plays a short beep sound in a non-blocking way."""
    if beep_wave_int16 is not None:
        try:
            play_obj = sa.play_buffer(beep_wave_int16, 1, 2, samplerate)
        except Exception as e:
            print(f"Error playing beep sound: {e}")
    else:
        announce("Beep!") # Fallback if beep generation failed

def announce(text):
    """Speaks the given text in a non-blocking way to prevent UI freezing."""
    print(f"ANNOUNCEMENT: {text}")
    
    def speak():
        try:
            engine.say(text)
            engine.runAndWait()
        except Exception as e:
            print(f"TTS Error: {e}")
    
    tts_thread = threading.Thread(target=speak)
    tts_thread.daemon = True
    tts_thread.start()

# --- Global Variables and Configuration Parameters ---
# Skeleton tracker related (for MediaPipe poses)
tracked_skeletons = {}  # {skeleton_id: {'last_pose': [landmarks], 'bbox': [x,y,w,h], 'last_seen_frame': frame_idx, 'is_master': False}}
next_id = 0             # Next available skeleton ID
MAX_DIST_THRESHOLD = 80 # Skeleton center point distance threshold for ID matching (pixels, adjust based on actual conditions)
MAX_MISS_FRAMES = 15    # Remove ID after skeleton disappears for how many frames, prevent ID accumulation

# Master calibration related
MASTER_ID = -1          # Current master skeleton ID (-1 means not set)
MASTER_FACE_ENCODINGS = [] # Store multiple master face encodings for better recognition
MASTER_FACE_RECORDED = False # Whether master face has been recorded
MASTER_SKELETON_RECORDED = False # Whether master skeleton has been recorded

# Occlusion handling related (IoU-based)
OCCLUSION_IOU_THRESHOLD = 0.4 # IoU threshold to consider an occlusion
OCCLUSION_STOP_SECONDS = 3 # Time in seconds to stop and wait after occlusion
OCCLUSION_START_TIME = None # Timestamp when occlusion started
IS_OCCLUDED_STATE = False # State variable to explicitly track if system is in occlusion handling phase

# Detection thresholds
YOLO_PERSON_CONFIDENCE_THRESHOLD = 0.5 # Minimum confidence for YOLO to detect a person
FACE_RECOGNITION_TOLERANCE = 0.3 # Face recognition tolerance, smaller value = stricter
MIN_PERSON_AREA_THRESHOLD = 0.01 # Minimum relative area (bbox_area / image_area) for a person to be considered significant

# State variables for the main loop
current_state = "STEP1_FACE_RECORDING"  # STEP1_FACE_RECORDING, STEP2_SKELETON_SELECTION, STEP3_TRACKING
STEP1_RECORDED_FACES_COUNT = 0
STEP1_FACE_POSITIONS_TO_RECORD = ["left profile", "front face", "right profile"]
STEP1_CURRENT_FACE_POSITION_IDX = 0
STEP1_LAST_INSTRUCTION_TIME = 0
STEP1_INSTRUCTION_INTERVAL = 3 # Seconds between instructions
STEP1_RETRY_DELAY = 1 # Seconds to wait before retrying a face capture
STEP1_CAPTURE_WINDOW_OPEN = False # Flag to indicate when to try capturing a face
STEP1_FACE_DETECTION_INTERVAL = 3 # Frames interval for worker to process new face detection


# Shared cache for face detection results from worker thread
step1_cached_face_data = {'locations': [], 'encodings': [], 'frame_count': -1}
step1_cached_face_data_lock = threading.Lock() # To protect access to shared cache


# STEP3 master direction detection variables
master_last_center_x = None   # Track master's last horizontal position
master_disappeared_direction = None   # "LEFT", "RIGHT", "CENTER", or None (will remove "CENTER")

# --- Face Detection Worker Thread ---
class FaceDetectionWorker(threading.Thread):
    def __init__(self, input_queue, output_queue, stop_event, interval_frames):
        super().__init__()
        self.input_queue = input_queue
        self.output_queue = output_queue
        self.stop_event = stop_event
        self.interval_frames = interval_frames
        self.daemon = True # Allow the program to exit even if this thread is still running
        print("FaceDetectionWorker initialized.")

    def run(self):
        last_processed_frame = -1
        while not self.stop_event.is_set():
            try:
                # Get the latest frame and its frame_count, non-blocking
                frame_rgb, frame_count = self.input_queue.get(timeout=0.1) # Short timeout
                
                if frame_count - last_processed_frame >= self.interval_frames:
                    # Perform face detection and encoding
                    face_locations = face_recognition.face_locations(frame_rgb, model="cnn")
                    face_encodings = face_recognition.face_encodings(frame_rgb, face_locations)
                    
                    # Put results in the output queue, overwrite if full to always get latest
                    try:
                        self.output_queue.put_nowait((face_locations, face_encodings, frame_count))
                    except queue.Full:
                        # If queue is full, just empty it and put the new one
                        while not self.output_queue.empty():
                            self.output_queue.get_nowait()
                        self.output_queue.put_nowait((face_locations, face_encodings, frame_count))
                    
                    last_processed_frame = frame_count
                
                self.input_queue.task_done() # Mark task as done
            except queue.Empty:
                time.sleep(0.01) # No frame, sleep briefly
            except Exception as e:
                print(f"FaceDetectionWorker error: {e}")
                time.sleep(0.1) # Wait before retrying on error


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
    """
    Update and maintain MediaPipe skeleton IDs.
    This function focuses on MediaPipe skeleton tracking and ID assignment/persistence.
    The 'is_master' flag of tracked skeletons is managed by the main state machine,
    not directly by this function for core re-identification logic.
    """
    global next_id, tracked_skeletons, MASTER_ID

    new_tracked_skeletons = {}
    
    for current_idx, (landmarks, pose_results) in enumerate(current_skeletons_data):
        current_center = np.mean([[lm.x, lm.y] for lm in landmarks.landmark], axis=0) * np.array([image_width, image_height])
        current_bbox = get_bbox_from_landmarks(landmarks, image_width, image_height)

        best_match_id = -1
        min_dist = float('inf')

        # Try to match with existing tracked skeletons
        for s_id, s_data in tracked_skeletons.items():
            last_center = np.mean([[lm.x, lm.y] for lm in s_data['last_pose'].pose_landmarks.landmark], axis=0) * np.array([image_width, image_height])
            dist = np.linalg.norm(current_center - last_center)
            
            if dist < min_dist and dist < MAX_DIST_THRESHOLD:
                min_dist = dist
                best_match_id = s_id
        
        if best_match_id != -1 and best_match_id not in new_tracked_skeletons:
            # If a match is found and not already claimed by another current skeleton
            new_tracked_skeletons[best_match_id] = {
                'last_pose': pose_results,
                'bbox': current_bbox,
                'last_seen_frame': frame_idx,
                'is_master': tracked_skeletons[best_match_id].get('is_master', False), # Preserve master status if existed
            }
        else:
            # No match or matched ID already taken, assign a new ID
            new_tracked_skeletons[next_id] = {
                'last_pose': pose_results,
                'bbox': current_bbox,
                'last_seen_frame': frame_idx,
                'is_master': False, # New skeletons are not master by default
            }
            next_id += 1

    # Prune old skeletons (only remove if not seen for too long AND not part of new detections)
    final_tracked_skeletons = {}
    for s_id, s_data in new_tracked_skeletons.items():
        if frame_idx - s_data['last_seen_frame'] <= MAX_MISS_FRAMES:
            final_tracked_skeletons[s_id] = s_data
        else:
            # If a skeleton (including potentially the master) is lost by MediaPipe's internal tracker,
            # its `is_master` status is implicitly handled by the main state machine.
            if s_id == MASTER_ID and s_data.get('is_master', False):
                # Optionally print a debug message if master skeleton is truly lost by MediaPipe
                # print(f"MediaPipe internal tracker: Master ID {s_id} no longer consistently seen.")
                s_data['is_master'] = False # Mark as not master in this list if it was.
                # However, the skeleton might still be needed for historical context if it's the master
                # and just stepped out for a moment. So, remove only if REALLY old.
                if frame_idx - s_data['last_seen_frame'] > MAX_MISS_FRAMES * 2: # Keep master longer
                     pass # Don't add to final_tracked_skeletons, effectively remove
                else:
                    final_tracked_skeletons[s_id] = s_data # Keep for longer if master
            else: # Not master, remove if not seen
                 pass # Don't add to final_tracked_skeletons, effectively remove
    
    # Ensure any previous master (if it's not in new_tracked_skeletons but still within MAX_MISS_FRAMES)
    # is carried over correctly. This prevents instant removal if MediaPipe has a brief glitch.
    for s_id, s_data in tracked_skeletons.items():
        if s_id not in final_tracked_skeletons and frame_idx - s_data['last_seen_frame'] <= MAX_MISS_FRAMES:
             final_tracked_skeletons[s_id] = s_data


    tracked_skeletons = final_tracked_skeletons

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

def detect_master_disappearance_direction(image_width):
    """
    Detect which direction the master disappeared from (LEFT or RIGHT) based on last known position.
    Removed "CENTER" disappearance as it's often confused with occlusion.
    """
    global master_last_center_x, master_disappeared_direction
    
    if master_last_center_x is None:
        return None # Cannot determine direction if no last known position
    
    screen_center_x = image_width / 2
    # Define thresholds as "near edges" for a distinct "disappeared out this side" message
    left_edge_zone_threshold = screen_center_x * 0.3 # If master was in left 30%
    right_edge_zone_threshold = screen_center_x * 1.7 # If master was in right 30%

    if master_last_center_x < left_edge_zone_threshold:
        master_disappeared_direction = "LEFT"
        return "LEFT"
    elif master_last_center_x > right_edge_zone_threshold:
        master_disappeared_direction = "RIGHT"
        return "RIGHT"
    else:
        # If master disappeared from the center region, we don't give a specific direction
        # as it's more ambiguous (could be directly away, or occluded).
        # We rely on occlusion logic or "lost, no specific direction" in these cases.
        master_disappeared_direction = None # Reset if it was set
        return None # Indicate no clear left/right disappearance

def update_master_position_tracking(master_bbox):
    """Update master's position for direction tracking (STEP3 only)."""
    global master_last_center_x
    
    if master_bbox and len(master_bbox) >= 4:
        master_last_center_x = master_bbox[0] + master_bbox[2] / 2
    else:
        master_last_center_x = None # Reset if master bbox is invalid/missing

# --- Main Program Logic ---
def main():
    global MASTER_ID, MASTER_FACE_ENCODINGS, next_id, tracked_skeletons
    global MASTER_FACE_RECORDED, MASTER_SKELETON_RECORDED
    global current_state, OCCLUSION_START_TIME, IS_OCCLUDED_STATE
    global STEP1_RECORDED_FACES_COUNT, STEP1_CURRENT_FACE_POSITION_IDX, STEP1_LAST_INSTRUCTION_TIME
    global STEP1_FACE_POSITIONS_TO_RECORD, step1_cached_face_data, STEP1_CAPTURE_WINDOW_OPEN
    global master_last_center_x, master_disappeared_direction

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

    print("\n--- Autonomous Human Following Algorithm ---")
    announce("Starting human following system. Please stand in front of the camera for face recording.")
    print("=== STEP 1: Record Master Face ===")
    announce("I will now capture your face. Please look to your left. Then straight. Then to your right. I will beep before each capture.")
    print("1. Please look to your left, then straight, then to your right when prompted by the system.")
    print("2. Listen for the beep before each capture.")
    print("3. Press 'r' to reset face records if needed.")
    
    # Initialize these here, as they were causing NameError in helper functions' first call
    global MASTER_LAST_KNOWN_POSITION, MASTER_OCCLUSION_FRAMES
    MASTER_LAST_KNOWN_POSITION = None
    MASTER_OCCLUSION_FRAMES = 0 # MediaPipe's internal frame counter for master loss.


    yolo_model = None # Initialize YOLO model as None, load only when needed

    # Setup worker thread for face detection
    face_input_queue = queue.Queue(maxsize=1) # Only store the latest frame
    face_output_queue = queue.Queue(maxsize=1) # Only store the latest detection results
    face_worker_stop_event = threading.Event()
    face_detection_worker = FaceDetectionWorker(face_input_queue, face_output_queue, face_worker_stop_event, STEP1_FACE_DETECTION_INTERVAL)
    face_detection_worker.start()

    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        frame_count = 0
        step2_introduced = False
        
        # FPS monitoring
        fps_start_time = time.time()
        fps_frame_count = 0
        current_fps = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1
            
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image_rgb.flags.writeable = False # Read-only for performance
            
            # MediaPipe pose detection (runs on all frames)
            pose_results = pose.process(image_rgb)
            
            image_rgb.flags.writeable = True # Make writable for drawing
            image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

            current_skeletons_in_frame_data = []
            if pose_results.pose_landmarks:
                current_skeletons_in_frame_data.append((pose_results.pose_landmarks, pose_results))

            # Update MediaPipe skeleton IDs (this runs regardless of state)
            update_skeleton_ids(current_skeletons_in_frame_data, frame_count, image_width, image_height)
            
            # --- Main State Machine Logic ---
            current_time = time.time()

            if current_state == "STEP1_FACE_RECORDING":
                # Feed frame to face detection worker (non-blocking)
                try:
                    face_input_queue.put_nowait((image_rgb.copy(), frame_count)) # Send a copy
                except queue.Full:
                    pass # Worker is busy, skip this frame for face detection input

                # Try to get latest processed face data from worker (non-blocking)
                try:
                    locs, encs, worker_frame_count = face_output_queue.get_nowait()
                    with step1_cached_face_data_lock:
                        step1_cached_face_data['locations'] = locs
                        step1_cached_face_data['encodings'] = encs
                        step1_cached_face_data['frame_count'] = worker_frame_count
                except queue.Empty:
                    pass # No new data from worker yet

                # Retrieve face data for display/processing
                with step1_cached_face_data_lock:
                    face_locations = step1_cached_face_data['locations']
                    face_encodings = step1_cached_face_data['encodings']

                # Autonomous instruction and capture logic
                if STEP1_CURRENT_FACE_POSITION_IDX < len(STEP1_FACE_POSITIONS_TO_RECORD):
                    position_to_record = STEP1_FACE_POSITIONS_TO_RECORD[STEP1_CURRENT_FACE_POSITION_IDX]
                    
                    if current_time - STEP1_LAST_INSTRUCTION_TIME > STEP1_INSTRUCTION_INTERVAL:
                        announce(f"Now capturing your {position_to_record}. Please look at the camera.")
                        STEP1_LAST_INSTRUCTION_TIME = current_time
                        STEP1_CAPTURE_WINDOW_OPEN = True # Open window for capture
                    
                    if STEP1_CAPTURE_WINDOW_OPEN and (current_time - STEP1_LAST_INSTRUCTION_TIME < STEP1_INSTRUCTION_INTERVAL + STEP1_RETRY_DELAY):
                        if face_encodings:
                            # Assuming the largest face is the one to record
                            largest_face_idx = np.argmax([(loc[2] - loc[0]) * (loc[1] - loc[3]) for loc in face_locations])
                            largest_face_encoding = face_encodings[largest_face_idx]
                            
                            MASTER_FACE_ENCODINGS.append(largest_face_encoding)
                            STEP1_RECORDED_FACES_COUNT += 1
                            print(f"Captured {position_to_record}! Total recorded: {len(MASTER_FACE_ENCODINGS)}")
                            play_beep() # Indicate successful capture
                            
                            STEP1_CURRENT_FACE_POSITION_IDX += 1 # Move to next position
                            STEP1_LAST_INSTRUCTION_TIME = current_time # Reset timer for next instruction
                            STEP1_CAPTURE_WINDOW_OPEN = False # Close window after capture
                            time.sleep(0.5) # Short pause after capture for user feedback

                            if STEP1_RECORDED_FACES_COUNT == len(STEP1_FACE_POSITIONS_TO_RECORD):
                                current_state = "STEP2_SKELETON_RECORDING"
                                MASTER_FACE_RECORDED = True
                                announce("All face positions captured. Proceeding to automatic skeleton selection.")
                                print("All face positions captured. Automatically moving to STEP2_SKELETON_SELECTION.")
                        # else: If no face detected, the window stays open for a short retry delay until next instruction
                else: # All positions recorded, transition logic
                    current_state = "STEP2_SKELETON_RECORDING"
                    MASTER_FACE_RECORDED = True
                    announce("Face recording complete. Proceeding to automatic skeleton selection.")
                    print("Face recording complete. Automatically moving to STEP2_SKELETON_SELECTION.")
                    time.sleep(0.5)

                # Display info for Step 1
                cv2.putText(image_bgr, "STEP 1: Autonomous Face Recording", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)
                cv2.putText(image_bgr, f"Recorded faces: {len(MASTER_FACE_ENCODINGS)} / {len(STEP1_FACE_POSITIONS_TO_RECORD)}", (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 1, cv2.LINE_AA)
                if STEP1_CURRENT_FACE_POSITION_IDX < len(STEP1_FACE_POSITIONS_TO_RECORD):
                    cv2.putText(image_bgr, f"Capturing: {STEP1_FACE_POSITIONS_TO_RECORD[STEP1_CURRENT_FACE_POSITION_IDX]}", (10, 90),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 1, cv2.LINE_AA)
                
                if not face_encodings:
                    cv2.putText(image_bgr, "No Face Detected. Please stand in front of camera.", (10, 120),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_AA)

                # Draw detected faces
                for (top, right, bottom, left) in face_locations:
                    cv2.rectangle(image_bgr, (left, top), (right, bottom), (0, 255, 0), 2)
                    cv2.putText(image_bgr, "Face Detected", (left, top - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)

                # Draw MediaPipe skeletons (gray)
                for s_id, s_data in tracked_skeletons.items():
                    color = (128, 128, 128)
                    label = f"ID: {s_id}"
                    mp_drawing.draw_landmarks(image_bgr, s_data['last_pose'].pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                             mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2),
                                             mp_drawing.DrawingSpec(color=(128,128,128), thickness=2, circle_radius=2))
                    x, y, w, h = s_data['bbox']
                    cv2.rectangle(image_bgr, (x, y), (x + w, y + h), color, 2)
                    cv2.putText(image_bgr, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

            elif current_state == "STEP2_SKELETON_RECORDING":
                # STEP 2: Automatic Master Skeleton Selection
                if not step2_introduced:
                    print("\n=== STEP 2: Select Master Skeleton (Closest Person) ===")
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
                            # Update MASTER_LAST_KNOWN_POSITION at calibration
                            # This is for MediaPipe's internal check, but also for disappearance direction
                            MASTER_LAST_KNOWN_POSITION = [master_bbox[0] + master_bbox[2] // 2, 
                                                          master_bbox[1] + master_bbox[3] // 2]
                            
                            current_state = "STEP3_TRACKING"
                            announce("Master skeleton selected! Starting enhanced tracking mode.")
                            print(f"--- Master skeleton {MASTER_ID} automatically selected! ---")
                        else:
                            # Re-selection if already recorded but current master lost
                            if MASTER_ID not in tracked_skeletons or not tracked_skeletons[MASTER_ID]['is_master']:
                                MASTER_ID = closest_skeleton_id
                                tracked_skeletons[MASTER_ID]['is_master'] = True
                                master_bbox = tracked_skeletons[MASTER_ID]['bbox']
                                MASTER_LAST_KNOWN_POSITION = [master_bbox[0] + master_bbox[2] // 2, 
                                                              master_bbox[1] + master_bbox[3] // 2]
                                announce("Master re-selected. Resuming tracking.")
                                print(f"--- Master skeleton re-selected to {MASTER_ID} ---")
                            current_state = "STEP3_TRACKING" # Ensure state transition happens

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
                
                # Draw YOLO person bounding boxes for context
                if yolo_model is None: # Lazy load YOLO for display if needed here
                     try:
                        yolo_model = YOLO('yolov8n.pt')
                     except Exception as e:
                        print(f"Error loading YOLO model in S2 for display: {e}")
                        yolo_model = None

                if yolo_model:
                    person_detections_step2 = get_person_detections(yolo_model, image_bgr, 0.4)
                    for p_det in person_detections_step2:
                        x, y, w, h = p_det['bbox']
                        cv2.rectangle(image_bgr, (x, y), (x + w, y + h), (100, 100, 255), 1)
                        cv2.putText(image_bgr, f"Person (YOLO)", (x, y + h + 20),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 255), 1, cv2.LINE_AA)

            elif current_state == "STEP3_TRACKING":
                if yolo_model is None:
                    try:
                        print("Loading YOLO model for enhanced tracking mode...")
                        yolo_model = YOLO('yolov8n.pt')
                        print("YOLO model loaded successfully!")
                    except Exception as e:
                        print(f"Error loading YOLO model: {e}")
                        announce("Error: Could not load object detection model.")
                        current_state = "STEP2_SKELETON_RECORDING" # Fallback to re-calibration
                        continue

                cv2.putText(image_bgr, "STEP 3: Enhanced Tracking (MediaPipe + YOLO)", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

                # Get YOLO person detections for both occlusion checks and drawing
                person_detections = get_person_detections(yolo_model, image_bgr, YOLO_PERSON_CONFIDENCE_THRESHOLD)
                significant_persons = filter_significant_persons(person_detections, image_width, image_height, MIN_PERSON_AREA_THRESHOLD)

                # Check if master is currently tracked by MediaPipe
                master_skel_tracked_this_frame = (MASTER_ID != -1 and MASTER_ID in tracked_skeletons and tracked_skeletons[MASTER_ID]['is_master'])
                
                # Store whether master *was* tracked in the previous frame
                previous_master_was_tracked = (MASTER_LAST_KNOWN_POSITION is not None)

                # Update master's last known position for direction tracking IF master is visible
                if master_skel_tracked_this_frame:
                    master_skel_bbox = tracked_skeletons[MASTER_ID]['bbox']
                    update_master_position_tracking(master_skel_bbox)
                    # Clear any disappearance direction if master is now found
                    master_disappeared_direction = None 
                    MASTER_OCCLUSION_FRAMES = 0 # Reset MediaPipe's internal loss counter
                else:
                    # Master is NOT tracked by MediaPipe this frame
                    MASTER_OCCLUSION_FRAMES += 1 # Increment MediaPipe's internal loss counter
                    if previous_master_was_tracked and master_disappeared_direction is None:
                        # Only set disappearance direction once when master first disappears from MediaPipe
                        direction_of_loss = detect_master_disappearance_direction(image_width)
                        if direction_of_loss: # Only announce if a clear left/right disappearance
                            print(f"*** Master skeleton disappeared from {direction_of_loss} side ***")
                            announce(f"Master disappeared from the {direction_of_loss} side.")


                # --- Occlusion Prioritization Logic ---

                # Scenario 1: Explicit IoU Occlusion (Master is visible, another person overlaps heavily)
                occlusion_by_iou = False
                max_iou_with_other_person = 0.0 # Initialize here to ensure it's always defined
                if master_skel_tracked_this_frame: # Only check explicit IoU if master is currently visible
                    master_skel_bbox = tracked_skeletons[MASTER_ID]['bbox']
                    master_yolo_bbox_for_iou_check = None
                    # Try to find the YOLO bbox that strongly corresponds to the master skeleton
                    best_iou_to_master_skel = 0.0
                    for yolo_p_det in significant_persons:
                        iou_val = calculate_iou(master_skel_bbox, yolo_p_det['bbox'])
                        if iou_val > best_iou_to_master_skel and iou_val > 0.3: # Require a decent overlap
                            best_iou_to_master_skel = iou_val
                            master_yolo_bbox_for_iou_check = yolo_p_det['bbox']

                    if master_yolo_bbox_for_iou_check: # If we found a YOLO box for the master
                        for p_det_other in significant_persons:
                            if p_det_other['bbox'] != master_yolo_bbox_for_iou_check: # Compare with other people
                                iou_val_with_other = calculate_iou(master_yolo_bbox_for_iou_check, p_det_other['bbox'])
                                max_iou_with_other_person = max(max_iou_with_other_person, iou_val_with_other)
                                if iou_val_with_other > OCCLUSION_IOU_THRESHOLD:
                                    occlusion_by_iou = True
                                    break # Exit inner loop, occlusion confirmed
                
                # Scenario 2: Implied Occlusion (Master was just tracked, but now MediaPipe lost it suddenly)
                # This catches quick occlusions that might not register high IoU for a full frame.
                implied_occlusion_from_loss = (not master_skel_tracked_this_frame and previous_master_was_tracked)
                
                # Decision to enter OCCLUSION_WAIT state
                if (occlusion_by_iou or implied_occlusion_from_loss) and not IS_OCCLUDED_STATE:
                    IS_OCCLUDED_STATE = True # Enter occlusion state
                    OCCLUSION_START_TIME = time.time()
                    announce(f"Occlusion detected! Master please stop for {OCCLUSION_STOP_SECONDS} seconds.")
                    print(f"Occlusion detected! Robot will stop and wait for re-identification.")
                    if occlusion_by_iou:
                        print(f"Reason: Explicit IoU overlap (max IoU with other: {max_iou_with_other_person:.2f}).") 
                    elif implied_occlusion_from_loss:
                        print("Reason: Master skeleton suddenly lost from view (implied occlusion).")
                
                # Update status display based on current state
                if IS_OCCLUDED_STATE:
                    elapsed_time = time.time() - OCCLUSION_START_TIME
                    remaining_time = max(0, OCCLUSION_STOP_SECONDS - elapsed_time)
                    cv2.putText(image_bgr, f"Master Status: Occluded! Waiting: {remaining_time:.1f}s", (10, 60),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2) # Orange for occlusion
                    if remaining_time <= 0:
                        # Time's up, exit occlusion state and enter re-identification process
                        IS_OCCLUDED_STATE = False
                        OCCLUSION_START_TIME = None
                        MASTER_ID = -1 # Force re-identification of master skeleton
                        announce("Occlusion wait finished. Initiating re-identification.")
                        print("Occlusion wait period ended. Beginning re-identification process.")
                        # This will now fall through to the re-identification logic below.
                elif master_skel_tracked_this_frame:
                    cv2.putText(image_bgr, f"Master Status: Tracking (ID: {MASTER_ID})", (10, 60),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2) # Red for master
                    if master_last_center_x is not None:
                        cv2.putText(image_bgr, f"Master X: {master_last_center_x:.0f}", (10, 90),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                else: # Master is lost by MediaPipe and not in explicit occlusion wait
                    cv2.putText(image_bgr, "Master Status: Lost, Re-identifying...", (10, 60),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2) # Yellow for lost
                    cv2.putText(image_bgr, f"Lost for: {MASTER_OCCLUSION_FRAMES} frames", (10, 90),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
                    if master_disappeared_direction: # Only if it disappeared left/right
                        cv2.putText(image_bgr, f"Last Direction: {master_disappeared_direction}", (10, 110),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 100, 100), 1)

                # --- Re-identification Logic (when master is not tracked AND not in OCCLUSION_WAIT) ---
                if not master_skel_tracked_this_frame and not IS_OCCLUDED_STATE:
                    # Get the latest face data from worker for re-identification
                    with step1_cached_face_data_lock: # Using step1_cached_face_data as general face cache
                        face_locations_full_img = step1_cached_face_data['locations']
                        face_encodings_full_img = step1_cached_face_data['encodings']

                    if len(significant_persons) == 1:
                        # Scenario 1: Only one significant person detected by YOLO
                        solo_person_yolo_bbox = significant_persons[0]['bbox']
                        
                        master_reidentified_via_face = False
                        if face_encodings_full_img: # Only proceed if worker found faces recently
                            for i, (top, right, bottom, left) in enumerate(face_locations_full_img):
                                face_bbox_check = [left, top, right - left, bottom - top]
                                if calculate_iou(face_bbox_check, solo_person_yolo_bbox) > 0.5:
                                    is_master_face, best_distance = recognize_master_face([face_encodings_full_img[i]])
                                    if is_master_face:
                                        best_skel_match_id = -1
                                        max_skel_iou = 0
                                        for s_id, s_data in tracked_skeletons.items():
                                            skel_iou = calculate_iou(s_data['bbox'], solo_person_yolo_bbox)
                                            if skel_iou > max_skel_iou:
                                                max_skel_iou = skel_iou
                                                best_skel_match_id = s_id
                                        
                                        if best_skel_match_id != -1 and max_skel_iou > 0.3:
                                            MASTER_ID = best_skel_match_id
                                            tracked_skeletons[MASTER_ID]['is_master'] = True
                                            master_bbox = tracked_skeletons[MASTER_ID]['bbox']
                                            # Update last known position and reset loss counter
                                            MASTER_LAST_KNOWN_POSITION = [master_bbox[0] + master_bbox[2] // 2, 
                                                                          master_bbox[1] + master_bbox[3] // 2]
                                            MASTER_OCCLUSION_FRAMES = 0
                                            announce("Master re-identified! Following.")
                                            print(f"Master re-identified (ID: {MASTER_ID}) via single person YOLO and face match.")
                                            master_reidentified_via_face = True
                                            break
                        
                        if not master_reidentified_via_face:
                            print("Single person detected, but master face not confirmed. Attempting skeleton-based re-id.")
                            best_skel_match_id = -1
                            max_skel_iou = 0
                            for s_id, s_data in tracked_skeletons.items():
                                skel_iou = calculate_iou(s_data['bbox'], solo_person_yolo_bbox)
                                if skel_iou > max_skel_iou:
                                    max_skel_iou = skel_iou
                                    best_skel_match_id = s_id
                            
                            if best_skel_match_id != -1 and max_skel_iou > 0.3:
                                MASTER_ID = best_skel_match_id
                                tracked_skeletons[MASTER_ID]['is_master'] = True
                                master_bbox = tracked_skeletons[MASTER_ID]['bbox']
                                MASTER_LAST_KNOWN_POSITION = [master_bbox[0] + master_bbox[2] // 2, 
                                                              master_bbox[1] + master_bbox[3] // 2]
                                MASTER_OCCLUSION_FRAMES = 0
                                announce("Assuming solo person is master. Following.")
                                print(f"Master re-identified (ID: {MASTER_ID}) via single person YOLO (no face match).")
                            else:
                                announce("Master not found. Please come in front of me.")
                                print("No suitable MediaPipe skeleton found for solo YOLO person.")
                        
                    elif len(significant_persons) > 1:
                        # Scenario 2: Multiple significant people detected
                        announce("Multiple people detected. Master, please look at me.")
                        cv2.putText(image_bgr, "Master, please look at me.", (image_width // 2 - 150, image_height - 50),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA) # Cyan instruction

                        master_reidentified_via_face_in_crowd = False
                        if face_encodings_full_img: # Only proceed if worker found faces recently
                            for i, (face_top, face_right, face_bottom, face_left) in enumerate(face_locations_full_img):
                                is_master_face, best_distance = recognize_master_face([face_encodings_full_img[i]])
                                if is_master_face:
                                    face_bbox_reid = [face_left, face_top, face_right - face_left, face_bottom - face_top]
                                    
                                    # Try to link this master face to a YOLO person detection, then to MediaPipe skeleton
                                    best_yolo_match = None
                                    best_yolo_iou = 0
                                    for p_det in significant_persons:
                                        iou_with_yolo = calculate_iou(face_bbox_reid, p_det['bbox'])
                                        if iou_with_yolo > best_yolo_iou:
                                            best_yolo_iou = iou_with_yolo
                                            best_yolo_match = p_det['bbox']
                                    
                                    if best_yolo_match and best_yolo_iou > 0.3: # Link face to a YOLO person
                                        # Now find the MediaPipe skeleton closest to this matched YOLO person
                                        best_skel_match_id = -1
                                        max_skel_iou = 0
                                        for s_id, s_data in tracked_skeletons.items():
                                            skel_iou = calculate_iou(s_data['bbox'], best_yolo_match)
                                            if skel_iou > max_skel_iou:
                                                max_skel_iou = skel_iou
                                                best_skel_match_id = s_id
                                        
                                        if best_skel_match_id != -1 and max_skel_iou > 0.3: # Link YOLO person to MediaPipe skeleton
                                            MASTER_ID = best_skel_match_id
                                            tracked_skeletons[MASTER_ID]['is_master'] = True
                                            master_bbox = tracked_skeletons[MASTER_ID]['bbox']
                                            MASTER_LAST_KNOWN_POSITION = [master_bbox[0] + master_bbox[2] // 2, 
                                                                          master_bbox[1] + master_bbox[3] // 2]
                                            MASTER_OCCLUSION_FRAMES = 0
                                            announce("Master re-identified! Following.")
                                            print(f"Master re-identified (ID: {MASTER_ID}) through face recognition in crowd.")
                                            master_reidentified_via_face_in_crowd = True
                                            break # Master re-identified, exit face loop
                        
                        if not master_reidentified_via_face_in_crowd:
                            print("Master face not found among multiple people.")
                            cv2.putText(image_bgr, "Waiting for Master's face...", (image_width // 2 - 150, image_height - 20),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)

                    elif len(significant_persons) == 0:
                        # Scenario 3: No significant people detected
                        announce("Master, I can't see you. Please come in front of me and look at me.")
                        cv2.putText(image_bgr, "Master, I can't see you. Please come in front of me and look at me.", (image_width // 2 - 350, image_height - 50),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA) # Red instruction

                        master_reidentified_no_people = False
                        if face_encodings_full_img: # Only proceed if worker found faces recently
                            for i, (face_top, face_right, face_bottom, face_left) in enumerate(face_locations_full_img):
                                is_master_face, best_distance = recognize_master_face([face_encodings_full_img[i]])
                                if is_master_face:
                                    face_bbox_reid = [face_left, face_top, face_right - face_left, face_bottom - face_top]
                                    # Find the closest MediaPipe skeleton to this master face
                                    best_skel_match_id = -1
                                    max_skel_iou = 0
                                    for s_id, s_data in tracked_skeletons.items():
                                        skel_bbox = s_data['bbox']
                                        skel_head_bbox = [skel_bbox[0], skel_bbox[1], skel_bbox[2], skel_bbox[3] // 3] # Approx head region
                                        skel_iou = calculate_iou(face_bbox_reid, skel_head_bbox)
                                        if skel_iou > max_skel_iou:
                                            max_skel_iou = skel_iou
                                            best_skel_match_id = s_id
                                    
                                    if best_skel_match_id != -1 and max_skel_iou > 0.1:
                                        MASTER_ID = best_skel_match_id
                                        tracked_skeletons[MASTER_ID]['is_master'] = True
                                        master_bbox = tracked_skeletons[MASTER_ID]['bbox']
                                        MASTER_LAST_KNOWN_POSITION = [master_bbox[0] + master_bbox[2] // 2, 
                                                                      master_bbox[1] + master_bbox[3] // 2]
                                        MASTER_OCCLUSION_FRAMES = 0
                                        announce("Master re-identified! Following.")
                                        print(f"Master re-identified (ID: {MASTER_ID}) through face recognition (no people in sight).")
                                        master_reidentified_no_people = True
                                        break
                        if not master_reidentified_no_people:
                            print("Master face not found when no people are visible.")
                            cv2.putText(image_bgr, "Still looking for Master's face...", (image_width // 2 - 150, image_height - 20),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)

                # --- Drawing for Normal Tracking Mode (MediaPipe & YOLO) ---
                for s_id, s_data in tracked_skeletons.items():
                    color = (0, 255, 0) # Green for others
                    label = f"ID: {s_id}"
                    if s_data['is_master']:
                        color = (0, 0, 255) # Red for master
                        label = f"Master ID: {s_id}"
                    
                    # Draw MediaPipe skeleton
                    mp_drawing.draw_landmarks(image_bgr, s_data['last_pose'].pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                             mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2),
                                             mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2))
                    
                    x, y, w, h = s_data['bbox']
                    cv2.rectangle(image_bgr, (x, y), (x + w, y + h), color, 2)
                    cv2.putText(image_bgr, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

                # Draw YOLO person detections with color coding
                for p_det in person_detections:
                    x, y, w, h = p_det['bbox']
                    color = (0, 255, 0) # Green for others by default
                    label = f"YOLO: {p_det['conf']:.2f}"
                    
                    # If master is currently tracked by MediaPipe, try to match its YOLO bbox to draw red
                    if master_skel_tracked_this_frame:
                        master_skel_bbox = tracked_skeletons[MASTER_ID]['bbox']
                        # Check for significant overlap or if MediaPipe skeleton center is inside YOLO box
                        center_x_skel = master_skel_bbox[0] + master_skel_bbox[2] // 2
                        center_y_skel = master_skel_bbox[1] + master_skel_bbox[3] // 2
                        
                        if calculate_iou(p_det['bbox'], master_skel_bbox) > 0.3 or \
                           (x < center_x_skel < x + w and y < center_y_skel < y + h):
                            color = (0, 0, 255) # Red for master
                            label = f"YOLO Master: {p_det['conf']:.2f}"
                    
                    cv2.rectangle(image_bgr, (x, y), (x + w, y + h), color, 2)
                    cv2.putText(image_bgr, label, (x, y + h + 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

            elif IS_OCCLUDED_STATE: # This state is only entered via STEP3_TRACKING
                # IoU-based occlusion wait period
                elapsed_time = time.time() - OCCLUSION_START_TIME
                remaining_time = max(0, OCCLUSION_STOP_SECONDS - elapsed_time)
                
                cv2.putText(image_bgr, f"Occlusion Detected! Waiting: {remaining_time:.1f}s", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2)
                
                if remaining_time <= 0:
                    current_state = "STEP3_TRACKING" # Go back to tracking mode to initiate re-ID
                    IS_OCCLUDED_STATE = False
                    OCCLUSION_START_TIME = None
                    MASTER_ID = -1 # Force re-identification after wait
                    announce("Occlusion wait period ended. Initiating re-identification.")
                    print("Occlusion wait period ended. Beginning re-identification process.")
                
                # Still draw skeletons/detections during wait
                # Get YOLO person detections for drawing
                # Ensuring yolo_model is initialized if we somehow jumped to OCCLUSION_WAIT without it
                if yolo_model is None:
                    try:
                        yolo_model = YOLO('yolov8n.pt')
                    except Exception as e:
                        print(f"Error loading YOLO model in OCCLUSION_WAIT for display: {e}")
                        yolo_model = None
                
                person_detections = get_person_detections(yolo_model, image_bgr, YOLO_PERSON_CONFIDENCE_THRESHOLD) if yolo_model else []
                
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
                    # Try to match master (if master was visible when occlusion started)
                    master_is_tracked_by_mediapipe_at_start = (MASTER_ID != -1 and MASTER_ID in tracked_skeletons and tracked_skeletons[MASTER_ID].get('is_master', False))
                    if master_is_tracked_by_mediapipe_at_start: 
                        master_skel_bbox_at_start = tracked_skeletons[MASTER_ID]['bbox']
                        if calculate_iou(p_det['bbox'], master_skel_bbox_at_start) > 0.3:
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

            # Current State Display
            cv2.putText(image_bgr, f"State: {current_state}", (image_width - 250, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2, cv2.LINE_AA)

            cv2.imshow('Human Following Algorithm', image_bgr)

            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('r'):
                MASTER_FACE_ENCODINGS = []
                MASTER_ID = -1
                MASTER_FACE_RECORDED = False
                MASTER_SKELETON_RECORDED = False
                
                # Reset all dynamic state variables related to tracking
                MASTER_LAST_KNOWN_POSITION = None
                MASTER_OCCLUSION_FRAMES = 0
                tracked_skeletons = {}
                master_last_center_x = None
                master_disappeared_direction = None

                current_state = "STEP1_FACE_RECORDING"
                OCCLUSION_START_TIME = None
                IS_OCCLUDED_STATE = False
                
                # Reset STEP1 specific variables
                STEP1_RECORDED_FACES_COUNT = 0
                STEP1_CURRENT_FACE_POSITION_IDX = 0
                STEP1_LAST_INSTRUCTION_TIME = 0 # Reset instruction timer
                step1_last_face_detection_frame = 0
                step1_cached_face_data = {'locations': [], 'encodings': [], 'frame_count': -1}
                
                # Reset for the new automatic capture flow
                # STEP1_LAST_CAPTURE_TIME = 0 # This will be set by the new flow's first instruction
                # STEP1_CAPTURE_COUNT = 0 # This will be set by the new flow's first instruction
                
                # IMPORTANT: Clear queues and stop/restart worker thread properly
                face_worker_stop_event.set()
                face_detection_worker.join() # Wait for thread to finish
                while not face_input_queue.empty(): face_input_queue.get_nowait()
                while not face_output_queue.empty(): face_output_queue.get_nowait()

                # Re-create the worker thread for the next cycle
                face_worker_stop_event.clear()
                face_detection_worker = FaceDetectionWorker(face_input_queue, face_output_queue, face_worker_stop_event, STEP1_FACE_DETECTION_INTERVAL)
                face_detection_worker.start()

                yolo_model = None # YOLO will be re-initialized when STEP3 is entered again
                
                print("--- All recorded information and master status has been reset ---")
                print("=== Back to Step 1: Autonomous Face Recording ===")
                announce("System reset. Please stand in front of the camera to start face recording.")
                announce("I will now capture your face. Please look to your left. Then straight. Then to your right. I will beep before each capture.")

            elif key == ord('q'):
                break

    # Stop worker thread gracefully when the main loop exits
    face_worker_stop_event.set()
    face_detection_worker.join() # Wait for the thread to complete

    cap.release()
    cv2.destroyAllWindows()
    announce("System shutting down. Goodbye!")
    engine.stop() # Ensure the speech engine stops gracefully

if __name__ == '__main__':
    main()