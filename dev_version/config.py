# --- Configuration Parameters ---

# Skeleton tracker related
MAX_DIST_THRESHOLD = 80  # Skeleton center point distance threshold for ID matching (pixels)
MAX_MISS_FRAMES = 15     # Remove ID after skeleton disappears for how many frames

# Master calibration related
MASTER_HISTORY_THRESHOLD = 5  # At least 5 consecutive frames of calibration action to be considered successful

# New IoU-based occlusion handling related
OCCLUSION_IOU_THRESHOLD = 0.3  # IoU threshold to detect occlusion (when master and other skeleton overlap)
OCCLUSION_STOP_TIME = 3  # Time in seconds to ask master to stop when occlusion detected
OCCLUSION_DETECTION_FRAMES = 5  # Number of consecutive frames with IoU > threshold to confirm occlusion
DISTANCE_THRESHOLD_FOR_TRACKING = 200  # Distance threshold to consider skeleton for tracking (pixels)

# Face recognition related
FACE_RECOGNITION_TOLERANCE = 0.4  # Face recognition tolerance, smaller value = stricter

# Pose detection thresholds
MIN_DETECTION_CONFIDENCE = 0.5
MIN_TRACKING_CONFIDENCE = 0.5

# Hands on hips detection thresholds
WRIST_HIP_Y_THRESHOLD_REL = 0.08
WRIST_HIP_X_THRESHOLD_REL = 0.05
ELBOW_ANGLE_THRESHOLD_MIN = 60
ELBOW_ANGLE_THRESHOLD_MAX = 120
KEYPOINT_VISIBILITY_THRESHOLD = 0.7

# Bounding box padding
BBOX_PADDING = 20

# Re-identification thresholds
REID_IOU_THRESHOLD = 0.05
REID_SCORE_THRESHOLD = 0.1

# Voice announcement settings
ENABLE_VOICE_ANNOUNCEMENTS = True  # Set to False to disable voice announcements
VOICE_LANGUAGE = "en"  # Language for voice announcements ("en" for English, "zh" for Chinese) 