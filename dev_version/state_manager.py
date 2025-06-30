"""
State Manager for Human Tracking System
Manages all global state variables and provides methods to access/modify them
"""

import numpy as np
import time
from config import *

class StateManager:
    def __init__(self):
        # Skeleton tracker related
        self.tracked_skeletons = {}  # {skeleton_id: {'last_pose': [landmarks], 'bbox': [x,y,w,h], 'last_seen_frame': frame_idx, 'is_master': False}}
        self.next_id = 0             # Next available skeleton ID
        
        # Master calibration related
        self.master_id = -1          # Current master skeleton ID (-1 means not set)
        self.master_face_encodings = [] # Store multiple master face encodings for better recognition
        self.master_pose_buffer = [] # Store master calibration action history frame results
        self.master_face_recorded = False # Whether master face has been recorded
        self.master_skeleton_recorded = False # Whether master skeleton has been recorded
        
        # New IoU-based occlusion handling related
        self.occlusion_detected = False  # Whether occlusion is currently detected
        self.occlusion_start_time = None  # When occlusion was first detected
        self.occlusion_iou_frames = 0  # Number of consecutive frames with high IoU
        self.occlusion_stop_announced = False  # Whether stop announcement has been made
        self.occlusion_recovery_mode = False  # Whether in recovery mode after occlusion
        self.occlusion_recovery_start_time = None  # When recovery mode started
        
        # Legacy occlusion handling (kept for compatibility)
        self.master_last_known_position = None # Last known position of master before occlusion
        self.master_occlusion_frames = 0 # Number of frames master has been occluded
    
    # Skeleton tracking methods
    def get_tracked_skeletons(self):
        return self.tracked_skeletons
    
    def set_tracked_skeletons(self, skeletons):
        self.tracked_skeletons = skeletons
    
    def get_next_id(self):
        return self.next_id
    
    def increment_next_id(self):
        self.next_id += 1
    
    # Master related methods
    def get_master_id(self):
        return self.master_id
    
    def set_master_id(self, master_id):
        self.master_id = master_id
    
    def get_master_face_encodings(self):
        return self.master_face_encodings
    
    def add_master_face_encoding(self, encoding):
        self.master_face_encodings.append(encoding)
    
    def clear_master_face_encodings(self):
        self.master_face_encodings = []
    
    def get_master_pose_buffer(self):
        return self.master_pose_buffer
    
    def add_to_master_pose_buffer(self, value):
        self.master_pose_buffer.append(value)
        if len(self.master_pose_buffer) > MASTER_HISTORY_THRESHOLD:
            self.master_pose_buffer.pop(0)
    
    def clear_master_pose_buffer(self):
        self.master_pose_buffer = []
    
    def is_master_face_recorded(self):
        return self.master_face_recorded
    
    def set_master_face_recorded(self, recorded):
        self.master_face_recorded = recorded
    
    def is_master_skeleton_recorded(self):
        return self.master_skeleton_recorded
    
    def set_master_skeleton_recorded(self, recorded):
        self.master_skeleton_recorded = recorded
    
    # New IoU-based occlusion handling methods
    def is_occlusion_detected(self):
        return self.occlusion_detected
    
    def set_occlusion_detected(self, detected):
        if detected and not self.occlusion_detected:
            # Occlusion just started
            self.occlusion_start_time = time.time()
            self.occlusion_stop_announced = False
        elif not detected and self.occlusion_detected:
            # Occlusion ended, enter recovery mode
            self.occlusion_recovery_mode = True
            self.occlusion_recovery_start_time = time.time()
        
        self.occlusion_detected = detected
    
    def get_occlusion_start_time(self):
        return self.occlusion_start_time
    
    def get_occlusion_iou_frames(self):
        return self.occlusion_iou_frames
    
    def increment_occlusion_iou_frames(self):
        self.occlusion_iou_frames += 1
    
    def reset_occlusion_iou_frames(self):
        self.occlusion_iou_frames = 0
    
    def is_occlusion_stop_announced(self):
        return self.occlusion_stop_announced
    
    def set_occlusion_stop_announced(self, announced):
        self.occlusion_stop_announced = announced
    
    def is_occlusion_recovery_mode(self):
        return self.occlusion_recovery_mode
    
    def set_occlusion_recovery_mode(self, mode):
        self.occlusion_recovery_mode = mode
    
    def get_occlusion_recovery_start_time(self):
        return self.occlusion_recovery_start_time
    
    def reset_occlusion_state(self):
        """Reset all occlusion-related state"""
        self.occlusion_detected = False
        self.occlusion_start_time = None
        self.occlusion_iou_frames = 0
        self.occlusion_stop_announced = False
        self.occlusion_recovery_mode = False
        self.occlusion_recovery_start_time = None
    
    # Legacy occlusion handling methods (kept for compatibility)
    def get_master_last_known_position(self):
        return self.master_last_known_position
    
    def set_master_last_known_position(self, position):
        self.master_last_known_position = position
    
    def get_master_occlusion_frames(self):
        return self.master_occlusion_frames
    
    def increment_master_occlusion_frames(self):
        self.master_occlusion_frames += 1
    
    def reset_master_occlusion_frames(self):
        self.master_occlusion_frames = 0
    
    def reset_master_occlusion_state(self):
        self.master_last_known_position = None
        self.master_occlusion_frames = 0
    
    # Reset all state
    def reset_all(self):
        self.tracked_skeletons = {}
        self.next_id = 0
        self.master_id = -1
        self.master_face_encodings = []
        self.master_pose_buffer = []
        self.master_face_recorded = False
        self.master_skeleton_recorded = False
        self.master_last_known_position = None
        self.master_occlusion_frames = 0
        self.reset_occlusion_state()

# Global state manager instance
state_manager = StateManager() 