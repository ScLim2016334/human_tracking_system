"""
Face Recognition Utilities
Contains functions for face detection, recognition, and re-identification
"""

import face_recognition
import numpy as np
from config import *

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

def recognize_master_face(face_encodings, master_face_encodings):
    """
    Recognize if any of the given face encodings match the master.
    Returns (is_master, best_distance) tuple.
    """
    if not master_face_encodings:
        return False, float('inf')
    
    best_distance = float('inf')
    is_master = False
    
    for face_encoding in face_encodings:
        # Compare with all recorded master face encodings
        distances = []
        matches = []
        
        for master_encoding in master_face_encodings:
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

def detect_faces_in_image(image):
    """
    Detect all faces in the image and return their locations and encodings.
    Returns (face_locations, face_encodings)
    """
    face_locations = face_recognition.face_locations(image)
    face_encodings = face_recognition.face_encodings(image, face_locations)
    return face_locations, face_encodings

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