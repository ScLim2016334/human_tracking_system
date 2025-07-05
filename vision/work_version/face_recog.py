import cv2
import face_recognition
import numpy as np
import time

# --- Configuration Parameters ---
MASTER_FACE_ENCODING = None # Store master face encoding
FACE_RECOGNITION_TOLERANCE = 0.2 # Face recognition tolerance, smaller value = stricter (0.6 is common default)
MASTER_FACE_ENCODINGS = [] # Store multiple master face encodings for better recognition

# --- Main Program Logic ---
def main():
    global MASTER_FACE_ENCODING, MASTER_FACE_ENCODINGS, FACE_RECOGNITION_TOLERANCE

    cap = cv2.VideoCapture(0) # 0 represents default camera
    if not cap.isOpened():
        print("Error: Cannot open camera. Please check device connection or permissions.")
        return

    print("\n--- Face Recognition Module Test ---")
    print("1. First appearance, please face the camera and press 's' to record your face as 'Master'.")
    print("2. Recommended: Record from multiple angles - front, side, different expressions, press 's' multiple times.")
    print("3. After recording, you can leave and return to see if you can be recognized as 'Master'.")
    print("4. Press 't' to adjust tolerance, 'r' to reset records, 'q' to quit.")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Cannot read frame, exiting.")
            break

        # Convert image from BGR to RGB (face_recognition library requires RGB format)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Detect all faces in current frame
        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        current_face_status = "Unknown" # Current frame recognition status

        if not MASTER_FACE_ENCODINGS:
            # --- When no master face information is recorded ---
            cv2.putText(frame, "Status: Press 's' to set Master Face", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(frame, f"Recorded faces: {len(MASTER_FACE_ENCODINGS)}", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 1, cv2.LINE_AA)
            if face_locations:
                # Draw bounding boxes for detected faces (green)
                for (top, right, bottom, left) in face_locations:
                    cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                    cv2.putText(frame, "Found Face (Press 's')", (left, top - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)
        else:
            # --- When master face information is recorded, attempt re-identification ---
            found_master_in_frame = False
            best_match_distance = float('inf')
            
            for i, ((top, right, bottom, left), face_encoding) in enumerate(zip(face_locations, face_encodings)):
                # Compare with all recorded master face encodings
                matches = []
                distances = []
                
                for master_encoding in MASTER_FACE_ENCODINGS:
                    # Calculate face distance
                    distance = face_recognition.face_distance([master_encoding], face_encoding)[0]
                    distances.append(distance)
                    matches.append(distance <= FACE_RECOGNITION_TOLERANCE)
                
                # Find best match
                min_distance = min(distances)
                best_match = any(matches)
                
                name = "Stranger"
                color = (0, 0, 255) # Red for Stranger
                confidence = f"Distance: {min_distance:.3f}"
                
                if best_match:
                    name = "Master!"
                    color = (0, 255, 0) # Green for Master
                    found_master_in_frame = True
                    best_match_distance = min(best_match_distance, min_distance)
                
                # Draw face bounding box and label
                cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
                cv2.putText(frame, name, (left, top - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)
                cv2.putText(frame, confidence, (left, bottom + 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
            
            if found_master_in_frame:
                current_face_status = f"Master Recognized (Distance: {best_match_distance:.3f})"
            else:
                current_face_status = "Master not in view or not recognized"
            
            cv2.putText(frame, f"Status: {current_face_status}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(frame, f"Tolerance: {FACE_RECOGNITION_TOLERANCE}", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(frame, f"Recorded faces: {len(MASTER_FACE_ENCODINGS)}", (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1, cv2.LINE_AA)

        cv2.imshow('Face Recognition Test', frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('s'): # Press 's' to record master face information
            if face_encodings:
                # Record all faces detected in current frame
                for i, face_encoding in enumerate(face_encodings):
                    MASTER_FACE_ENCODINGS.append(face_encoding)
                    print(f"--- Recorded {len(MASTER_FACE_ENCODINGS)}th master face information ---")
                    print(f"Face encoding: {face_encoding[:5]}...") # Print first few digits of encoding
                print(f"Currently recorded {len(MASTER_FACE_ENCODINGS)} face information")
            else:
                print("No face detected, cannot record face information. Please ensure you are facing the camera.")
        elif key == ord('r'): # Press 'r' to reset records
            MASTER_FACE_ENCODINGS = []
            print("--- All recorded face information has been reset ---")
        elif key == ord('t'): # Press 't' to adjust tolerance
            if FACE_RECOGNITION_TOLERANCE == 0.6:
                FACE_RECOGNITION_TOLERANCE = 0.7
            elif FACE_RECOGNITION_TOLERANCE == 0.7:
                FACE_RECOGNITION_TOLERANCE = 0.8
            else:
                FACE_RECOGNITION_TOLERANCE = 0.6
            print(f"--- Tolerance adjusted to: {FACE_RECOGNITION_TOLERANCE} ---")
        elif key == ord('q'): # Press 'q' to quit
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()