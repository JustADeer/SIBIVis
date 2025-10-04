# Data Analysis
import warnings
import numpy as np

#From PIP
import cv2
from mediapipe import solutions as mps
from time import sleep

import processor

from joblib import load as jlload

mps_drawing = mps.drawing_utils

def warn(*args, **kwargs) -> None:
    pass

def extract_hand_landmarks(results: object) -> np.ndarray:
    if not results.multi_hand_landmarks:
        return np.array([])
    
    landmarks = results.multi_hand_landmarks[0].landmark
    all_coords = np.array(
        [[lm.x, lm.y, lm.z] for lm in landmarks],
        dtype=np.float32)
    relative_marks = all_coords - all_coords[0] # Wrist as origin

    return (relative_marks[1:]).flatten() # Remove wrist coords, return 60-dim array

def recognize_hand() -> None:
    capture = cv2.VideoCapture(0)
    if not capture.isOpened():
        print('Unable to open camera!')
        sleep(1)
        return None
    
    print('Camera Success!')

    hands = mps.hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.5)

    
    model = jlload('model/svc_model_20251004.pkl')
    scaler = jlload('model/svc_scaler_20251004.pkl')

    while True:
        ret_val, img = capture.read()
        if not ret_val:
            print('Camera broken!')
            sleep(1)
            break

        # Open Camera View
        img = cv2.flip(img, 1)
        p_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Recognition
        results = hands.process(p_img)
        if results.multi_hand_landmarks:
            landmarks = extract_hand_landmarks(results)
            rot = processor.calc_finger_rot(landmarks)
            landmarks = np.concatenate([landmarks, rot])
            if landmarks.shape == (65,):
                scaled_data = scaler.transform(landmarks.reshape(1, -1))
                prediction = model.predict(scaled_data)
                print(f"Predicted Sign: {str(prediction).upper()}", end='\r')
            for hand_landmarks in results.multi_hand_landmarks:
                mps_drawing.draw_landmarks(img, hand_landmarks, mps.hands.HAND_CONNECTIONS)
        
        # Show window
        cv2.imshow('SIBIVis', img)

        # Exit Route
        if cv2.waitKey(1) == 27: #ESC Key
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    warnings.warn = warn
    recognize_hand()