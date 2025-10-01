# Data Analysis
import pandas as pd
import numpy as np

#From PIP
import os
import cv2
from mediapipe import solutions as mps
from time import sleep

mps_drawing = mps.drawing_utils

def extract_hand_landmarks(results: object) -> np.ndarray:
    if not results.multi_hand_landmarks:
        return np.array([])
    
    landmarks = results.multi_hand_landmarks[0].landmark
    all_coords = np.array(
        [[lm.x, lm.y, lm.z] for lm in landmarks],
        dtype=np.float32)
    relative_marks = all_coords - all_coords[0] # Wrist as origin

    return relative_marks[1:].flatten() # Remove wrist coords, return 60-dim array

def collect_hand_sign_data(label: str) -> None:
    # Folder Checker
    label_dir = os.path.join('data')
    os.makedirs(label_dir, exist_ok=True)

    capture = cv2.VideoCapture(0)
    if not capture.isOpened():
        print('Unable to open camera!')
        sleep(1)
        return None
    
    print('Camera Success!')

    hands = mps.hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.7)
    hand_arr = np.empty((0, 60), dtype=np.float32) # 60-dim array for hand landmarks (X,Y,Z for 20 points, excluding wrist)

    while True:
        ret_val, img = capture.read()
        if not ret_val:
            print('Camera broken')
            sleep(1)
            break

        # Open Camera View
        img = cv2.flip(img, 1)

        # Recognition
        results = hands.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        if results.multi_hand_landmarks:
            landmarks = extract_hand_landmarks(results)
            if landmarks.shape == (60,):
                hand_arr = np.vstack([hand_arr, landmarks])
            for hand_landmarks in results.multi_hand_landmarks:
                mps_drawing.draw_landmarks(img, hand_landmarks, mps.hands.HAND_CONNECTIONS)
        
        # Show window
        cv2.imshow('SIBIVis', img)

        # Add if hand_arr row has exceeded 300
        print(hand_arr.shape[0])
        if hand_arr.shape[0] >= 300: # Fix this
            print(f'Collected 300 samples for label "{label}"')
            break

        # Exit Route
        if cv2.waitKey(1) == 27: #ESC Key
            break
        elif cv2.waitKey(25):
            if 0xFF == ord('r'):
                hand_arr = np.empty((0, 60), dtype=np.float32)
            elif 0xFF == ord('p'):
                print('Force quitting...')
                sleep(1)
                quit()
    
    df = pd.DataFrame(data=hand_arr)
    print(df.head(5))
    df.to_csv(f'{label_dir}/{label}_data_raw.csv', index=False)

    cv2.destroyAllWindows()

if __name__ == "__main__":
    while 1:
        if input('Ready to collect data? (y/n): ').lower() == 'n':
            break
        target = input('Label for target: ')
        collect_hand_sign_data(label=target)