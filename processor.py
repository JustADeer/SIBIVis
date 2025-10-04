import numpy as np

def calc_rotation_angle(A: np.ndarray, B: np.ndarray, C: np.ndarray) -> float:
    AB = A - B
    BC = C - B

    dot = np.dot(AB, BC)
    cross = np.cross(AB, BC)

    angle = np.arctan2(cross, dot)
    return angle

def calc_finger_rot(row) -> np.ndarray:
    finger_data = np.array(row).reshape(5, 4, 3) # 5 Fingers, 4 Points, 3 x,y,z
    finger_rot = np.empty(5, dtype=np.float32)

    for i, landmark_points in enumerate(finger_data):
        # Slicing creates 2D vectors (x, y)
        p0 = landmark_points[0][0:2]
        p1 = landmark_points[1][0:2]
        p2 = landmark_points[2][0:2]
        p3 = landmark_points[3][0:2]
        
        finger_rot[i] = calc_rotation_angle(p0, (p1 + p2)/2, p3)

    return finger_rot