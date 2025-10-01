import numpy as np

def new_calculate_rotation(landmarks) -> float:
    nl = ([lm.x, lm.y] for lm in landmarks)
    print(nl)
    

if __name__ == "__main__":
    print(new_calculate_rotation([{'x': 0, 'y': 1, 'z': 0}, {'x': 2, 'y': 0, 'z': -5}, {'x': -1, 'y': -3, 'z': 5}])) 