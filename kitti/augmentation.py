import numpy as np

def random_shift_box2d(box2d: np.ndarray, img_height: int, img_width: int, shift_ratio: float = 0.1) -> np.ndarray:
    """
    Randomly shifts and scales a 2D bounding box for data augmentation.
    """
    r = shift_ratio
    xmin, ymin, xmax, ymax = box2d
    h, w = ymax - ymin, xmax - xmin
    
    # Ensure box has a valid area
    if h <= 0 or w <= 0:
        return box2d

    cx, cy = (xmin + xmax) / 2.0, (ymin + ymax) / 2.0
    
    # Try up to 10 times to generate a valid new box
    for _ in range(10):
        cx2 = cx + w * r * (np.random.random() * 2 - 1)
        cy2 = cy + h * r * (np.random.random() * 2 - 1)
        h2 = h * (1 + np.random.random() * 2 * r - r)
        w2 = w * (1 + np.random.random() * 2 * r - r)
        
        new_box2d = np.array([cx2 - w2 / 2.0, cy2 - h2 / 2.0, cx2 + w2 / 2.0, cy2 + h2 / 2.0])
        
        new_box2d[[0, 2]] = np.clip(new_box2d[[0, 2]], 0, img_width - 1)
        new_box2d[[1, 3]] = np.clip(new_box2d[[1, 3]], 0, img_height - 1)
        
        if new_box2d[0] < new_box2d[2] and new_box2d[1] < new_box2d[3]:
            return new_box2d
            
    # If failed, return original box
    return box2d
