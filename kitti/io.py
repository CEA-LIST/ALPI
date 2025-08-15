import os
import pickle
import numpy as np
from typing import List, Tuple

def read_data_indices(filepath: str) -> List[int]:
    """Reads a list of data indices from a text file (e.g., train.txt)."""
    with open(filepath, 'r') as f:
        return [int(line.strip()) for line in f if line.strip()]

def read_detection_file(filepath: str) -> Tuple[List[int], List[str], List[np.ndarray], List[float]]:
    """
    Parses a KITTI-style detection text file.
    Format: img_path typeid confidence xmin ymin xmax ymax
    """
    det_id2str = {1: 'Pedestrian', 2: 'Car', 3: 'Cyclist'}
    id_list, type_list, box2d_list, prob_list = [], [], [], []
    
    with open(filepath, 'r') as f:
        for line in f:
            parts = line.strip().split(" ")
            img_name = os.path.basename(parts[0]).rstrip('.png')
            id_list.append(int(img_name))
            
            try:
                cls_type = det_id2str[int(parts[1])]
            except (ValueError, KeyError):
                cls_type = parts[1] # Handle string type names
            type_list.append(cls_type)
            
            prob_list.append(float(parts[2]))
            box2d_list.append(np.array([float(v) for v in parts[3:7]]))
            
    return id_list, type_list, box2d_list, prob_list

def save_frustum_data(output_path: str, data: dict):
    """Saves the processed frustum data to a pickle file."""
    print(f"Saving processed data to {output_path}...")
    with open(output_path, 'wb') as fp:
        pickle.dump(data, fp)
    print(f"Save complete. Total objects: {len(data['id_list'])}")

