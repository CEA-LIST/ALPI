import os
import argparse
import pickle
import numpy as np
from tqdm import tqdm
from typing import List, Dict, Any, Tuple


import kitti_util as utils
from kitti_object import kitti_object
from geometry import extract_pc_in_box3d, angle_between

ROOT_DIR = '/data/slahlali/kitti'
KITTI_DATA_DIR = os.path.join(ROOT_DIR, 'data/kitti')
IMAGE_SETS_DIR = os.path.join('./kitti', 'image_sets')
OUTPUT_PICKLE_DIR = os.path.join('./kitti', 'data/pickle_data')

IMG_HEIGHT = 375
IMG_WIDTH = 1242


def calculate_iou(box1: np.ndarray, box2: np.ndarray) -> float:
    """
    Calculates the Intersection over Union (IoU) of two 2D bounding boxes.

    Args:
        box1 (np.ndarray): A numpy array of shape (4,) with coordinates [x_left, y_top, x_right, y_bottom].
        box2 (np.ndarray): A numpy array of shape (4,) with coordinates [x_left, y_top, x_right, y_bottom].

    Returns:
        float: The IoU score, a value between 0.0 and 1.0.
    """
    # Get coordinates of the intersection rectangle
    x_left = max(box1[0], box2[0])
    y_top = max(box1[1], box2[1])
    x_right = min(box1[2], box2[2])
    y_bottom = min(box1[3], box2[3])

    # The '+ 1' is for inclusive pixel coordinates (e.g., a 1x1 box from [0,0] to [0,0])
    intersection_area = max(0, x_right - x_left + 1) * max(0, y_bottom - y_top + 1)

    # Calculate the area of both bounding boxes
    box1_area = (box1[2] - box1[0] + 1) * (box1[3] - box1[1] + 1)
    box2_area = (box2[2] - box2[0] + 1) * (box2[3] - box2[1] + 1)

    # Calculate IoU
    union_area = float(box1_area + box2_area - intersection_area)

    # Avoid division by zero
    return intersection_area / union_area if union_area > 0 else 0.0

def gen_instances(step: int, ratio: int):
    """
    Generates, filters, and saves object instances from the KITTI dataset based on pseudo-labels.
    """
    print(f" Starting instance generation for step={step} and ratio={ratio}")

    # --- 1. Load Data ---
    pseudo_label_path = f'{ROOT_DIR}/fconv/output/pedestrian_step_0{step}/_best_train/val_nms/result'
    if not os.path.isdir(pseudo_label_path):
        print(f" Error: Pseudo-label directory not found at: {pseudo_label_path}")
        return

    print(f" Loading pseudo-labels from: {pseudo_label_path}")
    dataset_pseudo_labels = kitti_object(pseudo_label_path, pseudo=True)
    dataset = kitti_object(KITTI_DATA_DIR, 'training')

    idx_filename = os.path.join(IMAGE_SETS_DIR, 'train_gt.txt')
    try:
        data_idx_list = [int(line.rstrip()) for line in open(idx_filename)]
    except FileNotFoundError:
        print(f" Error: Index file not found: {idx_filename}")
        return

    # --- 2. Process Frames and Extract Instances ---
    all_instances: List[Dict[str, Any]] = []

    for data_idx in tqdm(data_idx_list, desc="Processing Frames"):
        calib = dataset.get_calibration(data_idx)
        objects = dataset_pseudo_labels.get_label_objects(data_idx)
        pc_velo = dataset.get_lidar(data_idx)
        pc_rect = np.zeros_like(pc_velo)
        pc_rect[:, 0:3] = calib.project_velo_to_rect(pc_velo[:, 0:3])
        pc_rect[:, 3] = pc_velo[:, 3]

        for obj in objects:
            box3d_pts_2d, box3d_pts_3d = utils.compute_box_3d(obj, calib.P)
            
            # Create predicted 2D box from 3D box projection
            box2d_pred = np.array([
                min(box3d_pts_2d[0]), min(box3d_pts_2d[1]),
                max(box3d_pts_2d[0]), max(box3d_pts_2d[1])
            ])
            box2d_pred[[0, 2]] = np.clip(box2d_pred[[0, 2]], 0, IMG_WIDTH - 1)
            box2d_pred[[1, 3]] = np.clip(box2d_pred[[1, 3]], 0, IMG_HEIGHT - 1)

            points, _ = extract_pc_in_box3d(pc_rect, box3d_pts_3d)
            
            angle = angle_between([1, 0], [obj.t[0], obj.t[2]])
            iou = calculate_iou(obj.box2d, box2d_pred)

            all_instances.append({
                'pts': points,
                'obj': obj,
                'iou': iou,
                'angle': angle
            })

    if not all_instances:
        print(" No instances were extracted. Exiting.")
        return

    # --- 3. Analyze and Filter Instances ---
    nb_pts_list = [inst['pts'].shape[0] for inst in all_instances]
    
    print("\n Instance Statistics (before filtering):")
    print(f"  - Total Instances: {len(all_instances)}")
    print(f"  - Points per Box (Median): {np.median(nb_pts_list):.2f}")
    print(f"  - Points per Box (75% Quantile): {np.quantile(nb_pts_list, 0.75):.2f}")
    
    # We assume if step is 0, all instances should be kept.
    if step > 0:
        print("\n Filtering instances...")
        filtered_instances = [
            inst for inst in all_instances
            if inst['obj'].ry >= 0 and 10 < inst['obj'].t[2] < 35
        ]
        percentage_left = (len(filtered_instances) / len(all_instances)) * 100
        print(f"  - Kept {len(filtered_instances)} / {len(all_instances)} ({percentage_left:.2f}%)")
    else:
        filtered_instances = all_instances
        print("\n No filtering applied for step 0.")

    # --- 4. Save Results ---
    os.makedirs(OUTPUT_PICKLE_DIR, exist_ok=True)
    
    if step > 0:
        filename = f'instance_dictionary_step_0{step}_ratio_0{int(ratio/10)}.pkl'
    else:
        filename = f'instance_dictionary_step_0{step}.pkl'
    
    output_path = os.path.join(OUTPUT_PICKLE_DIR, filename)

    with open(output_path, 'wb') as fp:
        pickle.dump(filtered_instances, fp)

    print(f"\n Successfully saved {len(filtered_instances)} instances to:\n   {output_path}\n")


def main():
    """Main function to parse arguments and run the instance generation."""
    parser = argparse.ArgumentParser(
        description="Generate and filter object instances from KITTI dataset.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--step', type=int, default=0,
                        help='Current step of the pseudo-labeling loop.')
    parser.add_argument('--ratio', type=int, default=100,
                        help='Ratio of fake/inserted objects.')
    parser.add_argument('--car_only', action='store_true', help='(Currently Unused) Only generate cars.')
    parser.add_argument('--people_only', action='store_true', help='(Currently Unused) Only generate pedestrians.')

    args = parser.parse_args()

    if args.car_only or args.people_only:
        print("Warning: --car_only and --people_only flags are defined but not used in the current logic.")

    gen_instances(step=args.step, ratio=args.ratio)


if __name__ == '__main__':
    main()