import os
import sys
import argparse
import numpy as np
from tqdm import tqdm

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(ROOT_DIR)

from kitti_object import kitti_object
import geometry
import augmentation
import io

import random 

from data_processing_common import SceneData, append_data, generate_impostor, initialize_data_dict

def extract_frustum_data(idx_filename, split, output_filename,
                         perturb_box2d=False, augmentX=1, type_whitelist=['Car'], impostors=False):
    """
    Main pipeline to extract frustum data from the KITTI dataset based on ground truth.
    """
    data_idx_list = io.read_data_indices(idx_filename)
    dataset = kitti_object(os.path.join(ROOT_DIR, 'data/kitti'), split)
    
    data_keys = ['id', 'box2d', 'box3d', 'input', 'label', 'type', 'heading', 'box3d_size', 'frustum_angle', 'calib']
    data_output = initialize_data_dict(data_keys)

    for data_idx in tqdm(data_idx_list, desc=f"Processing GT {split}"):
        scene = SceneData(dataset, data_idx)
        objects = dataset.get_label_objects(data_idx)

        for obj in objects:
            if obj.type not in type_whitelist or (obj.box2d[3] - obj.box2d[1]) < 25:
                continue

            for _ in range(augmentX):
                # --- 1. Process Real Object ---
                box2d = augmentation.random_shift_box2d(obj.box2d, scene.img_shape[0], scene.img_shape[1]) if perturb_box2d else obj.box2d
                pc_in_frustum, frustum_angle = geometry.extract_pc_in_frustum(
                    scene.pc_rect, scene.pc_image_coord, scene.img_fov_inds, box2d, scene.calib
                )
                
                if len(pc_in_frustum) == 0: continue

                _, inds = geometry.extract_pc_in_box3d(pc_in_frustum, obj.box3d_pts_3d.T)
                labels = np.zeros(len(pc_in_frustum), dtype=np.int32)
                labels[inds] = 1
                
                if np.sum(labels) == 0: continue

                append_data(data_output, id=data_idx, box2d=box2d, box3d=obj.box3d_pts_3d.T,
                            input=pc_in_frustum.astype(np.float32), label=labels, type=obj.type,
                            heading=obj.ry, box3d_size=np.array([obj.l, obj.w, obj.h]),
                            frustum_angle=frustum_angle, calib=scene.calib.calib_dict)

                # --- 2. Generate Impostor for Augmentation ---
                if impostors:
                    impostor = generate_impostor(scene, obj, pc_in_frustum)
                    append_data(data_output, id=data_idx, box2d=impostor['box2d'], box3d=impostor['box3d'],
                                input=impostor['points'], label=impostor['label'], type=impostor['type'],
                                heading=impostor['heading'], box3d_size=impostor['size'],
                                frustum_angle=impostor['frustum_angle'], calib=scene.calib.calib_dict)
                            
    io.save_frustum_data(output_filename, data_output)

def extract_frustum_data_rgb_detection(det_filename, split, output_filename,
                                       type_whitelist=['Car'], img_height_threshold=25, lidar_point_threshold=5):
    """
    Extracts frustum data based on 2D detections from an external file.
    """
    dataset = kitti_object(os.path.join(ROOT_DIR, 'data/kitti'), split=split)
    det_id_list, det_type_list, det_box2d_list, det_prob_list = io.read_detection_file(det_filename)
    
    data_keys = ['id', 'box2d', 'input', 'type', 'frustum_angle', 'prob', 'calib']
    data_output = initialize_data_dict(data_keys)
    
    cached_scene = None
    last_data_idx = -1

    for i in tqdm(range(len(det_id_list)), desc=f"Processing RGB Detections {split}"):
        data_idx = det_id_list[i]
        
        # Cache scene data to avoid reloading for multiple detections in the same frame
        if last_data_idx != data_idx:
            cached_scene = SceneData(dataset, data_idx)
            last_data_idx = data_idx

        if det_type_list[i] not in type_whitelist: continue

        box2d = det_box2d_list[i]
        if (box2d[3] - box2d[1]) < img_height_threshold: continue
            
        pc_in_frustum, frustum_angle = geometry.extract_pc_in_frustum(
            cached_scene.pc_rect, cached_scene.pc_image_coord, cached_scene.img_fov_inds, box2d, cached_scene.calib
        )
        
        if len(pc_in_frustum) < lidar_point_threshold: continue

        append_data(data_output, id=data_idx, type=det_type_list[i], box2d=box2d,
                    prob=det_prob_list[i], input=pc_in_frustum.astype(np.float32),
                    frustum_angle=frustum_angle, calib=cached_scene.calib.calib_dict)

    io.save_frustum_data(output_filename, data_output)

def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(description="KITTI Frustum Data Extractor")
    parser.add_argument('--gen_train', action='store_true', help='Generate train split frustum data with perturbed GT 2D boxes')
    parser.add_argument('--gen_val', action='store_true', help='Generate val split frustum data with GT 2D boxes')
    parser.add_argument('--gen_val_rgb_detection', action='store_true', help='Generate val split frustum data with RGB detection 2D boxes')
    parser.add_argument('--gen_trainval', action='store_true', help='Generate trainval split frustum data with perturbed GT 2D boxes')
    parser.add_argument('--gen_test_rgb_detection', action='store_true', help='Generate test split frustum data with RGB detection 2D boxes')
    parser.add_argument('--car_only', action='store_true', help='Only generate cars')
    parser.add_argument('--people_only', action='store_true', help='Only generate peds and cycs')
    parser.add_argument('--save_dir', default=os.path.join(ROOT_DIR, 'data/pickle_data'), type=str, help='data directory to save data')
    args = parser.parse_args()

    np.random.seed(0)
    os.makedirs(args.save_dir, exist_ok=True)

    if args.car_only:
        type_whitelist = ['Car']
        output_prefix = 'frustum_caronly_'
    elif args.people_only:
        type_whitelist = ['Pedestrian', 'Cyclist']
        output_prefix = 'frustum_pedcyc_'
    else:
        type_whitelist = ['Car', 'Pedestrian', 'Cyclist']
        output_prefix = 'frustum_carpedcyc_'

    # --- Execute Data Generation Based on Arguments ---
    if args.gen_train:
        print("--- Generating Training Data ---")

        output_name = f'{output_prefix}train_step_00.pickle'

        extract_frustum_data(
            idx_filename=os.path.join(ROOT_DIR, 'image_sets/train.txt'),
            split='training',
            output_filename=os.path.join(args.save_dir, output_name),
            perturb_box2d=True, augmentX=5,
            type_whitelist=type_whitelist,
            impostors=True
        )

    if args.gen_trainval:
        print("--- Generating Trainval Data ---")
        extract_frustum_data(
            idx_filename=os.path.join(ROOT_DIR, 'image_sets/trainval.txt'),
            split='training',
            output_filename=os.path.join(args.save_dir, f'{output_prefix}trainval.pickle'),
            perturb_box2d=True, augmentX=5,
            type_whitelist=type_whitelist,
            impostors=True)

    if args.gen_val:
        print("--- Generating Validation Data (from GT) ---")
        extract_frustum_data(
            idx_filename=os.path.join(ROOT_DIR, 'image_sets/val.txt'),
            split='training',
            output_filename=os.path.join(args.save_dir, f'{output_prefix}val.pickle'),
            perturb_box2d=False, augmentX=1,
            type_whitelist=type_whitelist,
            impostors=False)

    if args.gen_val_rgb_detection:
        print("--- Generating Validation Data (from RGB Detections) ---")
        extract_frustum_data_rgb_detection(
            det_filename=os.path.join(ROOT_DIR, 'rgb_detections/rgb_detection_val.txt'),
            split='training',
            output_filename=os.path.join(args.save_dir, f'{output_prefix}val_rgb_detection.pickle'),
            type_whitelist=type_whitelist)

    if args.gen_test_rgb_detection:
        print("--- Generating Test Data (from RGB Detections) ---")
        extract_frustum_data_rgb_detection(
            det_filename=os.path.join(ROOT_DIR, 'rgb_detections/rgb_detection_test.txt'),
            split='testing',
            output_filename=os.path.join(args.save_dir, f'{output_prefix}test_rgb_detection.pickle'),
            type_whitelist=type_whitelist)

if __name__ == '__main__':
    main()
