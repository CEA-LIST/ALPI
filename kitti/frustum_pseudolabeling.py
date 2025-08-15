import os
import sys
import argparse
import numpy as np
import pickle
import random
from tqdm import tqdm

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(ROOT_DIR)

from kitti_object import kitti_object
from utils import geometry, augmentation, io
from priors import ObjectPriors

from data_processing_common import SceneData, append_data, estimate_augmented_object_center, initialize_data_dict


def _get_instance_dictionary_path(save_dir, class_name, step, ratio):
    """Determines the path to the instance dictionary based on the current step and class."""
    if step >= 1:
        return os.path.join(save_dir, f'instance_dictionary_{class_name}_step_{step}_ratio_{int(ratio/10)}.pkl')
    return os.path.join(save_dir, f'instance_dictionary_{class_name}_step_{step}.pkl')

def _select_instance_to_insert(dico_instances, querry_angle, n=10):
    """Selects a random instance from the top N closest instances based on heading angle."""
    angle_list = np.array(list(dico_instances.keys()))
    indexed_list = list(enumerate(np.abs(angle_list - querry_angle)))
    sorted_list = sorted(indexed_list, key=lambda x: x[1])
    top_n_indices = [index for index, _ in sorted_list[:n]]
    random_key_index = random.choice(top_n_indices)
    return angle_list[random_key_index]

def load_instance_dictionaries(type_whitelist, save_dir, step, ratio):
    """Loads instance dictionaries for each specified class."""
    instance_dictionaries = {}
    for class_name in type_whitelist:
        dict_path = _get_instance_dictionary_path(save_dir, class_name, step, ratio)
        print(f"Loading instance dictionary for '{class_name}' from: {dict_path}")
        try:
            with open(dict_path, 'rb') as fp:
                instance_dictionaries[class_name] = pickle.load(fp)
        except FileNotFoundError:
            print(f"  -> WARNING: Instance dictionary not found for '{class_name}'.")
            instance_dictionaries[class_name] = None
    return instance_dictionaries

# --- Main Data Extraction Logic ---

def extract_frustum_data(idx_filename, split, output_filename,
                         perturb_box2d=False, augmentX=1, type_whitelist=['Car'], 
                         ratio=100, step=0, save_dir='.', impostors=False):
    """
    Main pipeline to extract frustum data and augment it with fake and pseudo-labeled objects.
    """
    data_idx_list = io.read_data_indices(idx_filename)
    dataset = kitti_object(os.path.join(ROOT_DIR, 'data/kitti'), split)
    
    instance_dictionaries = load_instance_dictionaries(type_whitelist, save_dir, step, ratio)
    used_instances = {cls: {key: 0 for key in (dico or {})} for cls, dico in instance_dictionaries.items()}

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
                
                if len(pc_in_frustum) < 5: continue
                
                _, inds = geometry.extract_pc_in_box3d(pc_in_frustum, obj.box3d_pts_3d.T)
                labels = np.zeros(len(pc_in_frustum), dtype=np.int32)
                labels[inds] = 1
                
                if np.sum(labels) == 0: continue

                append_data(data_output, id=data_idx, box2d=box2d, box3d=obj.box3d_pts_3d.T,
                            input=pc_in_frustum.astype(np.float32), label=labels, type=obj.type,
                            heading=obj.ry, box3d_size=np.array([obj.l, obj.w, obj.h]),
                            frustum_angle=frustum_angle, calib=scene.calib.calib_dict)

                # --- 2. Augment with Fake or Pseudo Objects ---
                if not impostors: continue

                aug_center = estimate_augmented_object_center(scene.calib, obj, pc_in_frustum, obj.type)
                if aug_center is None: continue

                should_insert_pseudo = random.uniform(0, 1) >= (ratio / 100.0)
                dico_inst = instance_dictionaries.get(obj.type)

                if should_insert_pseudo and dico_inst:
                    # --- 2a. Insert Pseudo-labeled Object ---
                    key_to_insert = _select_instance_to_insert(dico_inst, np.arctan2(aug_center[0], aug_center[2]))
                    instance_obj = dico_inst[key_to_insert]['obj']
                    instance_pts = dico_inst[key_to_insert]['pts']
                    
                    aug_size = np.array([instance_obj.l, instance_obj.w, instance_obj.h])
                    aug_heading = instance_obj.ry % np.pi
                    
                    background_mask = np.ones(len(pc_in_frustum), dtype=bool)
                    background_mask[inds] = False
                    background_points = pc_in_frustum[background_mask, :3]
                    
                    transformed_instance_pts = instance_pts[:, :3] - instance_obj.t + aug_center
                    aug_points = np.vstack((background_points, transformed_instance_pts))
                    
                    used_instances[obj.type][key_to_insert] += 1
                else:
                    # --- 2b. Generate Fake Object ---
                    aug_size = ObjectPriors.generate_proxy_size(obj.type)
                    aug_heading = (random.randint(0, 11) / 12.0) * np.pi
                    
                    num_fake_points = random.randint(50, 150)
                    aug_points = np.random.rand(num_fake_points, 3) - 0.5
                    aug_points *= aug_size
                    aug_points = (geometry.roty(aug_heading) @ aug_points.T).T
                    aug_points += aug_center
                    
                # --- Common logic for both augmented types ---
                aug_box3d_3d = geometry.in_camera_coordinate(aug_center, aug_size, aug_heading).T
                aug_pts_2d = geometry.project_to_image(aug_box3d_3d.T, scene.calib.P)
                aug_box2d = np.array([np.min(aug_pts_2d[0]), np.min(aug_pts_2d[1]), 
                                      np.max(aug_pts_2d[0]), np.max(aug_pts_2d[1])])
                
                _, aug_inds = geometry.extract_pc_in_box3d(aug_points, aug_box3d_3d)
                aug_labels = np.zeros(len(aug_points), dtype=np.int32)
                aug_labels[aug_inds] = 1
                
                append_data(data_output, id=data_idx, box2d=aug_box2d, box3d=aug_box3d_3d,
                            input=aug_points.astype(np.float32), label=aug_labels, type=f'Impostor_{obj.type}',
                            heading=aug_heading, box3d_size=aug_size, frustum_angle=frustum_angle, 
                            calib=scene.calib.calib_dict)

    io.save_frustum_data(output_filename, data_output)
    
    # Print usage statistics for pseudo-labeled instances
    for cls, usage_dict in used_instances.items():
        if usage_dict:
            occ = np.array(list(usage_dict.values()))
            print(f"\n--- Instance Usage Statistics for '{cls}' ---")
            print(f"  Mean: {np.mean(occ):.2f}, Median: {np.median(occ)}, Max: {np.max(occ)}")
            print(f"  Instances used: {np.sum(occ > 0)} / {len(occ)}")

def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(description="Unified KITTI Frustum Data Preparation Script for Pseudo-Labeling.")
    
    parser.add_argument('--gen_train', action='store_true', help='Generate augmented training data from ground truth.')
    parser.add_argument('--gen_val', action='store_true', help='Generate clean validation data from ground truth.')
    
    parser.add_argument('--type_whitelist', nargs='+', default=['Car'], 
                        help="List of class names to process (e.g., Car Pedestrian Cyclist).")
    parser.add_argument('--save_dir', default=os.path.join(ROOT_DIR, 'data/pickle_data'), 
                        type=str, help='Directory to save data and load instance dictionaries from.')
    parser.add_argument('--step', default=0, type=int, help='Current step of the pseudo-labeling loop.')
    parser.add_argument('--ratio', default=100, type=int, 
                        help='Ratio of fake to pseudo-labeled objects (0-100). 100 means all fake, 0 means all pseudo.')

    args = parser.parse_args()
    np.random.seed(args.step)

    output_prefix = f'frustum_{"_".join(args.type_whitelist).lower()}_'
    os.makedirs(args.save_dir, exist_ok=True)

    if args.gen_train:
        print(f"--- Generating Training Data for: {args.type_whitelist} ---")
        filename = f'{output_prefix}train_step_0{args.step}_ratio_0{int(args.ratio/10)}.pickle'
        output_filename = os.path.join(args.save_dir, filename)
        extract_frustum_data(
            idx_filename=os.path.join(ROOT_DIR, 'image_sets/train.txt'),
            split='training', output_filename=output_filename,
            perturb_box2d=True, augmentX=1, type_whitelist=args.type_whitelist,
            ratio=args.ratio, step=args.step, save_dir=args.save_dir, impostors=True
        )

    if args.gen_val:
        print(f"--- Generating Validation Data for: {args.type_whitelist} ---")
        output_filename = os.path.join(args.save_dir, f'{output_prefix}val.pickle')
        extract_frustum_data(
            idx_filename=os.path.join(ROOT_DIR, 'image_sets/val.txt'),
            split='training', output_filename=output_filename,
            perturb_box2d=False, augmentX=1, type_whitelist=args.type_whitelist,
            save_dir=args.save_dir, impostors=False
        )

if __name__ == '__main__':
    main()
