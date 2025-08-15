import os
import sys
import argparse
import numpy as np
import pickle
from tqdm import tqdm


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(ROOT_DIR)


from kitti_object import kitti_object
import kitti_util as utils
import geometry
import augmentation
import io

from data_processing_common import SceneData, append_data, initialize_data_dict

def extract_refinement_data(idx_filename, split, output_filename,
                            perturb_box=False, augmentX=1, type_whitelist=['Car'], remove_diff=False):
    """
    Extracts point clouds within enlarged 3D boxes for refinement tasks.
    """
    dataset = kitti_object(os.path.join(ROOT_DIR, 'data/kitti'), split, pseudo=True)
    data_idx_list = io.read_data_indices(idx_filename)

    data_keys = [
        'id', 'box3d', 'input', 'label', 'type', 'heading', 'box3d_size',
        'frustum_angle', 'gt_box2d', 'calib', 'enlarge_box3d',
        'enlarge_box3d_size', 'enlarge_box3d_angle'
    ]
    data_output = initialize_data_dict(data_keys)

    pos_cnt = 0
    all_cnt = 0

    for data_idx in tqdm(data_idx_list, desc=f"Processing Refinement Data {split}"):
        scene = SceneData(dataset, data_idx)
        objects = dataset.get_label_objects(data_idx)

        for obj in objects:
            if obj.type not in type_whitelist:
                continue

            if remove_diff and (obj.occlusion > 2 or obj.truncation > 0.5 or (obj.box2d[3] - obj.box2d[1]) < 25):
                continue

            # Create a 7-element array representing the ground truth box
            # [cx, cy, cz, l, w, h, ry]
            gt_box_center_y = obj.t[1] - obj.h / 2
            gt_obj_array = np.array([obj.t[0], gt_box_center_y, obj.t[2], obj.l, obj.w, obj.h, obj.ry])
            gt_box3d_corners = geometry.compute_box_3d_corners_from_array(gt_obj_array)

            # Define the enlarged box for point cloud extraction
            enlarge_ratio = 1.2
            enlarge_obj_array = gt_obj_array.copy()
            enlarge_obj_array[3:6] *= enlarge_ratio

            for _ in range(augmentX):
                final_enlarge_array = enlarge_obj_array
                if perturb_box:
                    final_enlarge_array = augmentation.random_shift_rotate_box3d(enlarge_obj_array, 0.05)
                
                enlarge_box3d_corners = geometry.compute_box_3d_corners_from_array(final_enlarge_array)

                # Extract points within the enlarged box
                pc_in_enlarged_box, _ = geometry.extract_pc_in_box3d(scene.pc_rect, enlarge_box3d_corners)
                if len(pc_in_enlarged_box) == 0:
                    continue

                # Create labels: 1 for points inside the original GT box, 0 otherwise
                _, inds_in_gt_box = geometry.extract_pc_in_box3d(pc_in_enlarged_box, gt_box3d_corners)
                labels = np.zeros(len(pc_in_enlarged_box), dtype=np.int32)
                labels[inds_in_gt_box] = 1

                if np.sum(labels) == 0:
                    continue
                
                # The "frustum angle" is computed from the center of the enlarged box
                box3d_center = final_enlarge_array[:3]
                frustum_angle = -1 * np.arctan2(box3d_center[2], box3d_center[0])

                append_data(data_output,
                            id=data_idx,
                            box3d=gt_box3d_corners,
                            input=pc_in_enlarged_box.astype(np.float32),
                            label=labels,
                            type=obj.type,
                            heading=obj.ry,
                            box3d_size=np.array([obj.l, obj.w, obj.h]),
                            frustum_angle=frustum_angle,
                            gt_box2d=obj.box2d,
                            calib=scene.calib.calib_dict,
                            enlarge_box3d=enlarge_box3d_corners,
                            enlarge_box3d_size=final_enlarge_array[3:6],
                            enlarge_box3d_angle=final_enlarge_array[6])
                
                pos_cnt += np.sum(labels)
                all_cnt += len(pc_in_enlarged_box)

    print(f'Total objects processed: {len(data_output["id_list"])}')
    if all_cnt > 0:
        print(f'Average positive point ratio: {pos_cnt / float(all_cnt):.4f}')
        print(f'Average points per box: {float(all_cnt) / len(data_output["id_list"]):.2f}')

    io.save_pickle_data(output_filename, data_output)


def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(description="KITTI Data Preparation for Bounding Box Refinement")
    parser.add_argument('--gen_train', action='store_true', help='Generate augmented training data.')
    parser.add_argument('--gen_val', action='store_true', help='Generate clean validation data.')
    parser.add_argument('--car_only', action='store_true', help='Only generate for Car class.')
    parser.add_argument('--people_only', action='store_true', help='Only generate for Pedestrian and Cyclist classes.')
    parser.add_argument('--save_dir', type=str, default='data/pickle_data_refine', help='Directory to save data.')
    args = parser.parse_args()

    np.random.seed(0)
    
    save_dir = os.path.join(ROOT_DIR, args.save_dir)
    os.makedirs(save_dir, exist_ok=True)

    if args.car_only:
        type_whitelist = ['Car']
        output_prefix = 'refine_caronly_'
    elif args.people_only:
        type_whitelist = ['Pedestrian', 'Cyclist']
        output_prefix = 'refine_pedcyc_'
    else:
        type_whitelist = ['Car', 'Pedestrian', 'Cyclist']
        output_prefix = 'refine_carpedcyc_'

    if args.gen_train:
        print(f"--- Generating Training Data for Refinement: {type_whitelist} ---")
        output_filename = os.path.join(save_dir, f'{output_prefix}train.pickle')
        extract_refinement_data(
            idx_filename=os.path.join(ROOT_DIR, 'image_sets/train.txt'),
            split='training',
            output_filename=output_filename,
            perturb_box=True, augmentX=5,
            type_whitelist=type_whitelist
        )

    if args.gen_val:
        print(f"--- Generating Validation Data for Refinement: {type_whitelist} ---")
        output_filename = os.path.join(save_dir, f'{output_prefix}val.pickle')
        extract_refinement_data(
            idx_filename=os.path.join(ROOT_DIR, 'image_sets/val.txt'),
            split='training',
            output_filename=output_filename,
            perturb_box=False, augmentX=1,
            type_whitelist=type_whitelist,
            remove_diff=True
        )

if __name__ == '__main__':
    main()
