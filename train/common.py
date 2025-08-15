# utils/common.py

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import argparse
import random as pyrandom
import logging
import subprocess
import numpy as np
import torch

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(os.path.dirname(BASE_DIR))
sys.path.append(ROOT_DIR)

from configs.config import cfg
from datasets.provider_sample import compute_alpha

def parse_args():
    """Universal argument parser."""
    parser = argparse.ArgumentParser(description='PointNet Detection Framework')
    parser.add_argument('--cfg', dest='cfg_file', help='Config file', default=None, type=str)
    parser.add_argument('--step', help='Step for training', default=100, type=int)
    parser.add_argument('--ratio', dest='ratio', help='Ratio of fake/inserted objects', default=0, type=int)
    parser.add_argument('--weak', action='store_true', help='Train under weak supervision')
    parser.add_argument('--car_only', action='store_true', help='Use only car data')
    parser.add_argument('--people_only', action='store_true', help='Use only pedestrian data')
    parser.add_argument('opts', help='Modify config options using the command-line', default=None, nargs=argparse.REMAINDER)
    
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
        
    return parser.parse_args()

def set_random_seed(seed=3):
    """Sets the random seed for reproducibility."""
    pyrandom.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def fill_files(output_dir, to_fill_filename_list):
    """Create empty files if they don't exist."""
    for filename in to_fill_filename_list:
        filepath = os.path.join(output_dir, filename)
        if not os.path.exists(filepath):
            with open(filepath, 'w') as f:
                pass

def write_detection_results(output_dir, det_results):
    """Writes detection results to files."""
    results = {}
    for idx in det_results:
        for class_type, dets in det_results[idx].items():
            for i in range(len(dets)):
                box2d = dets[i][:4]
                tx, ty, tz, h, w, l, ry = dets[i][4:-1]
                score = dets[i][-1]
                alpha = compute_alpha(tx, tz, ry)

                output_str = f"{class_type} -1 -1 {alpha:.4f} "
                output_str += f"{box2d[0]:.4f} {box2d[1]:.4f} {box2d[2]:.4f} {box2d[3]:.4f} "
                output_str += f"{h:.4f} {w:.4f} {l:.4f} {tx:.4f} {ty:.4f} {tz:.4f} {ry:.4f} {score}"
                
                results.setdefault(idx, []).append(output_str)

    result_dir = os.path.join(output_dir, 'data')
    os.makedirs(result_dir, exist_ok=True)

    for idx, lines in results.items():
        pred_filename = os.path.join(result_dir, f'{idx:06d}.txt')
        with open(pred_filename, 'w') as f:
            for line in lines:
                f.write(line + '\n')

    idx_path = f'kitti/image_sets/{cfg.TEST.DATASET}.txt'
    to_fill_filename_list = [line.rstrip() + '.txt' for line in open(idx_path)]
    fill_files(result_dir, to_fill_filename_list)

def evaluate_py_wrapper(output_dir, async_eval=False):
    """Official KITTI evaluation using Python."""
    gt_dir = 'data/kitti/training/label_2/'
    command = (f'./train/kitti_eval/evaluate_object_3d_offline {gt_dir} {output_dir} '
               f'2>&1 | tee -a {os.path.join(output_dir, "log_test.txt")}')
    if async_eval:
        subprocess.Popen(command, shell=True)
    elif os.system(command) != 0:
        raise AssertionError("Evaluation failed.")

def evaluate_cuda_wrapper(result_dir, image_set='val', async_eval=False):
    """KITTI evaluation using the CUDA-based script."""
    if cfg.DATA.CAR_ONLY:
        classes_idx = '0'
    elif cfg.DATA.PEOPLE_ONLY:
        classes_idx = '1,2'
    else:
        classes_idx = '0,1,2'

    gt_dir = 'data/kitti/training/label_2/'
    label_split_file = f'./data/kitti/{image_set}.txt'
    command = (f"CUDA_VISIBLE_DEVICES=0 python ../kitti-object-eval-python/evaluate.py evaluate "
               f"--label_path={gt_dir} --result_path={result_dir} --label_split_file={label_split_file} "
               f"--current_class={classes_idx} --coco=False 2>&1 | tee -a "
               f"{os.path.join(result_dir, '..', 'log_test_new.txt')}")
    if async_eval:
        subprocess.Popen(command, shell=True)
    elif os.system(command) != 0:
        raise AssertionError("CUDA-based evaluation failed.")