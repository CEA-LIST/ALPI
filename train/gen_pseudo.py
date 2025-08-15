from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import time
import logging
import numpy as np
import torch

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(ROOT_DIR)

from configs.config import cfg, merge_cfg_from_file, merge_cfg_from_list, assert_and_infer_cfg
from utils.utils import AverageMeter, import_from_file, get_logger
from datasets.provider_sample import from_prediction_to_label_format, compute_alpha
from datasets.dataset_info import DATASET_INFO
from ops.pybind11.rbbox_iou import rotate_nms_3d_cc as cube_nms

from common import parse_args, set_random_seed, fill_files, evaluate_py_wrapper, evaluate_cuda_wrapper

def write_pseudo_results(output_dir, det_results):
    """
    Specialized result writer for pseudo-label generation.
    Writes results against the 'train_gt' set.
    """
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
    
    # Key difference: Use a hardcoded image set for pseudo-labeling
    idx_path = 'kitti/image_sets/train_gt.txt'
    to_fill_filename_list = [line.rstrip() + '.txt' for line in open(idx_path)]
    fill_files(result_dir, to_fill_filename_list)


def generate_labels(model, test_dataset, test_loader, result_dir=None):
    """Main pseudo-label generation loop."""
    model.eval()
    fw_time_meter = AverageMeter()
    det_results = {}
    
    for i, data_dicts in enumerate(test_loader):
        # This loop is identical to the test() function in test_net_det.py
        # For brevity, we assume the same logic. If there are subtle differences,
        # they would be implemented here.
        point_clouds = data_dicts['point_cloud']
        rot_angles = data_dicts['rot_angle']
        ref_centers = data_dicts.get('ref_center', torch.zeros((point_clouds.shape[0], 3)))
        rgb_probs = data_dicts.get('rgb_prob', torch.ones_like(rot_angles))

        batch_size = point_clouds.shape[0]
        data_dicts_var = {key: value.cuda() for key, value in data_dicts.items() if isinstance(value, torch.Tensor)}

        with torch.no_grad():
            outputs = model(data_dicts_var)
        
        cls_probs, center_preds, heading_preds, size_preds, _, _ = outputs
        cls_probs, center_preds, heading_preds, size_preds = [v.data.cpu().numpy() for v in [cls_probs, center_preds, heading_preds, size_preds]]

        for b in range(batch_size):
            fg_idx = np.array([np.argmax(cls_probs[b, :, 1])])
            single_scores = cls_probs[b, fg_idx, 1] + rgb_probs[b].numpy()
            data_idx = test_dataset.id_list[test_loader.batch_size * i + b]
            
            det_results.setdefault(data_idx, {})
            class_type = test_dataset.type_list[test_loader.batch_size * i + b]
            det_results[data_idx].setdefault(class_type, [])

            for n, idx in enumerate(fg_idx):
                h, w, l, tx, ty, tz, ry = from_prediction_to_label_format(
                    center_preds[b, idx], heading_preds[b, idx], size_preds[b, idx], 
                    rot_angles[b].numpy(), ref_centers[b].numpy())
                
                if h < 0.01 or w < 0.01 or l < 0.01: continue
                
                output = [*test_dataset.box2d_list[test_loader.batch_size * i + b], tx, ty, tz, h, w, l, ry, single_scores[n]]
                det_results[data_idx][class_type].append(output)

    # Use the specialized writer for pseudo-labels
    write_pseudo_results(result_dir, det_results)
    
    output_dir = os.path.join(result_dir, 'data')
    
    # Evaluate against the 'train_gt' set
    if os.path.exists('../kitti-object-eval-python'):
        evaluate_cuda_wrapper(output_dir, 'train_gt')
    else:
        evaluate_py_wrapper(result_dir)

if __name__ == '__main__':
    args = parse_args()
    set_random_seed()

    if args.cfg_file: merge_cfg_from_file(args.cfg_file)
    if args.opts: merge_cfg_from_list(args.opts)
    assert_and_infer_cfg()

    SAVE_DIR = os.path.join(cfg.OUTPUT_DIR, cfg.SAVE_SUB_DIR)
    os.makedirs(SAVE_DIR, exist_ok=True)
    
    log_file = f'{os.path.basename(args.cfg_file).split(".")[0]}_{time.strftime("%Y-%m-%d-%H-%M")}_pseudo.log'
    logger = get_logger(os.path.join(SAVE_DIR, log_file))

    model_path = "models/det_base_weak.py"
    model_def = import_from_file(model_path).PointNetDet

    dataset_def = import_from_file(cfg.DATA.FILE)
    
    # Key difference: Load the 'pseudo' split of the dataset
    test_dataset = dataset_def.ProviderDataset(
        cfg.DATA.NUM_SAMPLES, split='pseudo', one_hot=True, 
        random_flip=False, random_shift=False, extend_from_det=cfg.DATA.EXTEND_FROM_DET
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=cfg.TRAIN.BATCH_SIZE, shuffle=False,
        num_workers=cfg.NUM_WORKERS, pin_memory=True, drop_last=False, collate_fn=dataset_def.collate_fn
    )

    input_channels = cfg.DATA.EXTRA_FEAT_DIM if cfg.DATA.WITH_EXTRA_FEAT else 3
    datset_category_info = DATASET_INFO[cfg.DATA.DATASET_NAME]
    NUM_VEC = len(datset_category_info.CLASSES)
    
    model = model_def(input_channels, num_vec=NUM_VEC, num_classes=cfg.MODEL.NUM_CLASSES).cuda()

    if os.path.isfile(cfg.TEST.WEIGHTS):
        checkpoint = torch.load(cfg.TEST.WEIGHTS)
        model.load_state_dict(checkpoint.get('state_dict', checkpoint))
        logging.info(f"=> Loaded checkpoint '{cfg.TEST.WEIGHTS}'")
    else:
        raise FileNotFoundError(f"=> No checkpoint found at '{cfg.TEST.WEIGHTS}'")

    if cfg.NUM_GPUS > 1: model = torch.nn.DataParallel(model)
    
    result_folder = os.path.join(SAVE_DIR, 'result')
    os.makedirs(result_folder, exist_ok=True)
    
    generate_labels(model, test_dataset, test_loader, result_folder)