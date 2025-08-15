from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import time
import logging
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(ROOT_DIR)

from configs.config import cfg, merge_cfg_from_file, merge_cfg_from_list, assert_and_infer_cfg
from utils.training_states import TrainingStates
from utils.utils import AverageMeter, import_from_file, get_logger
from datasets.dataset_info import DATASET_INFO
from common import parse_args, set_random_seed
from test_net_det import test

def train(data_loader, model, optimizer, lr_scheduler, epoch, logger=None):
    model.train()
    training_states = TrainingStates()
    
    for i, data_dicts in enumerate(data_loader):
        batch_size = data_dicts['point_cloud'].shape[0]
        data_dicts_var = {key: value.cuda() for key, value in data_dicts.items() if isinstance(value, torch.Tensor)}
        
        optimizer.zero_grad()
        losses, metrics = model(data_dicts_var, epoch=epoch)
        loss = losses['total_loss'].mean()
        loss.backward()
        optimizer.step()

        losses_reduce = {k: v.detach().mean().item() for k, v in losses.items()}
        metrics_reduce = {k: v.detach().mean().item() for k, v in metrics.items()}
        training_states.update_states({**losses_reduce, **metrics_reduce}, batch_size)

        if (i + 1) % cfg.disp == 0 or (i + 1) == len(data_loader):
            states_str = training_states.format_states(training_states.get_states(avg=False))
            logging.info(f'Train Epoch: {epoch + 1:03d} [{i + 1:04d}/{len(data_loader)}] lr:{optimizer.param_groups[0]["lr"]:.6f} {states_str}')

    lr_scheduler.step()
    if logger:
        logger.add_scalars('train', training_states.get_states(avg=True), epoch)

def validate(data_loader, model, epoch, logger=None):
    model.eval()
    training_states = TrainingStates()
    
    with torch.no_grad():
        for data_dicts in data_loader:
            batch_size = data_dicts['point_cloud'].shape[0]
            data_dicts_var = {key: value.cuda() for key, value in data_dicts.items() if isinstance(value, torch.Tensor)}
            losses, metrics = model(data_dicts_var, epoch)
            losses_reduce = {k: v.detach().mean().item() for k, v in losses.items()}
            metrics_reduce = {k: v.detach().mean().item() for k, v in metrics.items()}
            training_states.update_states({**losses_reduce, **metrics_reduce}, batch_size)

    states = training_states.get_states(avg=True)
    logging.info(f'Validation Epoch: {epoch + 1:03d} {training_states.format_states(states)}')
    
    if logger:
        logger.add_scalars('val', states, epoch)
        
    return states['IoU_' + str(cfg.IOU_THRESH)]

def main():
    args = parse_args()
    if args.cfg_file: merge_cfg_from_file(args.cfg_file)
    if args.opts: merge_cfg_from_list(args.opts)
    assert_and_infer_cfg()

    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    log_file = os.path.join(cfg.OUTPUT_DIR, f'{os.path.basename(args.cfg_file).split(".")[0]}_{time.strftime("%Y-%m-%d-%H-%M")}_train.log')
    logger = get_logger(log_file)
    
    logger_train = SummaryWriter(os.path.join(cfg.OUTPUT_DIR, 'tb_logger', 'train'))
    logger_val = SummaryWriter(os.path.join(cfg.OUTPUT_DIR, 'tb_logger', 'val'))
    
    set_random_seed()
    
    dataset_def = import_from_file(cfg.DATA.FILE)
    collate_fn = dataset_def.collate_fn
    
    train_dataset = dataset_def.ProviderDataset(
        cfg.DATA.NUM_SAMPLES, split=cfg.TRAIN.DATASET, one_hot=True,
        random_flip=True, random_shift=True, extend_from_det=cfg.DATA.EXTEND_FROM_DET, 
        step=args.step, ratio=args.ratio)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=cfg.TRAIN.BATCH_SIZE, shuffle=True,
        num_workers=cfg.NUM_WORKERS, pin_memory=True, drop_last=True, collate_fn=collate_fn)

    val_dataset = dataset_def.ProviderDataset(
        cfg.DATA.NUM_SAMPLES, split=cfg.TEST.DATASET, one_hot=True,
        random_flip=False, random_shift=False, extend_from_det=cfg.DATA.EXTEND_FROM_DET, 
        step=args.step, ratio=args.ratio)
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=cfg.TEST.BATCH_SIZE, shuffle=False,
        num_workers=cfg.NUM_WORKERS, pin_memory=True, drop_last=False, collate_fn=collate_fn)
        
    model_path = "models/det_base_weak.py" if args.weak else "models/det_base_fully.py"
    model_def = import_from_file(model_path).PointNetDet
    
    input_channels = cfg.DATA.EXTRA_FEAT_DIM if cfg.DATA.WITH_EXTRA_FEAT else 3
    datset_category_info = DATASET_INFO[cfg.DATA.DATASET_NAME]
    NUM_VEC = len(datset_category_info.CLASSES)

    category = 0 if args.car_only else 1 if args.people_only else -1
    model = model_def(input_channels, num_vec=NUM_VEC, num_classes=cfg.MODEL.NUM_CLASSES, step=args.step, category=category).cuda()
    
    if cfg.NUM_GPUS > 1: model = torch.nn.DataParallel(model)

    optimizer = optim.Adam(model.parameters(), lr=cfg.TRAIN.BASE_LR, weight_decay=cfg.TRAIN.WEIGHT_DECAY)
    lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=cfg.TRAIN.LR_STEPS, gamma=cfg.TRAIN.GAMMA)
    
    start_epoch, best_prec1 = 0, 0
    
    for n in range(start_epoch, cfg.TRAIN.MAX_EPOCH):
        train(train_loader, model, optimizer, lr_scheduler, n, logger_train)
        prec1 = validate(val_loader, model, n, logger_val)
        
        if prec1 > best_prec1:
            best_prec1 = prec1
            logging.info(f'New best model with validation accuracy: {best_prec1:.6f}')
            torch.save({'epoch': n + 1, 'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict()},
                       os.path.join(cfg.OUTPUT_DIR, 'model_best.pth'))

if __name__ == '__main__':
    main()