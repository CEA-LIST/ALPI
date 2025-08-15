from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from configs.config import cfg
from models.base_model import BasePointNetDet
from models.common import softmax_focal_loss_ignore, get_accuracy
from ops.pybind11.box_ops_cc import rbbox_iou_3d_pair
from models.box_transform import size_decode, size_encode, center_decode, center_encode, angle_decode, angle_encode
from models.model_util import get_box3d_corners_helper

class PointNetDet(BasePointNetDet):
    def __init__(self, input_channel=3, num_vec=0, num_classes=2, step=0, category=0):
        super(PointNetDet, self).__init__(input_channel, num_vec, num_classes)
        self.step = step
        self.category = category
        self.std_size_array = self.category_info.STD_SIZE_ARRAY

    def bb3d_to_2d(self, center_preds, heading, size, calib):
        corners_3d = torch.transpose(get_box3d_corners_helper(center_preds, heading, size), 1, 2)
        points3d = torch.cat((corners_3d, torch.ones((corners_3d.size(0), 1, corners_3d.size(2)), device=corners_3d.device)), 1)
        points = torch.bmm(calib, points3d)
        points_ = torch.ones_like(points).cuda()
        points_[:, :2, :] = points[:, :2, :] / points[:, 2, :].view(points.size(0), 1, -1)
        points_2d = points_[:, :2, :]
        bb = torch.transpose(torch.stack([torch.min(points_2d[:, 0], dim=1)[0], torch.min(points_2d[:, 1], dim=1)[0],
                                          torch.max(points_2d[:, 0], dim=1)[0], torch.max(points_2d[:, 1], dim=1)[0]]), 0, 1)
        return bb

    def forward(self, data_dicts, epoch=0):
        point_cloud = data_dicts.get('point_cloud')
        one_hot_vec = data_dicts.get('one_hot')
        
        # ## Create boolean masks to distinguish between real and impostor objects.
        # 'Impostors' are objects whose class index is >= num_classes. 
        # These are synthetic objects used for training the 3D box regression head.
        idxsImpostors = (torch.argmax(one_hot_vec, dim=1) >= self.num_classes)
        
        # 'Reals' are the actual objects in the scene (class index < num_classes).
        # The 3D box regression loss will NOT be applied to these.
        idxsReals = (torch.argmax(one_hot_vec, dim=1) < self.num_classes)

        cls_label = data_dicts.get('cls_label')
        size_class_label = data_dicts.get('size_class')
        center_label = data_dicts.get('box3d_center')
        heading_label = data_dicts.get('box3d_heading')
        size_label = data_dicts.get('box3d_size')
        calib = data_dicts.get('calib')
        bb2d_gt = data_dicts.get('bb_2d_gt')

        center_ref1, center_ref2, center_ref3, center_ref4 = (data_dicts.get('center_ref1'), data_dicts.get('center_ref2'), 
                                                              data_dicts.get('center_ref3'), data_dicts.get('center_ref4'))
        
        batch_size = point_cloud.shape[0]
        object_point_cloud_xyz = point_cloud[:, :3, :].contiguous()
        object_point_cloud_i = point_cloud[:, [3], :].contiguous() if point_cloud.shape[1] > 3 else None
        mean_size_array = torch.from_numpy(self.mean_size_array).type_as(point_cloud)

        feat1, feat2, feat3, feat4 = self.feat_net(object_point_cloud_xyz, [center_ref1, center_ref2, center_ref3, center_ref4], 
                                                   object_point_cloud_i, one_hot_vec)
        x = self.conv_net(feat1, feat2, feat3, feat4)
        cls_scores = self.cls_out(x)
        outputs = self.reg_out(x)

        num_out = outputs.shape[2]
        output_size = outputs.shape[1]
        cls_scores = cls_scores.permute(0, 2, 1).contiguous().view(-1, 2)
        outputs = outputs.permute(0, 2, 1).contiguous().view(-1, output_size)
        center_ref2 = center_ref2.permute(0, 2, 1).contiguous().view(-1, 3)
        cls_probs = F.softmax(cls_scores, -1)

        # --- Inference Path ---
        if not self.training:
            center_boxnet, heading_scores, heading_res_norm, size_scores, size_res_norm = self._slice_output(outputs)
            heading_probs = F.softmax(heading_scores, -1)
            size_probs = F.softmax(size_scores, -1)
            heading_pred_label = torch.argmax(heading_probs, -1)
            size_pred_label = torch.argmax(size_probs, -1)

            center_preds = center_decode(center_ref2, center_boxnet)
            heading_preds = angle_decode(heading_res_norm, heading_pred_label, num_bins=self.num_bins)
            size_preds = size_decode(size_res_norm, mean_size_array, self.category * torch.ones_like(size_pred_label).long().cuda())
            
            cls_probs = cls_probs.view(batch_size, -1, 2)
            center_preds = center_preds.view(batch_size, -1, 3)
            size_preds = size_preds.view(batch_size, -1, 3)
            size_probs = size_probs.view(batch_size, -1, self.num_size_cluster)
            heading_preds = heading_preds.view(batch_size, -1)
            heading_probs = heading_probs.view(batch_size, -1, self.num_bins)

            return cls_probs, center_preds, heading_preds, size_preds, heading_probs, size_probs

        # --- Training Path ---
        fg_idx = (cls_label.view(-1) == 1).nonzero().view(-1)
        assert fg_idx.numel() != 0, "No foreground points found."

        outputs = outputs[fg_idx, :]
        center_ref2 = center_ref2[fg_idx]
        
        center_boxnet, heading_scores, heading_res_norm, size_scores, size_res_norm = self._slice_output(outputs)

        idxsImpostors = idxsImpostors.unsqueeze(1).expand(-1, num_out).contiguous().view(-1)[fg_idx]
        idxsReals = idxsReals.unsqueeze(1).expand(-1, num_out).contiguous().view(-1)[fg_idx]

        # Expand ground truth labels to match the output dimensions for loss calculation
        center_label_exp = center_label.unsqueeze(1).expand(-1, num_out, -1).contiguous().view(-1, 3)[fg_idx]
        heading_label_exp = heading_label.expand(-1, num_out).contiguous().view(-1)[fg_idx]
        size_label_exp = size_label.unsqueeze(1).expand(-1, num_out, -1).contiguous().view(-1, 3)[fg_idx]
        size_class_label_exp = size_class_label.expand(-1, num_out).contiguous().view(-1)[fg_idx]
        calib_exp = calib.unsqueeze(1).expand(-1, num_out, -1, -1).contiguous().view(-1, 3, 4)[fg_idx]
        bb2d_gt_exp = bb2d_gt.unsqueeze(1).expand(-1, num_out, -1).contiguous().view(-1, 4)[fg_idx]

        # Encode ground truth values for regression
        center_gt_offsets = center_encode(center_label_exp, center_ref2)
        heading_class_label, heading_res_norm_label = angle_encode(heading_label_exp, num_bins=self.num_bins)
        size_res_label_norm = size_encode(size_label_exp, mean_size_array, size_class_label_exp)
        size_class_label_cat = self.category * torch.ones_like(size_class_label_exp).long().cuda()
        
        # Decode network predictions to get 3D box parameters
        center_preds = center_decode(center_ref2, center_boxnet)
        heading = angle_decode(heading_res_norm, heading_class_label, num_bins=self.num_bins)
        size = size_decode(size_res_norm, mean_size_array, size_class_label_cat)

        # ## Loss Calculation
        # Classification loss is calculated for all objects (both real and impostor).
        cls_loss = softmax_focal_loss_ignore(cls_probs, cls_label.view(-1), ignore_idx=-1)
        
        # --- 3D BOX REGRESSION LOSSES (APPLIED ONLY ON IMPOSTORS) ---
        # The `idxsImpostors` mask is used here to select only the predictions and ground truths
        # corresponding to impostor objects. This forces the model to learn 3D box regression
        # exclusively from these specific samples.
        center_loss = self.get_center_loss(center_boxnet[idxsImpostors], center_gt_offsets[idxsImpostors]).mean()
        heading_class_loss, heading_res_norm_loss = self.get_heading_loss(heading_scores[idxsImpostors], heading_res_norm[idxsImpostors], heading_class_label[idxsImpostors], heading_res_norm_label[idxsImpostors])
        size_class_loss, size_res_norm_loss = self.get_size_loss(size_scores[idxsImpostors], size_res_norm[idxsImpostors], size_class_label_cat[idxsImpostors], size_res_label_norm[idxsImpostors])
        corners_loss, _ = self.get_corner_loss((center_preds[idxsImpostors], heading[idxsImpostors], size[idxsImpostors]), (center_label_exp[idxsImpostors], heading_label_exp[idxsImpostors], size_label_exp[idxsImpostors]))
        
        # --- 2D Box Regression Loss ---
        bb2d_pred = self.bb3d_to_2d(center_preds, heading, size, calib_exp)
        norm = torch.tensor([[1/(b[2]-b[0]), 1/(b[3]-b[1]), 1/(b[2]-b[0]), 1/(b[3]-b[1])] for b in bb2d_gt_exp], device=self.device)
        smooth_l1_loss = F.smooth_l1_loss(bb2d_pred, bb2d_gt_exp, reduction='none')
        bb3d_2d_loss = (0.1) * (norm * smooth_l1_loss).mean()

        mean = torch.tensor(self.mean_size_array[self.category], device=self.device)
        std = torch.tensor(self.std_size_array[self.category], device=self.device)
        batch_std_size, batch_mean_size = torch.std_mean(size, dim=0)
        size_constr_loss = (F.smooth_l1_loss(batch_std_size, std, reduction='none') + F.smooth_l1_loss(batch_mean_size, mean, reduction='none')).mean()

        # Combine all losses
        size_constr_weight = 0.5 if self.step > 0 else 0
        loss = cls_loss + cfg.LOSS.BOX_LOSS_WEIGHT * (center_loss + heading_class_loss + size_class_loss +
                                                    cfg.LOSS.HEAD_REG_WEIGHT * heading_res_norm_loss +
                                                    cfg.LOSS.SIZE_REG_WEIGHT * size_res_norm_loss +
                                                    cfg.LOSS.CORNER_LOSS_WEIGHT * corners_loss) + bb3d_2d_loss + size_constr_weight*size_constr_loss

        # ## Metrics Calculation
        with torch.no_grad():
            heading_probs = F.softmax(heading_scores, -1)
            size_probs = F.softmax(size_scores, -1)
            cls_prec = get_accuracy(cls_probs, cls_label.view(-1), ignore=-1)
            heading_prec = get_accuracy(heading_probs, heading_class_label.view(-1))
            size_prec = get_accuracy(size_probs, size_class_label_cat.view(-1))
            
            heading_pred_label = torch.argmax(heading_probs, -1)
            heading_preds = angle_decode(heading_res_norm, heading_pred_label, num_bins=self.num_bins)
            size_preds = size_decode(size_res_norm, mean_size_array, size_class_label_cat)

            # ## Calculate IoU metrics only on the REAL objects
            # In contrast to the loss, the IoU (Intersection over Union) metric is calculated
            # only for the 'real' objects using the `idxsReals` mask. This provides a more
            # meaningful evaluation of the model's 3D detection performance on the actual task.
            corner_preds = get_box3d_corners_helper(center_preds[idxsReals], heading_preds[idxsReals], size_preds[idxsReals])
            _, corner_gts = self.get_corner_loss((center_preds[idxsReals], heading[idxsReals], size[idxsReals]), (center_label_exp[idxsReals], heading_label_exp[idxsReals], size_label_exp[idxsReals]))
            
            overlap = rbbox_iou_3d_pair(corner_preds.detach().cpu().numpy(), corner_gts.detach().cpu().numpy())
            iou2ds, iou3ds = overlap[:, 0], overlap[:, 1]
            iou2d_mean = torch.tensor(iou2ds.mean()).type_as(cls_prec)
            iou3d_mean = torch.tensor(iou3ds.mean()).type_as(cls_prec)
            iou3d_gt_mean = torch.tensor((iou3ds >= cfg.IOU_THRESH).mean()).type_as(cls_prec)
        
        losses = {'total_loss': loss, 'cls_loss': cls_loss, 'center_loss': center_loss, 'head_cls_loss': heading_class_loss, 'head_res_loss': heading_res_norm_loss,
                  'size_cls_loss': size_class_loss, 'size_res_loss': size_res_norm_loss, 'corners_loss': corners_loss, 'bb3d_2d_loss': bb3d_2d_loss, 'size_constr_loss': size_constr_loss}
        metrics = {'cls_acc': cls_prec, 'head_acc': heading_prec, 'size_acc': size_prec, 'IoU_2D': iou2d_mean, 'IoU_3D': iou3d_mean, 'IoU_' + str(cfg.IOU_THRESH): iou3d_gt_mean}
        
        return losses, metrics