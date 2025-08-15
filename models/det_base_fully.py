from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import torch
import torch.nn.functional as F

from configs.config import cfg
from models.base_model import BasePointNetDet
from models.common import softmax_focal_loss_ignore, get_accuracy
from ops.pybind11.box_ops_cc import rbbox_iou_3d_pair
from models.box_transform import size_decode, size_encode, center_decode, center_encode, angle_decode, angle_encode
from models.model_util import get_box3d_corners_helper

class PointNetDet(BasePointNetDet):
    def __init__(self, input_channel=3, num_vec=0, num_classes=2):
        super(PointNetDet, self).__init__(input_channel, num_vec, num_classes)

    def forward(self, data_dicts):
        point_cloud = data_dicts.get('point_cloud')
        one_hot_vec = data_dicts.get('one_hot')
        cls_label = data_dicts.get('cls_label')
        size_class_label = data_dicts.get('size_class')
        center_label = data_dicts.get('box3d_center')
        heading_label = data_dicts.get('box3d_heading')
        size_label = data_dicts.get('box3d_size')

        center_ref1 = data_dicts.get('center_ref1')
        center_ref2 = data_dicts.get('center_ref2')
        center_ref3 = data_dicts.get('center_ref3')
        center_ref4 = data_dicts.get('center_ref4')

        batch_size = point_cloud.shape[0]
        object_point_cloud_xyz = point_cloud[:, :3, :].contiguous()
        object_point_cloud_i = point_cloud[:, [3], :].contiguous() if point_cloud.shape[1] > 3 else None
        
        mean_size_array = torch.from_numpy(self.mean_size_array).type_as(point_cloud)

        feat1, feat2, feat3, feat4 = self.feat_net(
            object_point_cloud_xyz,
            [center_ref1, center_ref2, center_ref3, center_ref4],
            object_point_cloud_i,
            one_hot_vec)

        x = self.conv_net(feat1, feat2, feat3, feat4)
        cls_scores = self.cls_out(x)
        outputs = self.reg_out(x)

        num_out = outputs.shape[2]
        output_size = outputs.shape[1]
        cls_scores = cls_scores.permute(0, 2, 1).contiguous().view(-1, 2)
        outputs = outputs.permute(0, 2, 1).contiguous().view(-1, output_size)
        center_ref2 = center_ref2.permute(0, 2, 1).contiguous().view(-1, 3)
        cls_probs = F.softmax(cls_scores, -1)

        if not self.training:
            center_boxnet, heading_scores, heading_res_norm, size_scores, size_res_norm = self._slice_output(outputs)
            heading_probs = F.softmax(heading_scores, -1)
            size_probs = F.softmax(size_scores, -1)
            heading_pred_label = torch.argmax(heading_probs, -1)
            size_pred_label = torch.argmax(size_probs, -1)

            center_preds = center_decode(center_ref2, center_boxnet)
            heading_preds = angle_decode(heading_res_norm, heading_pred_label, num_bins=self.num_bins)
            size_preds = size_decode(size_res_norm, mean_size_array, size_pred_label)

            cls_probs = cls_probs.view(batch_size, -1, 2)
            center_preds = center_preds.view(batch_size, -1, 3)
            size_preds = size_preds.view(batch_size, -1, 3)
            size_probs = size_probs.view(batch_size, -1, self.num_size_cluster)
            heading_preds = heading_preds.view(batch_size, -1)
            heading_probs = heading_probs.view(batch_size, -1, self.num_bins)
            
            return cls_probs, center_preds, heading_preds, size_preds, heading_probs, size_probs

        fg_idx = (cls_label.view(-1) == 1).nonzero().view(-1)
        assert fg_idx.numel() != 0, "No foreground points found."
        
        outputs = outputs[fg_idx, :]
        center_ref2 = center_ref2[fg_idx]

        center_boxnet, heading_scores, heading_res_norm, size_scores, size_res_norm = self._slice_output(outputs)
        heading_probs = F.softmax(heading_scores, -1)
        size_probs = F.softmax(size_scores, -1)

        cls_loss = softmax_focal_loss_ignore(cls_probs, cls_label.view(-1), ignore_idx=-1)

        center_label = center_label.unsqueeze(1).expand(-1, num_out, -1).contiguous().view(-1, 3)[fg_idx]
        heading_label = heading_label.expand(-1, num_out).contiguous().view(-1)[fg_idx]
        size_label = size_label.unsqueeze(1).expand(-1, num_out, -1).contiguous().view(-1, 3)[fg_idx]
        size_class_label = size_class_label.expand(-1, num_out).contiguous().view(-1)[fg_idx]

        center_gt_offsets = center_encode(center_label, center_ref2)
        heading_class_label, heading_res_norm_label = angle_encode(heading_label, num_bins=self.num_bins)
        size_res_label_norm = size_encode(size_label, mean_size_array, size_class_label)

        center_loss = self.get_center_loss(center_boxnet, center_gt_offsets)
        heading_class_loss, heading_res_norm_loss = self.get_heading_loss(heading_scores, heading_res_norm, heading_class_label, heading_res_norm_label)
        size_class_loss, size_res_norm_loss = self.get_size_loss(size_scores, size_res_norm, size_class_label, size_res_label_norm)

        center_preds = center_decode(center_ref2, center_boxnet)
        heading = angle_decode(heading_res_norm, heading_class_label, num_bins=self.num_bins)
        size = size_decode(size_res_norm, mean_size_array, size_class_label)
        corners_loss, corner_gts = self.get_corner_loss((center_preds, heading, size), (center_label, heading_label, size_label))

        loss = cls_loss + cfg.LOSS.BOX_LOSS_WEIGHT * (center_loss + heading_class_loss + size_class_loss + 
                                                     cfg.LOSS.HEAD_REG_WEIGHT * heading_res_norm_loss + 
                                                     cfg.LOSS.SIZE_REG_WEIGHT * size_res_norm_loss + 
                                                     cfg.LOSS.CORNER_LOSS_WEIGHT * corners_loss)

        with torch.no_grad():
            cls_prec = get_accuracy(cls_probs, cls_label.view(-1), ignore=-1)
            heading_prec = get_accuracy(heading_probs, heading_class_label.view(-1))
            size_prec = get_accuracy(size_probs, size_class_label.view(-1))

            heading_pred_label = torch.argmax(heading_probs, -1)
            size_pred_label = torch.argmax(size_probs, -1)
            heading_preds = angle_decode(heading_res_norm, heading_pred_label, num_bins=self.num_bins)
            size_preds = size_decode(size_res_norm, mean_size_array, size_pred_label)
            corner_preds = get_box3d_corners_helper(center_preds, heading_preds, size_preds)
            
            overlap = rbbox_iou_3d_pair(corner_preds.detach().cpu().numpy(), corner_gts.detach().cpu().numpy())
            iou2ds, iou3ds = overlap[:, 0], overlap[:, 1]
            iou2d_mean = torch.tensor(iou2ds.mean()).type_as(cls_prec)
            iou3d_mean = torch.tensor(iou3ds.mean()).type_as(cls_prec)
            iou3d_gt_mean = torch.tensor((iou3ds >= cfg.IOU_THRESH).mean()).type_as(cls_prec)

        losses = {'total_loss': loss, 'cls_loss': cls_loss, 'center_loss': center_loss, 'head_cls_loss': heading_class_loss, 
                  'head_res_loss': heading_res_norm_loss, 'size_cls_loss': size_class_loss, 'size_res_loss': size_res_norm_loss, 
                  'corners_loss': corners_loss}
        metrics = {'cls_acc': cls_prec, 'head_acc': heading_prec, 'size_acc': size_prec, 'IoU_2D': iou2d_mean, 
                   'IoU_3D': iou3d_mean, 'IoU_' + str(cfg.IOU_THRESH): iou3d_gt_mean}

        return losses, metrics