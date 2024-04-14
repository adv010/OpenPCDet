from .detector3d_template import Detector3DTemplate
import torch
from collections import defaultdict
from pcdet.ops.iou3d_nms import iou3d_nms_utils
from pcdet.datasets.augmentor import augmentor_utils
from pcdet.ops.roiaware_pool3d import roiaware_pool3d_utils
from .detector3d_template import Detector3DTemplate
from torch.nn import functional as F
from pcdet.utils import common_utils
import os
import pickle
import copy
import numpy as np

# ./adv_OpenPCDet/output/cfgs/kitti_models/pv_rcnn/tsne_pretrain_pvrcnn/ckpt/pretrained_tsne_ckpt_81.pth
# /mnt/data/adat01/adv_OpenPCDet/output/cfgs/kitti_models/pv_rcnn/tsne_pretrain_pvrcnn/tsne_scores.pkl

# Use above location to access pretrained_ckpt with shared_features for tsne of pretrained model.

class PVRCNN(Detector3DTemplate):
    def __init__(self, model_cfg, num_class, dataset):
        super().__init__(model_cfg=model_cfg, num_class=num_class, dataset=dataset)
        self.module_list = self.build_networks()
        self.model_cfg = model_cfg
        vals_to_store = ['iou_roi_pl', 'iou_roi_gt', 'obj_scores','gt_boxes',
                         'roi_scores','num_points_in_roi', 'class_labels', 'iteration', 'shared_features','frame_id', 'shared_features_gt']
        self.val_dict = defaultdict(list)
        for val in vals_to_store:
            self.val_dict[val] = []
    def forward(self, batch_dict):
        for cur_module in self.module_list:
            batch_dict = cur_module(batch_dict)

        if self.training:
            labeled_mask = batch_dict['labeled_mask'].view(-1)
            labeled_inds = torch.nonzero(labeled_mask).squeeze(1).long()
            batch_dict['labeled_inds'] = labeled_inds
            loss, tb_dict, disp_dict = self.get_training_loss(batch_dict)
            ret_dict = {
                    'loss': loss
                }
            if self.model_cfg.get('STORE_SCORES_IN_PKL', False):
                self.dump_statistics(batch_dict, labeled_inds)

            return ret_dict, tb_dict, disp_dict
        else:
            pred_dicts, recall_dicts = self.post_processing(batch_dict)
            return pred_dicts, recall_dicts, {}

    def get_training_loss(self,batch_dict):
        disp_dict = {}
        loss_rpn, tb_dict = self.dense_head.get_loss()
        loss_point, tb_dict = self.point_head.get_loss(tb_dict)
        loss_rcnn, tb_dict = self.roi_head.get_loss(tb_dict)
        loss = loss_rpn + loss_point + loss_rcnn
        return loss, tb_dict, disp_dict

    def dump_statistics(self, batch_dict, labeled_inds):
        embed_size = 256
        batch_size = batch_dict['batch_size']
        # Store different types of scores over all itrs and epochs and dump them in a pickle for offline modeling
        # TODO (shashank) : Can be optimized later to save computational time, currently takes about 0.002sec
        # batch_roi_labels = self.pv_rcnn.roi_head.forward_ret_dict['roi_labels'][unlabeled_inds]
        self.val_dict['frame_id'].extend(batch_dict['frame_id'].tolist())
        # self.val_dict['shared_features'].extend(batch_dict['shared_features'])
        # self.val_dict['shared_features_gt'].extend(batch_dict['shared_features_gt'])
        # self.val_dict['gt_boxes'].extend(batch_dict['gt_boxes'])
        batch_roi_labels =  self.roi_head.forward_ret_dict['roi_labels'][labeled_inds]
        batch_roi_labels = [roi_labels.clone().detach() for roi_labels in batch_roi_labels]

        batch_rois = self.roi_head.forward_ret_dict['rois'] [labeled_inds]
        batch_rois = [rois.clone().detach() for rois in batch_rois]

        batch_ori_gt_boxes = batch_dict['gt_boxes'] #self.roi_head.forward_ret_dict['ori_unlabeled_boxes']
        batch_ori_gt_boxes = [ori_gt_boxes.clone().detach() for ori_gt_boxes in batch_ori_gt_boxes]

        shared_features = batch_dict['shared_features'].view(batch_size,-1,embed_size)
        shared_features_gt = batch_dict['shared_features_gt'].view(batch_size,-1,embed_size)

        for i in range(len(batch_rois)): # B
            valid_rois_mask = torch.logical_not(torch.all(batch_rois[i] == 0, dim=-1))
            valid_rois = batch_rois[i][valid_rois_mask]
            valid_roi_labels = batch_roi_labels[i][valid_rois_mask]
            valid_roi_labels -= 1  # Starting class indices from zero

            
            sh_ft =  shared_features[i][valid_rois_mask]
            self.val_dict['shared_features'].extend(sh_ft)

            valid_gt_boxes_mask = torch.logical_not(torch.all(batch_ori_gt_boxes[i] == 0, dim=-1))
            valid_gt_boxes = batch_ori_gt_boxes[i][valid_gt_boxes_mask]
            valid_gt_boxes[:, -1] -= 1  # Starting class indices from zero

            sh_ft_gt =  shared_features_gt[i][valid_gt_boxes_mask]
            self.val_dict['shared_features_gt'].extend(sh_ft_gt)

            # shared_features_gt = batch_dict['shared_features_gt'].view(batch_dict['batch_size'], -) 

            num_gts = valid_gt_boxes_mask.sum()
            num_preds = valid_rois_mask.sum()

            cur_unlabeled_ind = labeled_inds[i]
            if num_gts > 0 and num_preds > 0:
                # Find IoU between Student's ROI v/s Original GTs
                overlap = iou3d_nms_utils.boxes_iou3d_gpu(valid_rois[:, 0:7], valid_gt_boxes[:, 0:7])
                preds_iou_max, assigned_gt_inds = overlap.max(dim=1)
                self.val_dict['iou_roi_gt'].extend(preds_iou_max.tolist())

                cur_iou_roi_pl = self.roi_head.forward_ret_dict['gt_iou_of_rois'][cur_unlabeled_ind]
                self.val_dict['iou_roi_pl'].extend(cur_iou_roi_pl.tolist())

                cur_pred_score = torch.sigmoid(batch_dict['batch_cls_preds'][cur_unlabeled_ind]).squeeze()
                self.val_dict['obj_scores'].extend(cur_pred_score.tolist())

                self.val_dict['gt_boxes'].extend(valid_gt_boxes.tolist())

                # if 'rcnn_cls_score_teacher' in self.pv_rcnn.roi_head.forward_ret_dict:
                #     cur_teacher_pred_score = self.pv_rcnn.roi_head.forward_ret_dict['rcnn_cls_score_teacher'][
                #         cur_unlabeled_ind]
                #     self.val_dict['teacher_pred_scores'].extend(cur_teacher_pred_score.tolist())

                #     cur_weight = self.pv_rcnn.roi_head.forward_ret_dict['rcnn_cls_weights'][cur_unlabeled_ind]
                #     self.val_dict['weights'].extend(cur_weight.tolist())

                cur_roi_score = torch.sigmoid(self.roi_head.forward_ret_dict['roi_scores'][cur_unlabeled_ind])
                self.val_dict['roi_scores'].extend(cur_roi_score.tolist())

                cur_roi_label = self.roi_head.forward_ret_dict['roi_labels'][cur_unlabeled_ind].squeeze()
                self.val_dict['class_labels'].extend(cur_roi_label.tolist())

                cur_iteration = torch.ones_like(preds_iou_max) * (batch_dict['cur_iteration'])
                self.val_dict['iteration'].extend(cur_iteration.tolist())


        # Check post non-zero masking whether features,labels consistent
        assert len(self.val_dict['shared_features']) == len(self.val_dict['class_labels'])
        assert len(self.val_dict['shared_features_gt']) == len(self.val_dict['gt_boxes'])
        # replace old pickle data (if exists) with updated one
        output_dir = os.path.split(os.path.abspath(batch_dict['ckpt_save_dir']))[0]
        file_path = os.path.join(output_dir, 'tsne_scores_14042024_masked.pkl')
        pickle.dump(self.val_dict, open(file_path, 'wb'))