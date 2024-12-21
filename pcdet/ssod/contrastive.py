import torch
from pcdet.ops.roiaware_pool3d.roiaware_pool3d_utils import points_in_boxes_gpu
from pcdet.utils.loss_utils import DINOLoss
from train_utils.semi_utils import transform_aug, load_data_to_gpu
from tools.visual_utils import open3d_vis_utils as V
from pcdet.models import build_network
import copy
from torch import nn


class Contrastive(nn.Module):
    def __init__(self, cfgs, datasets, *args, **kwargs):
        super().__init__(*args, **kwargs)

        model_cfgs = copy.deepcopy(cfgs.MODEL)
        self.teacher = build_network(model_cfg=model_cfgs, num_class=len(cfgs.CLASS_NAMES), dataset=datasets['labeled'])
        model_cfgs = copy.deepcopy(cfgs.MODEL)
        self.student = build_network(model_cfg=model_cfgs, num_class=len(cfgs.CLASS_NAMES), dataset=datasets['labeled'])

        if cfgs.OPTIMIZATION.SEMI_SUP_LEARNING.TEACHER.NUM_ITERS_PER_UPDATE == -1:  # for pseudo label
            self.teacher.eval()  # Set to eval mode to avoid BN update and dropout
        else:  # for EMA teacher with consistency
            self.teacher.train()  # Set to train mode
        for t_param in self.teacher.parameters():
            t_param.requires_grad = False

        self.add_module('teacher', self.teacher)
        self.add_module('student', self.student)

        self.conf_thresh = torch.tensor(cfgs.OPTIMIZATION.SEMI_SUP_LEARNING.CONF_THRESH, dtype=torch.float32)
        self.sem_thresh = torch.tensor(cfgs.OPTIMIZATION.SEMI_SUP_LEARNING.SEM_THRESH, dtype=torch.float32)
        self.cat_lbl_ulb = cfgs.OPTIMIZATION.SEMI_SUP_LEARNING.CAT_LBL_ULB
        self.rpn_cls_ulb = cfgs.MODEL.RPN_CLS_LOSS_ULB
        self.rpn_reg_ulb = cfgs.MODEL.RPN_REG_LOSS_ULB
        self.rcnn_cls_ulb = cfgs.MODEL.RCNN_CLS_LOSS_ULB
        self.rcnn_reg_ulb = cfgs.MODEL.RCNN_REG_LOSS_ULB
        self.unlabeled_weight = cfgs.MODEL.UNLABELED_WEIGHT
        self.cfgs = cfgs

        if cfgs.MODEL.ROI_HEAD.DINO_LOSS.get('ENABLE', False):
            self.dino_loss = DINOLoss(**cfgs.MODEL.ROI_HEAD.DINO_LOSS)

    @torch.no_grad()
    def _forward_test_teacher(self, batch_dict):
        # self.pv_rcnn_ema.eval()  # https://github.com/yezhen17/3DIoUMatch-PVRCNN/issues/6
        for cur_module in self.teacher.module_list:
            try:
                batch_dict = cur_module(batch_dict, test_only=True)
            except TypeError as e:
                batch_dict = cur_module(batch_dict)

    def _forward_student(self, batch_dict):
        for cur_module in self.student.module_list:
            batch_dict = cur_module(batch_dict)

    @staticmethod
    def pack_boxes(pseudo_boxes):
        # Ignore the count of pseudo boxes if filled with default values(zeros) when no preds are made
        max_pseudo_box_num = max([torch.logical_not(torch.all(ps_box == 0, dim=-1)).sum().item() for ps_box in pseudo_boxes])
        bs = len(pseudo_boxes)
        dim = pseudo_boxes[0].shape[-1]
        new_boxes = torch.zeros((bs, max_pseudo_box_num, dim), device=pseudo_boxes[0].device)
        for i, pseudo_box in enumerate(pseudo_boxes):
            new_boxes[i, :pseudo_box.shape[0]] = pseudo_box
        return new_boxes

    def mask_dense_rois(self, batch_points, rois, inds=None, num_points=10):
        if inds is None:
            inds = torch.arange(rois.shape[0], device=rois.device)
        ulb_valid_rois_mask = []
        for i, ui in enumerate(inds):
            mask = batch_points[:, 0] == ui
            points = batch_points[mask, 1:4]
            box_idxs = points_in_boxes_gpu(points.unsqueeze(0), rois[i].unsqueeze(0))  # (num_points,)
            box_idxs = box_idxs[box_idxs >= 0]  # remove points that are not in any box
            # Count the number of points in each box
            box_point_counts = torch.bincount(box_idxs, minlength=rois[i].shape[0])
            valid_roi_mask = box_point_counts >= num_points
            ulb_valid_rois_mask.append(valid_roi_mask)
        ulb_valid_rois_mask = torch.vstack(ulb_valid_rois_mask)
        return ulb_valid_rois_mask

    @staticmethod
    def clone_dict(batch_dict, keys, by_ref=False):
        new_dict = {}
        for k in keys:
            if by_ref or not isinstance(batch_dict[k], torch.Tensor):
                new_dict[k] = batch_dict[k]
            else:
                new_dict[k] = batch_dict[k].clone()
        return new_dict

    def _get_dino_feats(self, batch_dict, rois, model='teacher', source_trans_dict=None):
        rois = rois.clone().detach()
        input_keys = ['points', 'voxels', 'voxel_coords', 'voxel_num_points', 'batch_size']
        # TODO(farzad): Check if by_ref=True is applicable
        batch_dict_tmp = self.clone_dict(batch_dict, input_keys, by_ref=False)
        if source_trans_dict is not None:
            rois = transform_aug(rois, source_trans_dict, batch_dict)

        if model == 'teacher':
            self._forward_test_teacher(batch_dict_tmp)
            batch_dict_tmp['rois'] = rois  # replace with the transformed rois
            with torch.no_grad():
                batch_feats = self.teacher.roi_head.get_proj_feats(batch_dict_tmp)
        else:
            # TODO(farzad): IS THIS WORKAROUND SAFE? Fix the issue with a better solution!
            batch_dict_tmp['gt_boxes'] = torch.zeros((rois.shape[0], 1, 8), device=rois.device)  # dummy
            self._forward_student(batch_dict_tmp)
            batch_dict_tmp['rois'] = rois  # replace with the transformed rois
            batch_feats = self.student.roi_head.get_proj_feats(batch_dict_tmp)

        batch_feats = batch_feats.view(*rois.shape[:2], -1)
        assert torch.logical_not(torch.eq(rois, 0).all(dim=-1)).all().item(), 'rois should not be zero!'
        return batch_feats

    def forward(self, batch_dict_wa_lbl, batch_dict_wa_ulb, batch_dict_sa_lbl, batch_dict_sa_ulb, epoch_id):
        load_data_to_gpu(batch_dict_wa_ulb)
        load_data_to_gpu(batch_dict_sa_lbl)
        load_data_to_gpu(batch_dict_sa_ulb)

        self._forward_test_teacher(batch_dict_wa_ulb)
        preds_ema, _ = self.teacher.post_processing(batch_dict_wa_ulb, no_recall_dict=True)
        pls = self.filter_pls(preds_ema)
        pl_boxes_sa = transform_aug(pls['boxes'], batch_dict_wa_ulb, batch_dict_sa_ulb)
        batch_dict_sa_ulb['gt_boxes'] = self.pack_boxes(pl_boxes_sa)

        self._forward_student(batch_dict_sa_lbl)

        loss = 0
        tb_dict, disp_dict = {}, {}
        loss_lbl, tb_dict_lbl, disp_dict_lbl = self.student.get_training_loss()
        loss += loss_lbl

        for cur_module in self.student.module_list:
            batch_dict_sa_ulb = cur_module(batch_dict_sa_ulb)

        loss_rpn_cls, loss_rpn_reg, tb_dict_rpn_ulb = self.student.dense_head.get_loss(separate_losses=True)
        loss_rcnn_cls, loss_rcnn_reg, tb_dict_rcnn_ulb = self.student.roi_head.get_loss(separate_losses=True)

        if self.rpn_cls_ulb:
            loss += loss_rpn_cls * self.unlabeled_weight
        if self.rpn_reg_ulb:
            loss += loss_rpn_reg * self.unlabeled_weight
        if self.rcnn_cls_ulb:
            loss += loss_rcnn_cls * self.unlabeled_weight
        if self.rcnn_reg_ulb:
            loss += loss_rcnn_reg * self.unlabeled_weight

        # TODO(farzad): Think about the point loss: Note that the point fg/bg predictions
        #  are directly used in roi_grid_pool and thus affect the roi proj features
        # loss_point, tb_dict = self.point_head.get_loss(tb_dict)
        # loss += loss_point

        if self.cfgs.MODEL.ROI_HEAD.DINO_LOSS.get('ENABLE', False):
            with torch.no_grad():
                # TODO: Design decision: use only student rois for both teacher and student? Check the quality of RoIs.
                rois = batch_dict_sa_ulb['rois']
                dummy_labels = torch.zeros(rois.shape[:2], device=rois.device)  # dummy
                rois = torch.cat([rois, dummy_labels.unsqueeze(2)], dim=-1)  # Warning: no clone
                t2 = self._get_dino_feats(batch_dict_sa_ulb, rois)
                t1 = self._get_dino_feats(batch_dict_wa_ulb, rois, source_trans_dict=batch_dict_sa_ulb)
            s2 = self._get_dino_feats(batch_dict_sa_ulb, rois, model='student')
            s1 = self._get_dino_feats(batch_dict_wa_ulb, rois, model='student', source_trans_dict=batch_dict_sa_ulb)
            teacher_output = torch.cat([t1, t2], dim=0).view(-1, t1.shape[-1])  # (BxN, C) N=128
            t1_centered, t2_centered = self.dino_loss.softmax_center_teacher(teacher_output).chunk(2)
            self.dino_loss.update_center(teacher_output)
            dino_loss = self.dino_loss.forward(s1, s2, t1_centered, t2_centered)
            tb_dict.update({'dino_loss_unlabeled': dino_loss.item()})
            loss += dino_loss * self.cfgs.MODEL.ROI_HEAD.DINO_LOSS.get('WEIGHT', 1.0)

        self.merge_tb_dicts(tb_dict_lbl, tb_dict, 'labeled')
        self.merge_tb_dicts(tb_dict_rpn_ulb, tb_dict, 'unlabeled')
        self.merge_tb_dicts(tb_dict_rcnn_ulb, tb_dict, 'unlabeled')

        return loss, tb_dict, disp_dict

    @staticmethod
    def vis(points, gt_boxes, gt_labels, ref_boxes=None, ref_labels=None, ref_scores=None, attributes=None):
        gt_boxes = gt_boxes.cpu().numpy()
        points = points.cpu().numpy()
        gt_labels = gt_labels.cpu().numpy()
        ref_boxes = ref_boxes.cpu().numpy() if ref_boxes is not None else None
        ref_labels = ref_labels.cpu().numpy() if ref_labels is not None else None
        ref_scores = ref_scores.cpu().numpy() if ref_scores is not None else None
        V.draw_scenes(points=points, gt_boxes=gt_boxes, gt_labels=gt_labels, ref_boxes=ref_boxes,
                      ref_labels=ref_labels, ref_scores=ref_scores, attributes=attributes)

    def filter_pls(self, pls_dict):
        pl_boxes, pl_scores, pl_sem_scores, pl_sem_logits = [], [], [], []

        def _fill_with_zeros():
            pl_boxes.append(labels.new_zeros((1, 8)).float())
            pl_scores.append(labels.new_zeros((1,)).float())
            pl_sem_scores.append(labels.new_zeros((1,)).float())
            pl_sem_logits.append(labels.new_zeros((1, 3)).float())

        for pl_dict in pls_dict:
            scores, boxs, labels = pl_dict['pred_scores'], pl_dict['pred_boxes'], pl_dict['pred_labels']
            sem_scores, sem_logits = pl_dict['pred_sem_scores'], pl_dict['pred_sem_logits']
            if len(labels) == 0:
                _fill_with_zeros()
                continue

            assert torch.all(labels == torch.argmax(sem_logits, dim=1) + 1), \
                f"labels: {labels}, sem_scores: {sem_scores}"  # sanity check
            mask = self.iou_match_3d_filtering(scores, sem_logits)

            if mask.sum() == 0:
                _fill_with_zeros()
                continue

            pl_boxes.append(torch.cat([boxs, labels.view(-1, 1).float()], dim=1)[mask])
            pl_scores.append(scores[mask])
            pl_sem_scores.append(sem_scores[mask])
            pl_sem_logits.append(sem_logits[mask])

        return {'boxes': pl_boxes, 'scores': pl_scores, 'sem_scores': pl_sem_scores, 'sem_logits': pl_sem_logits}

    def iou_match_3d_filtering(self, conf_scores, sem_logits):
        labels = torch.argmax(sem_logits, dim=1)
        sem_scores = sem_logits.sigmoid().max(dim=1)[0]
        conf_thresh = self.conf_thresh.to(conf_scores.device).expand_as(sem_logits).gather(dim=1, index=labels.unsqueeze(-1)).squeeze()
        sem_thresh = self.sem_thresh.to(sem_logits.device).expand_as(sem_logits).gather(dim=1, index=labels.unsqueeze(-1)).squeeze()
        return (conf_scores > conf_thresh) & (sem_scores > sem_thresh)
    
    @staticmethod
    def merge_tb_dicts(source_tb_dict, target_tb_dict, postfix=None):
        for key, val in source_tb_dict.items():
            target_tb_dict[f"{key}_{postfix}"] = val
