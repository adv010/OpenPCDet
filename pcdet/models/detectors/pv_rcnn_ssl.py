import copy
import os
import pickle
import numpy as np
import torch
from pcdet.datasets.augmentor import augmentor_utils
from pcdet.ops.iou3d_nms import iou3d_nms_utils
from pcdet.ops.roiaware_pool3d import roiaware_pool3d_utils
from .detector3d_template import Detector3DTemplate
from .pv_rcnn import PVRCNN
from matplotlib import pyplot as plt
import torch.nn.functional as F
from pcdet.utils import common_utils
from pcdet.utils.stats_utils import metrics_registry
from pcdet.utils.prototype_utils import feature_bank_registry
from collections import defaultdict
# from ssod import AdaptiveThresholding
#from visual_utils import open3d_vis_utils as V
from sklearn.metrics import precision_score


def _arr2dict2(array, ignore_zeros=False, ignore_nan=False):
    def should_include(value):
        return not ((ignore_zeros and value == 0) or (ignore_nan and np.isnan(value)))

    classes = ['Car', 'Pedestrian', 'Cyclist']
    classes = classes[:len(array)] 
    return {cls: array[cind] for cind, cls in enumerate(classes) if should_include(array[cind])}

class PVRCNN_SSL(Detector3DTemplate):
    def __init__(self, model_cfg, num_class, dataset):
        super().__init__(model_cfg=model_cfg, num_class=num_class, dataset=dataset)
        # something changes so need deep copy
        model_cfg_copy = copy.deepcopy(model_cfg)
        dataset_copy = copy.deepcopy(dataset)
        self.pv_rcnn = PVRCNN(model_cfg=model_cfg, num_class=num_class, dataset=dataset)

        self.pv_rcnn_ema = PVRCNN(model_cfg=model_cfg_copy, num_class=num_class, dataset=dataset_copy)
        for param in self.pv_rcnn_ema.parameters():
            param.detach_()
        self.add_module('pv_rcnn', self.pv_rcnn)
        self.add_module('pv_rcnn_ema', self.pv_rcnn_ema)
        self.accumulated_itr = 0

        self.thresh = model_cfg.THRESH
        self.sem_thresh = model_cfg.SEM_THRESH
        self.unlabeled_supervise_cls = model_cfg.UNLABELED_SUPERVISE_CLS
        self.unlabeled_supervise_refine = model_cfg.UNLABELED_SUPERVISE_REFINE
        self.unlabeled_weight = model_cfg.UNLABELED_WEIGHT
        self.no_nms = model_cfg.NO_NMS
        self.supervise_mode = model_cfg.SUPERVISE_MODE
        try:
            self.fixed_batch_dict = torch.load("batch_dict.pth")
        except:
            self.fixed_batch_dict = None
        self.thresh_alg = None
        if self.model_cfg.ADAPTIVE_THRESHOLDING.ENABLE:
            self.thresh_alg = AdaptiveThresholding(**self.model_cfg.ADAPTIVE_THRESHOLDING)

        for bank_configs in model_cfg.get("FEATURE_BANK_LIST", []):
            feature_bank_registry.register(tag=bank_configs["NAME"], **bank_configs)

        for metrics_configs in model_cfg.get("METRICS_BANK_LIST", []):
            if metrics_configs.ENABLE:
                metrics_registry.register(tag=metrics_configs["NAME"], dataset=self.dataset, **metrics_configs)

        vals_to_store = ['iou_roi_pl', 'iou_roi_gt', 'pred_scores', 'teacher_pred_scores',
                         'weights', 'roi_scores', 'pcv_scores', 'num_points_in_roi', 'class_labels', 'iteration']
        self.val_dict = {val: [] for val in vals_to_store}

    @staticmethod
    def _clone_gt_boxes_and_feats(batch_dict):
        return {
            "batch_size": batch_dict['batch_size'],
            "gt_boxes": batch_dict['gt_boxes'].clone().detach(),
            "ori_gt_boxes": batch_dict['ori_gt_boxes'].clone().detach(),
            "point_coords": batch_dict['point_coords'].clone().detach(),
            "point_features": batch_dict['point_features'].clone().detach(),
            "point_cls_scores": batch_dict['point_cls_scores'].clone().detach()
        }

    def _prep_bank_inputs(self, batch_dict, inds, num_points_threshold=20):
        selected_batch_dict = self._clone_gt_boxes_and_feats(batch_dict)
        with torch.no_grad():
            batch_gt_feats = self.pv_rcnn.roi_head.pool_features(selected_batch_dict, use_gtboxes=True)

        batch_gt_feats = batch_gt_feats.view(*batch_dict['gt_boxes'].shape[:2], -1)
        bank_inputs = defaultdict(list)
        for ix in inds:
            gt_boxes = selected_batch_dict['gt_boxes'][ix]
            nonzero_mask = torch.logical_not(torch.eq(gt_boxes, 0).all(dim=-1))
            if nonzero_mask.sum() == 0:
                print(f"no gt instance in frame {batch_dict['frame_id'][ix]}")
                continue
            gt_boxes = gt_boxes[nonzero_mask]
            sample_mask = batch_dict['points'][:, 0].int() == ix
            points = batch_dict['points'][sample_mask, 1:4]
            gt_feat = batch_gt_feats[ix][nonzero_mask]
            gt_labels = gt_boxes[:, 7].int() - 1
            gt_boxes = gt_boxes[:, :7]
            ins_idxs = batch_dict['instance_idx'][ix][nonzero_mask].int()
            smpl_id = torch.from_numpy(batch_dict['frame_id'].astype(np.int32))[ix].to(gt_boxes.device)

            # filter out gt instances with too few points when updating the bank
            num_points_in_gt = roiaware_pool3d_utils.points_in_boxes_cpu(points.cpu(), gt_boxes.cpu()).sum(dim=-1)
            valid_gts_mask = (num_points_in_gt >= num_points_threshold)
            # print(f"{(~valid_gts_mask).sum()} gt instance(s) with id(s) {ins_idxs[~valid_gts_mask].tolist()}"
            #       f" and num points {num_points_in_gt[~valid_gts_mask].tolist()} are filtered")
            if valid_gts_mask.sum() == 0:
                print(f"no valid gt instances with enough points in frame {batch_dict['frame_id'][ix]}")
                continue
            bank_inputs['feats'].append(gt_feat[valid_gts_mask])
            bank_inputs['labels'].append(gt_labels[valid_gts_mask])
            bank_inputs['ins_ids'].append(ins_idxs[valid_gts_mask])
            bank_inputs['smpl_ids'].append(smpl_id)

            # valid_boxes = gt_boxes[valid_gts_mask]
            # valid_box_labels = gt_labels[valid_gts_mask]
            # self.vis(valid_boxes, valid_box_labels, points)

        return bank_inputs

    def get_pseudo_projections(self, batch_dict, sem_loss=False):
            ori_gt_boxes =  torch.chunk(batch_dict['ori_gt_boxes'],2,dim=0)[1] # get unlabeled inds
            B,N = batch_dict['ori_gt_boxes'].shape[:2]
            ori_labels = ori_gt_boxes[..., 7].int()
            nonzero_mask = torch.logical_not(torch.eq(ori_gt_boxes, 0).all(dim=-1))
            ori_pooled_features = self.pv_rcnn.roi_head.roi_grid_pool(batch_dict, use_gtboxes=False, use_ori_gtboxes=True) #16*max_Box_num,216*2, C//2
            grid_size = self.model_cfg.ROI_HEAD.ROI_GRID_POOL.GRID_SIZE
            batch_size_rcnn = ori_pooled_features.shape[0]
            ori_pooled_features = ori_pooled_features.permute(0, 2, 1). \
                contiguous().view(batch_size_rcnn, -1, grid_size, grid_size, grid_size)  # (BxN, C, 6, 6, 6)
            ori_projections = self.pv_rcnn.roi_head.stg2_projector(ori_pooled_features.view(batch_size_rcnn,-1,1))
            ori_projections = ori_projections.view(B,N,-1)
            ori_projections = torch.chunk(ori_projections,2,dim=0)[1] # get unlabeled projections
            ori_projections = ori_projections[nonzero_mask]
            ori_labels = ori_labels[nonzero_mask]
            ori_rcnn_sem_preds = self.pv_rcnn.roi_head.sem_cls_layers(ori_projections.unsqueeze(-1))
            sem_cls_targets = ori_labels.long() - 1
            ori_rcnn_sem_preds = ori_rcnn_sem_preds.squeeze(-1)
            sem_cls_preds= ori_rcnn_sem_preds.view(-1, 3)
            sem_cls_targets = sem_cls_targets.view(-1)
            precision_ori = precision_score(sem_cls_targets.view(-1).cpu().numpy(), sem_cls_preds.max(dim=-1)[1].view(-1).cpu().numpy(), average=None, labels=range(3), zero_division="warn")       
            tb_dict = {
                'rcnn_sem_cls_precision_ori': _arr2dict2(precision_ori),
                'ori_gt_classes' : _arr2dict2(np.unique(ori_labels.cpu().numpy(), return_counts=True)[1])
            }
            loss_sem_cls_ori = None
            if sem_loss==True:
                loss_sem_cls_ori = F.cross_entropy(sem_cls_preds, sem_cls_targets, reduction='mean')
                tb_dict['loss_sem_cls_ori'] = loss_sem_cls_ori.item()
            return ori_projections, ori_labels, ori_gt_boxes, loss_sem_cls_ori, tb_dict


    def get_pl_pseudo_projections(self,batch_dict, pl_boxes_tensor, pl_labels_tensor, pl_conf_scores_tensor, masks_tensor, lpcont_conf_threshold, ulb_inds):
        
        pl_labels_unfiltered = pl_labels_tensor[pl_labels_tensor>0]
        pl_conf_scores_unfiltered = pl_conf_scores_tensor[pl_labels_tensor>0]
        masks_tensor = masks_tensor[pl_labels_tensor>0]
        selected_batch_dict = self._clone_gt_boxes_and_feats(batch_dict)
        B,N = batch_dict['gt_boxes'].shape[:2]
        pl_boxes =  torch.chunk(batch_dict['gt_boxes'],2,dim=0)[1] # get unlabeled  boxes
        pl_labels = pl_boxes[..., 7].int()
        pl_projections = torch.chunk(batch_dict['projected_features_gt'],2,dim=0)[1] # get projections for all unlabeled samples
        pl_projections = pl_projections.view(B//2,-1,256)
        # pl_projections = torch.chunk(pl_projections,2,dim=0)[1] # get projections for all unlabeled samples

        # Strip off zero padding
        valid_mask = torch.logical_not(torch.eq(pl_boxes, 0).all(dim=-1))
        pl_boxes = pl_boxes[valid_mask]
        pl_labels = pl_labels[valid_mask]
        pl_projections = pl_projections[valid_mask]
        
        assert torch.equal(pl_labels,pl_labels_unfiltered[masks_tensor])
        # Via above check, we confirm the gt_boxes after 0 stripping same as unfiltered[masked] pls
        
        lpcont_conf_thresh = torch.tensor(lpcont_conf_threshold, device = pl_conf_scores_tensor.device)
        lpcont_mask = pl_labels_unfiltered.clone()
        # Enter values of lpcont_conf_thresh for each label into lpcont_mask
        lpcont_mask[torch.where(pl_labels_unfiltered==1)[0]] = lpcont_conf_thresh[0]
        lpcont_mask[torch.where(pl_labels_unfiltered==2)[0]] = lpcont_conf_thresh[1]
        lpcont_mask[torch.where(pl_labels_unfiltered==3)[0]] = lpcont_conf_thresh[2]

        #Repeat conf * sem_thresh (pl_filtering_step)
        lpcont_mask =  lpcont_mask[masks_tensor]
        lpcont_conf_scores = pl_conf_scores_unfiltered[masks_tensor]
        # Finally select top confident PLs for LPCONT projections
        lpcont_final_mask = pl_conf_scores_unfiltered[masks_tensor] > lpcont_mask

        assert pl_projections.size(0) == lpcont_mask.size(0)

        #Apply LPCont Mask
        lpcont_boxes = pl_boxes[lpcont_final_mask]
        lpcont_labels = pl_labels[lpcont_final_mask]
        lpcont_conf_scores = lpcont_conf_scores[lpcont_final_mask]
        lpcont_projections = pl_projections[lpcont_final_mask]

        assert lpcont_projections.size(0) == lpcont_labels.size(0) == lpcont_conf_scores.size(0) == lpcont_boxes.size(0)
        return lpcont_projections, lpcont_labels, lpcont_conf_scores, lpcont_boxes 

    def _prep_wa_bank_inputs(self, batch_dict_ema, inds, ulb_inds, bank, iteration, num_points_threshold=20):
        projections_gt = batch_dict_ema['projected_features_gt']
        projections_gt = projections_gt.view(*batch_dict_ema['gt_boxes'].shape[:2], -1)
        
        projections_gt = torch.chunk(projections_gt,2,dim=0)[0] #  Teacher's labeled projections on GT boxes

        bank_inputs = defaultdict(list)
        for ix in inds:

            gt_boxes = batch_dict_ema['gt_boxes'][ix]
            gt_conf_preds = batch_dict_ema['gt_conf_scores'][ix]
            nonzero_mask = torch.logical_not(torch.eq(gt_boxes, 0).all(dim=-1))
            if nonzero_mask.sum() == 0:
                print(f"no gt instance in frame {batch_dict_ema['frame_id'][ix]}")
                continue
            gt_boxes = gt_boxes[nonzero_mask]
            sample_mask = batch_dict_ema['points'][:, 0].int() == ix
            points = batch_dict_ema['points'][sample_mask, 1:4]
            gt_feat = projections_gt[ix][nonzero_mask] # Store labeled projections into bank
            gt_labels = gt_boxes[:, 7].int()
            gt_boxes = gt_boxes[:, :7]
            ins_idxs = batch_dict_ema['instance_idx'][ix][nonzero_mask].int()
            smpl_id = torch.from_numpy(batch_dict_ema['frame_id'].astype(np.int32))[ix].to(gt_boxes.device)
            num_points_in_gt = roiaware_pool3d_utils.points_in_boxes_cpu(points.cpu(), gt_boxes.cpu()).sum(dim=-1)
            valid_gts_mask = (num_points_in_gt >= num_points_threshold)
            if valid_gts_mask.sum() == 0:
                print(f"no valid gt instances with enough points in frame {batch_dict_ema['frame_id'][ix]}")
                continue
            bank_inputs['feats'].append(gt_feat[valid_gts_mask]) # NOTE: Should have no_grad. Labeled features from teacher
            bank_inputs['labels'].append(gt_labels[valid_gts_mask])
            bank_inputs['ins_ids'].append(ins_idxs[valid_gts_mask])
            bank_inputs['smpl_ids'].append(smpl_id)
            bank_inputs['conf_scores'].append(gt_conf_preds[nonzero_mask][valid_gts_mask])
        return bank_inputs    


    def forward(self, batch_dict):
        if self.training:
            return self._forward_training(batch_dict)

        for cur_module in self.pv_rcnn.module_list:
            batch_dict = cur_module(batch_dict)
        pred_dicts, recall_dicts = self.pv_rcnn.post_processing(batch_dict)

        return pred_dicts, recall_dicts, {}
    @torch.no_grad()
    def _gen_pseudo_labels(self, batch_dict_ema):
        # self.pv_rcnn_ema.eval()  # https://github.com/yezhen17/3DIoUMatch-PVRCNN/issues/6
        for cur_module in self.pv_rcnn_ema.module_list:
            try:
                batch_dict_ema = cur_module(batch_dict_ema, test_only=True)
            except TypeError as e:
                batch_dict_ema = cur_module(batch_dict_ema)


    @staticmethod
    def _split_batch(batch_dict, tag='ema'):
        assert tag in ['ema', 'pre_gt_sample'], f'{tag} not in list [ema, pre_gt_sample]'
        # batch_dict['instance_idx_ema'] = batch_dict['instance_idx']
        batch_dict['frame_id_ema'] = batch_dict['frame_id']
        batch_dict['ori_gt_boxes'] = batch_dict['gt_boxes'].clone()
        batch_dict['ori_gt_boxes_ema'] = batch_dict['gt_boxes'].clone()
        batch_dict_out = {}
        keys = list(batch_dict.keys())
        for k in keys:
            if f'{k}_{tag}' in keys:
                continue
            if k.endswith(f'_{tag}'):
                batch_dict_out[k[:-(len(tag)+1)]] = batch_dict[k]
                batch_dict.pop(k)
            if k in ['batch_size']:
                batch_dict_out[k] = batch_dict[k]
        return batch_dict_out

    @staticmethod
    def _prep_batch_dict(batch_dict):
        labeled_mask = batch_dict['labeled_mask'].view(-1)
        labeled_inds = torch.nonzero(labeled_mask).squeeze(1).long()
        unlabeled_inds = torch.nonzero(1 - labeled_mask).squeeze(1).long()
        batch_dict['unlabeled_inds'] = unlabeled_inds
        batch_dict['labeled_inds'] = labeled_inds
        batch_dict['ori_unlabeled_boxes'] = batch_dict['gt_boxes'][unlabeled_inds, ...].clone().detach()
        return labeled_inds, unlabeled_inds

    @staticmethod
    def pad_tensor(tensor_in, max_len=50):
        assert tensor_in.dim() == 3, "Input tensor should be of shape (N, M, C), input shape is {}".format(tensor_in.shape)
        diff_ = max_len - tensor_in.shape[1]
        if diff_>0:
            tensor_in = torch.cat([tensor_in, torch.zeros((tensor_in.shape[0], diff_, tensor_in.shape[-1]), device=tensor_in.device)], dim=1)
        return tensor_in

    def _get_thresh_alg_inputs(self, scores, logits):
        scores_pad = [self.pad_tensor(s.unsqueeze(0).unsqueeze(2), max_len=100) for s in scores]
        logits_pad = [self.pad_tensor(s.unsqueeze(0), max_len=100) for s in logits]
        scores_pad = torch.cat(scores_pad).clone().detach()
        logits_pad = torch.cat(logits_pad).clone().detach()

        return scores_pad, logits_pad

    # This is being used for debugging the loss functions, specially the new ones,
    # to see if they can be minimized to zero or converged to their lowest expected value or not.
    def _get_fixed_batch_dict(self):
        batch_dict_out = {}
        if self.fixed_batch_dict is None:
            return
        for k, v in self.fixed_batch_dict.items():
            if isinstance(v, torch.Tensor):
                batch_dict_out[k] = v.clone().detach()
            else:
                batch_dict_out[k] = copy.deepcopy(v)
        return batch_dict_out

    def _forward_training(self, batch_dict):
        # batch_dict = self._get_fixed_batch_dict()
        lbl_inds, ulb_inds = self._prep_batch_dict(batch_dict)
        batch_dict_ema = self._split_batch(batch_dict, tag='ema')
        if self.model_cfg['ROI_HEAD'].get('ENABLE_ORI_LPCONT_LOSS', False)==True and self.model_cfg['ROI_HEAD'].get('ENABLE_LPCONT_LOSS', False)==True:
            raise AssertionError("Both LPCONT and ORI_LPCONT loss cannot be enabled at the same time")

        if self.supervise_mode == 1:
            pl_boxes, pl_conf_scores, pl_sem_scores, pl_sem_logits, pl_rect_scores, masks = self._get_gt_pls(batch_dict_ema, ulb_inds)
            pl_cls_count_pre_filter = torch.bincount(batch_dict_ema['gt_boxes'][ulb_inds, :, -1].view(-1).int(), minlength=4)[1:]
            pl_weights = [scores.new_ones(scores.shape[0], 1) for scores in pl_conf_scores]  # No weights for now
        else:
            self._gen_pseudo_labels(batch_dict_ema)
            pls_teacher_wa, _ = self.pv_rcnn_ema.post_processing(batch_dict_ema, no_recall_dict=True)
            ulb_pred_labels = torch.cat([pl['pred_labels'] for pl in pls_teacher_wa]).int().detach()
            pl_cls_count_pre_filter = torch.bincount(ulb_pred_labels, minlength=4)[1:]
            (pl_boxes, pl_conf_scores, pl_sem_scores, pl_sem_logits,
             pl_rect_scores, masks, pl_weights) = self._filter_pls(pls_teacher_wa, batch_dict_ema, ulb_inds)
                # pl_boxes_top, pl_conf_scores_top,pl_sem_scores_top, _,_, masks_top = self._filter_topk_pls(pls_teacher_wa, batch_dict_ema, ulb_inds)
            # if :
            #     lpcont_conf_thresh = self.model_cfg['ROI_HEAD'].get('LPCONT_CONF_THRESH')
            #     mask_lpcont_conf = pl_conf_scores > lpcont_conf_thresh

            pl_boxes_tensor = torch.cat((pl_boxes),dim=0)
            pl_conf_scores_tensor = torch.cat((pl_conf_scores),dim=0)
            pl_sem_scores_tensor = torch.cat((pl_sem_scores),dim=0)
            pl_labels_tensor =  torch.cat((pl_boxes),dim=0)[:,-1]
            masks_tensor = torch.cat((masks),dim=0)
            # assert pl_conf_scores_tensor.shape[0] == pl_labels_tensor.shape[0] == pl_sem_scores_tensor.shape[0] == masks_tensor.shape[0]
            pl_weights = [scores.new_ones(scores.shape[0], 1) for scores in pl_conf_scores]  # No weights for now

            if self.thresh_alg is not None:
                self._update_thresh_alg(pl_conf_scores, pl_sem_logits, pl_rect_scores, batch_dict_ema, pls_teacher_wa, lbl_inds)

        if 'pl_metrics' in metrics_registry.tags():
            self._update_pl_metrics(pl_boxes, pl_rect_scores, pl_weights, masks, batch_dict_ema['gt_boxes'][ulb_inds])

        # TODO(farzad): Check if commenting the following line and apply_augmentation is equal to a fully supervised setting
        self._fill_with_pls(batch_dict, pl_boxes, masks, ulb_inds, lbl_inds)

        pl_cls_count_post_filter = torch.bincount(batch_dict['gt_boxes'][ulb_inds][...,7].view(-1).int().detach(), minlength=4)[1:]
        gt_cls_count = torch.bincount(batch_dict['ori_unlabeled_boxes'][...,-1].view(-1).int().detach(), minlength=4)[1:]

        pl_count_dict = {'avg_num_gts_per_sample': self._arr2dict(gt_cls_count / len(ulb_inds)),  # backward compatibility. Added to stats_utils. Will be removed later.
                         'avg_num_pls_pre_filter_per_sample': self._arr2dict(pl_cls_count_pre_filter / len(ulb_inds)),
                         # backward compatibility. Added to stats_utils. Will be removed later.
                         'avg_num_pls_post_filter_per_sample': self._arr2dict(pl_cls_count_post_filter / len(ulb_inds))}

        # apply student's augs on teacher's pseudo-labels (filtered) only (not points)
        batch_dict = self.apply_augmentation(batch_dict, batch_dict, ulb_inds, key='gt_boxes')

        for cur_module in self.pv_rcnn.module_list:
            batch_dict = cur_module(batch_dict)

        if self.model_cfg['ROI_HEAD'].get('ENABLE_PROTOTYPING', False):
            # Update the bank with student's features from augmented labeled data
            bank = feature_bank_registry.get('gt_wa_lbl_prototypes')
            if bank.is_initialized() and self.model_cfg['ROI_HEAD'].get('ENABLE_LPCONT_LOSS', False):
                lpcont_conf_threshold = self.model_cfg['ROI_HEAD'].get('LPCONT_CONF_THRESH')
                pseudo_projections, pseudo_labels, pseudo_conf_scores, pseudo_boxes = self.get_pl_pseudo_projections(batch_dict, pl_boxes_tensor, pl_labels_tensor, pl_conf_scores_tensor, masks_tensor, lpcont_conf_threshold, ulb_inds)
            elif bank.is_initialized() and ((self.model_cfg['ROI_HEAD'].get('ENABLE_ORI_LPCONT_LOSS', False)) or (self.model_cfg['ROI_HEAD'].get('ORI_SEM_CE', False))):
                ori_pseudo_projections, ori_labels, ori_gt_boxes, loss_sem_cls_ori, tb_dicts = self.get_pseudo_projections(batch_dict,sem_loss=self.model_cfg['ROI_HEAD'].get('ORI_SEM_CE')) 
            
            wa_gt_lbl_inputs = self._prep_wa_bank_inputs(batch_dict_ema, lbl_inds, ulb_inds, bank, batch_dict['cur_iteration'], bank.num_points_thresh)
            bank.update(**wa_gt_lbl_inputs,iteration=batch_dict['cur_iteration'])


        # For metrics calculation
        self.pv_rcnn.roi_head.forward_ret_dict['unlabeled_inds'] = ulb_inds

        if self.model_cfg['ROI_HEAD'].get('ENABLE_SOFT_TEACHER', False):
            # using teacher to evaluate student's bg/fg proposals through its rcnn head
            with torch.no_grad():
                self._add_teacher_scores(batch_dict, batch_dict_ema, ulb_inds)

        disp_dict = {}
        loss_rpn_cls, loss_rpn_box, loss_rpn_dir, tb_dict = self.pv_rcnn.dense_head.get_loss()
        loss_point, tb_dict = self.pv_rcnn.point_head.get_loss(tb_dict)
        loss_rcnn_cls, loss_rcnn_box, tb_dict = self.pv_rcnn.roi_head.get_loss(tb_dict)

        loss = 0
        # Use the same reduction method as the baseline model (3diou) by the default
        reduce_loss_fn = getattr(torch, self.model_cfg.REDUCE_LOSS, 'sum')
        loss += reduce_loss_fn(loss_rpn_cls[lbl_inds, ...])
        loss += reduce_loss_fn(loss_rpn_box[lbl_inds, ...]) + reduce_loss_fn(loss_rpn_box[ulb_inds, ...]) * self.unlabeled_weight
        loss += reduce_loss_fn(loss_rpn_dir[lbl_inds, ...]) + reduce_loss_fn(loss_rpn_dir[ulb_inds, ...]) * self.unlabeled_weight
        loss += reduce_loss_fn(loss_point[lbl_inds, ...])
        loss += reduce_loss_fn(loss_rcnn_cls[lbl_inds, ...])
        loss += reduce_loss_fn(loss_rcnn_box[lbl_inds, ...])

        if self.unlabeled_supervise_cls:
            loss += reduce_loss_fn(loss_rpn_cls[ulb_inds, ...]) * self.unlabeled_weight
        if self.model_cfg['ROI_HEAD'].get('ENABLE_SOFT_TEACHER', False) or self.model_cfg.get('UNLABELED_SUPERVISE_OBJ', False):
            loss += reduce_loss_fn(loss_rcnn_cls[ulb_inds, ...]) * self.unlabeled_weight
        if self.unlabeled_supervise_refine:
            loss += reduce_loss_fn(loss_rcnn_box[ulb_inds, ...]) * self.unlabeled_weight
        if self.model_cfg['ROI_HEAD'].get('ENABLE_PROTO_CONTRASTIVE_LOSS', False):
            proto_cont_loss = self._get_proto_contrastive_loss(batch_dict, bank, ulb_inds)
            if proto_cont_loss is not None:
                loss += proto_cont_loss * self.model_cfg['ROI_HEAD']['PROTO_CONTRASTIVE_LOSS_WEIGHT']
                tb_dict['proto_cont_loss'] = proto_cont_loss.item()
        if self.model_cfg['ROI_HEAD'].get('ENABLE_ORI_LPCONT_LOSS', False):
            if not bank.is_initialized():
                ori_lpcont_loss = None
                sim_matrix =  None
            else:
                CLIP_CE = self.model_cfg['ROI_HEAD'].get('CLIP_CE', False)
                ori_lpcont_loss, sim_matrix = self._get_lpcont_loss(batch_dict, bank, ori_pseudo_projections, ori_labels, ori_gt_boxes,CLIP_CE)
            if ori_lpcont_loss is not None:
                loss += ori_lpcont_loss * self.model_cfg['ROI_HEAD']['LPCONT_LOSS_WEIGHT']
                tb_dict['ori_lpcont_loss'] = ori_lpcont_loss.item()
                tb_dict['sim_matrix'] = sim_matrix
                tb_dict['rcnn_sem_cls_precision_ori'] = tb_dicts['rcnn_sem_cls_precision_ori']
                tb_dict['ori_gt_classes'] = tb_dicts['ori_gt_classes']
        if self.model_cfg['ROI_HEAD'].get('ENABLE_LPCONT_LOSS', False):
            if not bank.is_initialized():
                lpcont_loss = None
                sim_matrix = None            
            else:
                CLIP_CE = self.model_cfg['ROI_HEAD'].get('CLIP_CE', False)
                lpcont_loss, sim_matrix = self._get_lpcont_loss_pls(batch_dict, bank, pseudo_projections, pseudo_labels, pseudo_conf_scores, pseudo_boxes,CLIP_CE)
            if lpcont_loss is not None:
                loss += lpcont_loss * self.model_cfg['ROI_HEAD']['LPCONT_LOSS_WEIGHT']
                tb_dict['lpcont_loss'] = lpcont_loss.item()
                tb_dict['sim_matrix'] = sim_matrix
        if self.model_cfg['ROI_HEAD'].get('ORI_SEM_CE', False) and bank.is_initialized():
                loss += 5 * loss_sem_cls_ori
                tb_dict['loss_sem_cls_ori'] = tb_dicts['loss_sem_cls_ori']
                tb_dict['rcnn_sem_cls_precision_ori'] = tb_dicts['rcnn_sem_cls_precision_ori']
                tb_dict['ori_gt_classes'] = tb_dicts['ori_gt_classes']

        tb_dict_ = self._prep_tb_dict(tb_dict, lbl_inds, ulb_inds, reduce_loss_fn)
        tb_dict_.update(**pl_count_dict)

        if self.model_cfg['ROI_HEAD'].get('ENABLE_ULB_CLS_DIST_LOSS', False):
            roi_head_forward_dict = self.pv_rcnn.roi_head.forward_ret_dict
            ulb_loss_cls_dist, cls_dist_dict = self.pv_rcnn.roi_head.get_ulb_cls_dist_loss(roi_head_forward_dict)
            loss += ulb_loss_cls_dist
            tb_dict_.update(cls_dist_dict)

        if self.model_cfg.get('STORE_SCORES_IN_PKL', False):
            self.dump_statistics(batch_dict, ulb_inds)

        if self.model_cfg['ROI_HEAD'].get('ENABLE_PROTOTYPING', False):

            for tag in feature_bank_registry.tags():
                feature_bank_registry.get(tag).compute()
            if feature_bank_registry._banks['gt_wa_lbl_prototypes']._computed is not None:
                prototype_cls_features, prototype_id_features, prototype_id_labels, num_updates = bank.get_computed_protos()
                class_proto_precision_dict= self.pv_rcnn_ema.roi_head.evaluate_class_prototype_rcnn_sem_precision(prototype_cls_features, torch.unique(prototype_id_labels), tb_dict_)
                inst_proto_precision_dict = self.pv_rcnn_ema.roi_head.evaluate_instance_prototype_rcnn_sem_precision(prototype_id_features, prototype_id_labels, tb_dict_)
                tb_dict_.update(**class_proto_precision_dict)
                tb_dict_.update(**inst_proto_precision_dict)
        # update dynamic thresh alg
        if self.thresh_alg is not None and (results := self.thresh_alg.compute()):
            tb_dict_.update(results)

        for tag in metrics_registry.tags():
            results = metrics_registry.get(tag).compute()
            if results is not None:
                tb_dict_.update({f"{tag}/{k}": v for k, v in zip(*results)})

        ret_dict = {
            'loss': loss
        }
        return ret_dict, tb_dict_, disp_dict

    def get_max_iou(self, anchors, gt_boxes, gt_classes, matched_threshold=0.6):
        num_anchors = anchors.shape[0]
        num_gts = gt_boxes.shape[0]

        ious = torch.zeros((num_anchors,), dtype=torch.float, device=anchors.device)
        labels = torch.ones((num_anchors,), dtype=torch.int64, device=anchors.device) * -1
        gt_to_anchor_max = torch.zeros((num_gts,), dtype=torch.float, device=anchors.device)

        if len(gt_boxes) > 0 and anchors.shape[0] > 0:
            anchor_by_gt_overlap = iou3d_nms_utils.boxes_iou3d_gpu(anchors[:, 0:7], gt_boxes[:, 0:7])
            gt_to_anchor_max = anchor_by_gt_overlap.max(dim=0)[0]
            anchor_to_gt_argmax = anchor_by_gt_overlap.argmax(dim=1)
            anchor_to_gt_max = anchor_by_gt_overlap[
                torch.arange(num_anchors, device=anchors.device), anchor_to_gt_argmax]

            pos_inds = anchor_to_gt_max >= matched_threshold
            gt_inds_over_thresh = anchor_to_gt_argmax[pos_inds]
            labels[pos_inds] = gt_classes[gt_inds_over_thresh]
            ious[:len(anchor_to_gt_max)] = anchor_to_gt_max

        return ious, labels, gt_to_anchor_max

    def _get_gt_pls(self, batch_dict, ulb_inds):
        pl_boxes = []
        pl_rect_scores = []
        pl_weights = []
        masks = []
        pl_sem_logits = []
        pl_conf_scores = []
        pl_sem_scores = []
        for i in ulb_inds:
            pboxes = batch_dict['gt_boxes'][i]
            pboxes = pboxes[pboxes[:, -1] != 0]
            pl_boxes.append(pboxes)
            sem_scores = torch.zeros((pboxes.shape[0], 3), device=pboxes.device).scatter_(1, pboxes[:, -1].long().view(-1, 1) - 1, 1)
            sem_logits = torch.zeros((pboxes.shape[0], 3), device=pboxes.device).scatter_(1, pboxes[:, -1].long().view(-1, 1) - 1, 1)
            pl_sem_logits.append(sem_logits)
            pl_rect_scores.append(sem_scores)
            pl_weights.append(torch.ones((pboxes.shape[0], 1), device=pboxes.device))
            pl_conf_scores.append(torch.zeros((pboxes.shape[0], 1), device=pboxes.device))
            pl_sem_scores.append(torch.zeros((pboxes.shape[0], 1), device=pboxes.device))
            masks.append(torch.ones((pboxes.shape[0],), dtype=torch.bool, device=pboxes.device))
        return pl_boxes, pl_conf_scores, pl_sem_scores, pl_sem_logits, pl_rect_scores, masks

    def _calc_true_ious(self, pls: [torch.Tensor], batch_gts: torch.Tensor):
        batch_ious = []
        for batch_idx, sample_pls in enumerate(pls):
            ious = torch.zeros((sample_pls.shape[0],), dtype=torch.float, device=sample_pls.device)
            gts = batch_gts[batch_idx]
            mask_gt = torch.logical_not(torch.all(gts == 0, dim=-1))
            mask_pl = torch.logical_not(torch.all(sample_pls == 0, dim=-1))

            valid_gts = gts[mask_gt]
            valid_pls = sample_pls[mask_pl]

            if len(valid_gts) > 0 and len(valid_pls) > 0:
                valid_gts_labels = valid_gts[:, -1].long() - 1
                valid_pls_labels = valid_pls[:, -1].long() - 1
                matched_threshold = torch.tensor(np.array([0.7, 0.5, 0.5]), dtype=torch.float, device=valid_pls_labels.device)[valid_pls_labels]
                valid_pls_iou_wrt_gt, assigned_label, gt_to_pls_max_iou = self.get_max_iou(valid_pls[:, 0:7],
                                                                                           valid_gts[:, 0:7],
                                                                                           valid_gts_labels,
                                                                                           matched_threshold=matched_threshold)
                ious[mask_pl] = valid_pls_iou_wrt_gt
            batch_ious.append(ious)

        return batch_ious

    def _arr2dict(self, array):
        return {cls: array[cind] for cind, cls in enumerate(self.class_names)}

    def _get_proto_contrastive_loss(self, batch_dict, bank, ulb_inds):
        gt_boxes = batch_dict['gt_boxes']
        B, N = gt_boxes.shape[:2]
        sa_pl_feats = self.pv_rcnn.roi_head.pool_features(batch_dict, use_gtboxes=True,shared=True).view(B * N, -1)
        pl_labels = batch_dict['gt_boxes'][..., 7].view(-1).long() - 1
        proto_cont_loss = bank.get_proto_contrastive_loss(sa_pl_feats, pl_labels)
        if proto_cont_loss is None:
            return
        nonzero_mask = torch.logical_not(torch.eq(gt_boxes, 0).all(dim=-1))
        ulb_nonzero_mask = nonzero_mask[ulb_inds]
        if ulb_nonzero_mask.sum() == 0:
            print(f"No pl instances predicted for strongly augmented frame(s) {batch_dict['frame_id'][ulb_inds]}")
            return
        return proto_cont_loss.view(B, N)[ulb_inds][ulb_nonzero_mask].mean()

    def _get_lpcont_loss(self, batch_dict, bank, ori_pseudo_projections, ori_labels, ori_gt_boxes, CLIP_CE):
        topk_list=[5,5,5]
        lp_cont_loss, sim_matrix = bank.get_lpcont_loss_ori(ori_pseudo_projections, ori_labels, topk_list, CLIP_CE)
        if lp_cont_loss is None:
            return
        return lp_cont_loss, sim_matrix


    def _get_lpcont_loss_pls(self, batch_dict, bank, pseudo_projections, pseudo_labels, pseudo_conf_scores, pseudo_boxes, CLIP_CE):
        topk_list=[5,5,5]
        lp_cont_loss ,sim_matrix = bank.get_lpcont_loss_pls(pseudo_projections, pseudo_labels, topk_list,pseudo_conf_scores, CLIP_CE)
        if lp_cont_loss is None:
            return
        return lp_cont_loss, sim_matrix


    @staticmethod
    def _prep_tb_dict(tb_dict, lbl_inds, ulb_inds, reduce_loss_fn):
        tb_dict_ = {}
        for key in tb_dict.keys():
            if key == 'proto_cont_loss' or key == 'ori_lpcont_loss' or key=='lpcont_loss' or key=='loss_sem_cls_ori' or key=='rcnn_sem_cls_precision_ori' or key=='ori_gt_classes':
                tb_dict_[key] = tb_dict[key]
            elif 'loss' in key or 'acc' in key or 'point_pos_num' in key:
                tb_dict_[f"{key}_labeled"] = reduce_loss_fn(tb_dict[key][lbl_inds, ...])
                tb_dict_[f"{key}_unlabeled"] = reduce_loss_fn(tb_dict[key][ulb_inds, ...])
            elif 'sim_matrix' in key:
                sim_matrix, labels, pseudo_labels = tb_dict[key]
                sim_matrix = sim_matrix.detach().cpu().numpy()
                labels = labels.cpu().numpy()
                pseudo_labels = pseudo_labels.cpu().numpy()
                fig, ax = plt.subplots(figsize=(max(4, len(pseudo_labels) * 0.2), max(4, len(labels) * 0.2))) 
                ax.imshow(sim_matrix, interpolation='nearest', cmap=plt.cm.Blues,vmin=0, vmax=1) #(sim_matrix, interpolation='nearest', cmap=plt.cm.Blues,vmin=0, vmax=1)
                ax.set_title(" LPCont Similarity matrix")
                fig.colorbar(ax.imshow(sim_matrix, interpolation='nearest', cmap=plt.cm.Blues))
                x_tick_marks = np.arange(len(pseudo_labels))
                ax.set_xticks(x_tick_marks)
                ax.set_xticklabels(pseudo_labels, rotation=45)
                y_tick_marks = np.arange(len(labels))   
                ax.set_yticks(y_tick_marks)
                ax.set_yticklabels(labels)
                ax.set_ylabel('Labeled features')
                ax.set_xlabel('Pseudo features')
                fig.tight_layout()
                tb_dict_[key] = fig
            else:
                tb_dict_[key] = tb_dict[key]

        return tb_dict_

    def _add_teacher_scores(self, batch_dict, batch_dict_ema, ulb_inds):
        batch_dict_std = {'unlabeled_inds': batch_dict['unlabeled_inds'],
                          'labeled_inds': batch_dict['labeled_inds'],
                          'rois': batch_dict['rois'].data.clone(),
                          'roi_scores': batch_dict['roi_scores'].data.clone(),
                          'roi_labels': batch_dict['roi_labels'].data.clone(),
                          'has_class_labels': batch_dict['has_class_labels'],
                          'batch_size': batch_dict['batch_size'],
                          # using teacher features
                          'point_features': batch_dict_ema['point_features'].data.clone(),
                          'point_coords': batch_dict_ema['point_coords'].data.clone(),
                          'point_cls_scores': batch_dict_ema['point_cls_scores'].data.clone()
        }

        batch_dict_std = self.reverse_augmentation(batch_dict_std, batch_dict, ulb_inds)

        # Perturb Student's ROIs before using them for Teacher's ROI head
        if self.model_cfg.ROI_HEAD.ROI_AUG.get('ENABLE', False):
            augment_rois = getattr(augmentor_utils, self.model_cfg.ROI_HEAD.ROI_AUG.AUG_TYPE, augmentor_utils.roi_aug_ros)
            # rois_before_aug is used only for debugging, can be removed later
            batch_dict_std['rois_before_aug'] = batch_dict_std['rois'].clone().detach()
            batch_dict_std['rois'][ulb_inds] = augment_rois(batch_dict_std['rois'][ulb_inds], self.model_cfg.ROI_HEAD)

        self.pv_rcnn_ema.roi_head.forward(batch_dict_std, test_only=True)
        batch_dict_std = self.apply_augmentation(batch_dict_std, batch_dict, ulb_inds, key='batch_box_preds')
        pred_dicts_std, recall_dicts_std = self.pv_rcnn_ema.post_processing(batch_dict_std,
                                                                            no_recall_dict=True,
                                                                            no_nms_for_unlabeled=True)
        rcnn_cls_score_teacher = -torch.ones_like(self.pv_rcnn.roi_head.forward_ret_dict['rcnn_cls_labels'])
        batch_box_preds_teacher = torch.zeros_like(self.pv_rcnn.roi_head.forward_ret_dict['batch_box_preds'])
        for uind in ulb_inds:
            rcnn_cls_score_teacher[uind] = pred_dicts_std[uind]['pred_scores']
            batch_box_preds_teacher[uind] = pred_dicts_std[uind]['pred_boxes']

        self.pv_rcnn.roi_head.forward_ret_dict['rcnn_cls_score_teacher'] = rcnn_cls_score_teacher
        self.pv_rcnn.roi_head.forward_ret_dict['batch_box_preds_teacher'] = batch_box_preds_teacher # for metrics

    @staticmethod
    def vis(boxes, box_labels, points):
        boxes = boxes.cpu().numpy()
        points = points.cpu().numpy()
        box_labels = box_labels.cpu().numpy()
        V.draw_scenes(points=points, gt_boxes=boxes, gt_labels=box_labels)

    def _update_thresh_alg(self, pl_conf_scores, pl_sem_logits, pl_rect_scores, batch_dict_ema, pls_teacher_wa, lbl_inds):
        thresh_inputs = dict()
        pl_scores_lbl = [pls_teacher_wa[i]['pred_scores'] for i in lbl_inds]
        pl_sem_logits_lbl = [pls_teacher_wa[i]['pred_sem_logits'] for i in lbl_inds]
        conf_scores_wa_lbl, sem_scores_wa_lbl = self._get_thresh_alg_inputs(pl_scores_lbl, pl_sem_logits_lbl)
        conf_scores_wa_ulb, sem_scores_wa_ulb = self._get_thresh_alg_inputs(pl_conf_scores, pl_sem_logits)
        thresh_inputs['conf_scores_wa'] = torch.cat([conf_scores_wa_lbl, conf_scores_wa_ulb])
        thresh_inputs['sem_scores_wa'] = torch.cat([sem_scores_wa_lbl, sem_scores_wa_ulb])
        thresh_inputs['gt_labels_wa'] = self.pad_tensor(batch_dict_ema['gt_boxes'][..., 7:8], max_len=100).detach().clone()
        thresh_inputs['gts_wa'] = self.pad_tensor(batch_dict_ema['gt_boxes'], max_len=100).detach().clone()
        pls_ws = [torch.cat([pl['pred_boxes'], pl['pred_labels'].view(-1, 1)], dim=-1) for pl in pls_teacher_wa]
        thresh_inputs['pls_wa'] = torch.cat([self.pad_tensor(pl.unsqueeze(0), max_len=100) for pl in pls_ws]).detach().clone()
        # TODO: Note that the following sem_scores rect are not filtered (since adamatch is dependent on them)
        ulb_rect_scores = torch.cat([self.pad_tensor(scores.unsqueeze(0), max_len=100) for scores in pl_rect_scores]).detach().clone()
        lb_rect_scores = torch.ones_like(ulb_rect_scores)
        thresh_inputs['scores_rect'] = torch.cat([lb_rect_scores, ulb_rect_scores])
        self.thresh_alg.update(**thresh_inputs)

    @staticmethod
    def _update_pl_metrics(pl_boxes, pl_scores, pl_weights, masks, gts):
        metrics_input = dict()
        metrics_input['rois'] = [pbox[mask] for pbox, mask in zip(pl_boxes, masks)]
        metrics_input['roi_scores'] = [score[mask] for score, mask in zip(pl_scores, masks)]
        metrics_input['ground_truths'] = [gtb for gtb in gts]
        metrics_input['roi_weights'] = [weight[mask] for weight, mask in zip(pl_weights, masks)]
        metrics_registry.get('pl_metrics').update(**metrics_input)

    def _filter_pls(self, pls_dict, batch_dict_ema, ulb_inds):
        pl_boxes = []
        pl_scores = []
        pl_sem_scores = []
        pl_sem_logits = []
        pl_rect_scores = []
        pl_weights = []
        masks = []

        def _fill_with_zeros():
            pl_boxes.append(labels.new_zeros((1, 8)).float())
            pl_scores.append(labels.new_zeros((1,)).float())
            pl_sem_scores.append(labels.new_zeros((1,)).float())
            pl_sem_logits.append(labels.new_zeros((1, 3)).float())
            pl_rect_scores.append(labels.new_zeros((1, 3)).float())
            pl_weights.append(labels.new_ones((1,)))
            masks.append(labels.new_ones((1,), dtype=torch.bool))

        for ind in ulb_inds:
            scores = pls_dict[ind]['pred_scores']  # Using gt scores for now
            boxs = pls_dict[ind]['pred_boxes']
            labels = pls_dict[ind]['pred_labels']
            sem_scores = pls_dict[ind]['pred_sem_scores']
            sem_logits = pls_dict[ind]['pred_sem_logits']

            if len(labels) == 0:
                _fill_with_zeros()
                continue
            else:
                if self.thresh_alg:
                    # Uncomment the following two lines to use the true ious as conf scores for the adaptive thresholding
                    pl_bboxes = torch.cat([boxs, labels.view(-1, 1).float()], dim=1)
                    scores = self._calc_true_ious([pl_bboxes], [batch_dict_ema['gt_boxes'][ind]])[0]
                    mask, rect_scores, weights = self.thresh_alg.get_mask(scores, sem_logits)
                else:  # 3dioumatch baseline
                    conf_thresh = torch.tensor(self.thresh, device=labels.device).expand_as(
                        sem_logits).gather(dim=1, index=(labels - 1).unsqueeze(-1)).squeeze()
                    sem_thresh = torch.tensor(self.sem_thresh, device=labels.device).unsqueeze(0).expand_as(
                        sem_logits).gather(dim=1, index=(labels - 1).unsqueeze(-1)).squeeze()
                    mask_conf = scores > conf_thresh
                    mask_sem = sem_scores > sem_thresh
                    mask = mask_conf & mask_sem
                    rect_scores = torch.sigmoid(sem_logits)  # in the baseline we don't rectify the scores
                    weights = torch.ones_like(scores)
                if mask.sum() == 0:
                    _fill_with_zeros()
                    continue

                pl_boxes.append(torch.cat([boxs, labels.view(-1, 1).float()], dim=1))
                pl_scores.append(scores)
                pl_sem_scores.append(sem_scores)
                pl_sem_logits.append(sem_logits)
                pl_rect_scores.append(rect_scores)
                pl_weights.append(weights)
                masks.append(mask)

        return pl_boxes, pl_scores, pl_sem_scores, pl_sem_logits, pl_rect_scores, masks, pl_weights

    @staticmethod
    def _fill_with_pls(batch_dict, pseudo_boxes, masks, ulb_inds, lb_inds, key=None):
        key = 'gt_boxes' if key is None else key
        max_box_num = batch_dict[key].shape[1]
        pseudo_boxes = [pboxes[mask] for pboxes, mask in zip(pseudo_boxes, masks)]

        # # Expand the gt_boxes to have the same shape as the pseudo_boxes
        # gt_scores = torch.zeros((batch_dict['gt_boxes'].shape[0], max_box_num, 1), device=batch_dict['gt_boxes'].device)
        # batch_dict['gt_boxes'] = torch.cat([batch_dict['gt_boxes'], gt_scores], dim=-1)
        # # Make sure that scores of labeled boxes are always 1, except for the padding rows which should remain zero.
        # valid_inds_lbl = torch.logical_not(torch.eq(batch_dict['gt_boxes'][labeled_inds], 0).all(dim=-1)).nonzero().long()
        # batch_dict['gt_boxes'][valid_inds_lbl[:, 0], valid_inds_lbl[:, 1], 8] = 1

        # Ignore the count of pseudo boxes if filled with default values(zeros) when no preds are made
        max_pseudo_box_num = max(
            [torch.logical_not(torch.all(ps_box == 0, dim=-1)).sum().item() for ps_box in pseudo_boxes])

        if max_box_num >= max_pseudo_box_num:
            for i, pseudo_box in enumerate(pseudo_boxes):
                diff = max_box_num - pseudo_box.shape[0]
                if diff > 0:
                    pseudo_box = torch.cat([pseudo_box, torch.zeros((diff, pseudo_box.shape[-1]), device=pseudo_box.device)], dim=0)
                batch_dict[key][ulb_inds[i]] = pseudo_box
        else:
            ori_boxes = batch_dict['gt_boxes']
            ori_ins_ids = batch_dict['instance_idx']
            new_boxes = torch.zeros((ori_boxes.shape[0], max_pseudo_box_num, ori_boxes.shape[-1]), device=ori_boxes.device)
            new_ins_idx = torch.full((ori_boxes.shape[0], max_pseudo_box_num), fill_value=-1, device=ori_boxes.device)
            for idx in lb_inds:
                diff = max_pseudo_box_num - ori_boxes[idx].shape[0]
                new_box = torch.cat([ori_boxes[idx], torch.zeros((diff, ori_boxes.shape[-1]), device=ori_boxes[idx].device)], dim=0)
                new_boxes[idx] = new_box
                new_ins_idx[idx] = torch.cat([ori_ins_ids[idx], -torch.ones((diff,), device=ori_boxes[idx].device)], dim=0)
            for i, pseudo_box in enumerate(pseudo_boxes):

                diff = max_pseudo_box_num - pseudo_box.shape[0]
                if diff > 0:
                    pseudo_box = torch.cat([pseudo_box, torch.zeros((diff, pseudo_box.shape[-1]), device=pseudo_box.device)], dim=0)
                new_boxes[ulb_inds[i]] = pseudo_box
            batch_dict[key] = new_boxes
            batch_dict['instance_idx'] = new_ins_idx

    @staticmethod
    def apply_augmentation(batch_dict, batch_dict_org, unlabeled_inds, key='rois'):
        batch_dict[key][unlabeled_inds] = augmentor_utils.random_flip_along_x_bbox(
            batch_dict[key][unlabeled_inds], batch_dict_org['flip_x'][unlabeled_inds])
        batch_dict[key][unlabeled_inds] = augmentor_utils.random_flip_along_y_bbox(
            batch_dict[key][unlabeled_inds], batch_dict_org['flip_y'][unlabeled_inds])
        batch_dict[key][unlabeled_inds] = augmentor_utils.global_rotation_bbox(
            batch_dict[key][unlabeled_inds], batch_dict_org['rot_angle'][unlabeled_inds])
        batch_dict[key][unlabeled_inds] = augmentor_utils.global_scaling_bbox(
            batch_dict[key][unlabeled_inds], batch_dict_org['scale'][unlabeled_inds])

        batch_dict[key][unlabeled_inds, :, 6] = common_utils.limit_period(
            batch_dict[key][unlabeled_inds, :, 6], offset=0.5, period=2 * np.pi
        )

        return batch_dict

    @staticmethod
    def reverse_augmentation(batch_dict, batch_dict_org, unlabeled_inds, key='rois'):
        batch_dict[key][unlabeled_inds] = augmentor_utils.global_scaling_bbox(
            batch_dict[key][unlabeled_inds], 1.0 / batch_dict_org['scale'][unlabeled_inds])
        batch_dict[key][unlabeled_inds] = augmentor_utils.global_rotation_bbox(
            batch_dict[key][unlabeled_inds], - batch_dict_org['rot_angle'][unlabeled_inds])
        batch_dict[key][unlabeled_inds] = augmentor_utils.random_flip_along_y_bbox(
            batch_dict[key][unlabeled_inds], batch_dict_org['flip_y'][unlabeled_inds])
        batch_dict[key][unlabeled_inds] = augmentor_utils.random_flip_along_x_bbox(
            batch_dict[key][unlabeled_inds], batch_dict_org['flip_x'][unlabeled_inds])

        batch_dict[key][unlabeled_inds, :, 6] = common_utils.limit_period(
            batch_dict[key][unlabeled_inds, :, 6], offset=0.5, period=2 * np.pi
        )

        return batch_dict

    def update_global_step(self):
        self.global_step += 1
        self.accumulated_itr += 1
        if self.accumulated_itr % self.model_cfg.EMA_UPDATE_INTERVAL != 0:
            return
        alpha = self.model_cfg.EMA_ALPHA
        # Use the true average until the exponential average is more correct
        alpha = min(1 - 1 / (self.global_step + 1), alpha)
        for ema_param, param in zip(self.pv_rcnn_ema.parameters(), self.pv_rcnn.parameters()):
            ema_param.data.mul_(alpha).add_((1 - alpha) * param.data)
        self.accumulated_itr = 0

    def load_params_from_file(self, filename, logger, to_cpu=False):
        if not os.path.isfile(filename):
            raise FileNotFoundError

        logger.info('==> Loading parameters from checkpoint %s to %s' % (filename, 'CPU' if to_cpu else 'GPU'))
        loc_type = torch.device('cpu') if to_cpu else None
        checkpoint = torch.load(filename, map_location=loc_type)
        model_state_disk = checkpoint['model_state']

        filename_semcls = '/mnt/data/adat01/adv_OpenPCDet/output/cfgs/kitti_models/pv_rcnn/pretrain_instance_id_rcnn_sem_cls_precision2/ckpt/checkpoint_epoch_56.pth'
        if not os.path.isfile(filename_semcls):
            raise FileNotFoundError
        checkpoint_rcnn_sem_cls = torch.load(filename_semcls, map_location=loc_type)
        model_state_disk_semcls = checkpoint_rcnn_sem_cls['model_state']
        print("Loading sem_cls_layers from '/mnt/data/adat01/adv_OpenPCDet/output/cfgs/kitti_models/pv_rcnn/pretrain_instance_id_rcnn_sem_cls_precision2/ckpt/checkpoint_epoch_56.pth")
        target_cls_keys = ['roi_head.sem_cls_layers.0.weight', 'roi_head.sem_cls_layers.1.weight', 'roi_head.sem_cls_layers.1.bias', 'roi_head.sem_cls_layers.1.running_mean', 'roi_head.sem_cls_layers.1.running_var', 'roi_head.sem_cls_layers.1.num_batches_tracked', 'roi_head.sem_cls_layers.4.weight', 'roi_head.sem_cls_layers.5.weight', 'roi_head.sem_cls_layers.5.bias', 'roi_head.sem_cls_layers.5.running_mean', 'roi_head.sem_cls_layers.5.running_var', 'roi_head.sem_cls_layers.5.num_batches_tracked', 'roi_head.sem_cls_layers.7.weight', 'roi_head.sem_cls_layers.7.bias']
        if 'version' in checkpoint:
            logger.info('==> Checkpoint trained from version: %s' % checkpoint['version'])

        update_model_state = {}
        for key, val in model_state_disk.items():
            new_key = 'pv_rcnn.' + key
            if new_key in self.state_dict() and self.state_dict()[new_key].shape == model_state_disk[key].shape:
                update_model_state[new_key] = val
                # logger.info('Update weight %s: %s' % (key, str(val.shape)))
            new_key = 'pv_rcnn_ema.' + key
            if new_key in self.state_dict() and self.state_dict()[new_key].shape == model_state_disk[key].shape:
                update_model_state[new_key] = val
            new_key = key
            if new_key in self.state_dict() and self.state_dict()[new_key].shape == model_state_disk[key].shape:
                update_model_state[new_key] = val
       
        for keys,vals in model_state_disk_semcls.items():
            if keys in target_cls_keys: #roi_head.sem_cls_layers.0.weight
                new_keys = 'pv_rcnn.' + keys
                if new_keys in self.state_dict() and self.state_dict()[new_keys].shape == model_state_disk_semcls[keys].shape:
                    update_model_state[new_keys] = vals
                new_keys = 'pv_rcnn_ema.' + keys
                if new_keys in self.state_dict() and self.state_dict()[new_keys].shape == model_state_disk_semcls[keys].shape:
                    update_model_state[new_keys] = vals
                new_keys = keys
                if new_keys in self.state_dict() and self.state_dict()[new_keys].shape == model_state_disk_semcls[keys].shape:
                    update_model_state[new_keys] = vals

        state_dict = self.state_dict()
        state_dict.update(update_model_state)
        self.load_state_dict(state_dict)

        for key in state_dict:
            if key not in update_model_state:
                logger.info('Not updated weight %s: %s' % (key, str(state_dict[key].shape)))

        logger.info('==> Done (loaded %d/%d)' % (len(update_model_state), len(self.state_dict())))
