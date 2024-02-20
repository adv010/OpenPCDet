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
import matplotlib.pyplot as plt
from pcdet.utils import common_utils
from pcdet.utils.stats_utils import metrics_registry
from pcdet.utils.prototype_utils import feature_bank_registry
from collections import defaultdict
from pcdet.utils.thresh_algs import thresh_registry
from pcdet.ops.iou3d_nms import iou3d_nms_utils
# from visual_utils import open3d_vis_utils as V


# class DynamicThreshRegistry(object):
#     def __init__(self, **kwargs):
#         self._tag_metrics = {}
#         self.dataset = kwargs.get('dataset', None)
#         self.model_cfg = kwargs.get('model_cfg', None)

#     def get(self, tag=None):
#         if tag is None:
#             tag = 'default'
#         if tag in self._tag_metrics.keys():
#             metric = self._tag_metrics[tag]
#         else:
#             metric = build_thresholding_method(tag=tag, dataset=self.dataset, config=self.model_cfg)
#             self._tag_metrics[tag] = metric
#         return metric

#     def tags(self):
#         return self._tag_metrics.keys()


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
        self.hybrid_thresh = model_cfg.HYBRID_THRESH
        self.unlabeled_supervise_cls = model_cfg.UNLABELED_SUPERVISE_CLS
        self.unlabeled_supervise_refine = model_cfg.UNLABELED_SUPERVISE_REFINE
        self.unlabeled_weight = model_cfg.UNLABELED_WEIGHT
        self.no_nms = model_cfg.NO_NMS
        self.supervise_mode = model_cfg.SUPERVISE_MODE
        self.thresh_config = self.model_cfg.ADAPTIVE_THRESH_CONFIG
        try:
            self.fixed_batch_dict = torch.load("batch_dict.pth")
        except:
            self.fixed_batch_dict = None
        self.thresh_alg = None
        for key, confs in self.thresh_config.items():
            thresh_registry.register(key, **confs)
            if confs['ENABLE']:
                self.thresh_alg = thresh_registry.get(key)

        for bank_configs in model_cfg.get("FEATURE_BANK_LIST", []):
            feature_bank_registry.register(tag=bank_configs["NAME"], **bank_configs)

        for metrics_configs in model_cfg.get("METRICS_BANK_LIST", []):
            if metrics_configs.ENABLE:
                metrics_registry.register(tag=metrics_configs["NAME"], dataset=self.dataset, **metrics_configs)

            # for name in metrics_configs['NAME']:
            #     metrics_configs['tag'] = name
            #     metrics_registry.register(dataset=self.dataset, **metrics_configs)
        vals_to_store = ['iou_roi_pl', 'iou_roi_gt', 'pred_scores', 'teacher_pred_scores',
                         'weights', 'roi_scores', 'num_points_in_roi', 'class_labels', 'iteration']
        self.val_dict = {val: [] for val in vals_to_store}
        # vals_to_store = ['instloss_car', 'unscaled_instloss_cyc', 'positive_pairs_duped', 'iteration', 'pseudo_instance_sim_scores_pl', 'rcnn_scores_pl', 'pred_scores', 'negative_pairs_duped', 'assigned_gt_pl_labels', 'weights', 'instloss_cyc', 'num_points_in_roi', 'pseudo_sem_scores_pl', 'unscaled_instloss_car', 'iou_pl_gt', 'ri_sim_scores', 'roi_instance_sim_scores', 'roi_scores', 'lbl_inst_freq', 'iou_roi_pl', 'unscaled_instloss_ped', 'pseudo_sim_scores_pl', 'class_labels', 'pl_iteration', 'iou_roi_gt', 'teacher_pred_scores', 'instloss_ped']
        # self.val_dict = {val: [] for val in vals_to_store}
        # self.val_dict['lbl_inst_freq'] = [0,0,0]
        # self.val_dict['positive_pairs_duped'] = [0,0,0]
        # self.val_dict['negative_pairs_duped'] = [1,1,1]
        # loss_dict_keys = {'cos_sim_pl_wa','cos_sim_pl_sa','pl_labels','proto_labels'}
        # self.loss_dict = {key: [] for key in loss_dict_keys}
        # mcont_dict = {'logits','iteration'}
        # self.mcont_dict = {key: [] for key in mcont_dict}
        
    @staticmethod
    def _clone_gt_boxes_and_feats(batch_dict):
        return {
            "batch_size": batch_dict['batch_size'],
            "gt_boxes": batch_dict['gt_boxes'].clone().detach(),
            "point_coords": batch_dict['point_coords'].clone().detach(),
            "point_features": batch_dict['point_features'].clone().detach(),
            "point_cls_scores": batch_dict['point_cls_scores'].clone().detach()
        }

    def _prep_bank_inputs(self, batch_dict, inds, num_points_threshold=20):
        selected_batch_dict = self._clone_gt_boxes_and_feats(batch_dict)
        with torch.no_grad():
            batch_gt_feats = self.pv_rcnn_ema.roi_head.pool_features(selected_batch_dict, use_gtboxes=True)
            batch_size_rcnn = batch_gt_feats.shape[0]
            shared_features = self.pv_rcnn_ema.roi_head.shared_fc_layer(batch_gt_feats.view(batch_size_rcnn, -1, 1))
        batch_gt_feats = shared_features.view(*batch_dict['gt_boxes'].shape[:2], -1)
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
        # pseudo_labels, _ = self.pv_rcnn_ema.post_processing(batch_dict_ema, no_recall_dict=True)

        # return batch_dict_ema,pseudo_labels,batch_dict_sa 

    @staticmethod
    def _split_batch(batch_dict, tag='ema'):
        assert tag in ['ema', 'pre_gt_sample'], f'{tag} not in list [ema, pre_gt_sample]'
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
        batch_dict['ori_unlabeled_boxes_ema'] = batch_dict['gt_boxes_ema'][unlabeled_inds, ...].clone().detach()
        return labeled_inds, unlabeled_inds

    @staticmethod
    def pad_tensor(tensor_in, max_len=50):
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
        self._gen_pseudo_labels(batch_dict_ema)
        pls_teacher_wa, _ = self.pv_rcnn_ema.post_processing(batch_dict_ema, no_recall_dict=True)
        pl_boxes, pl_scores, pl_sem_scores, pl_sem_logits = self._filter_pls_conf_scores(pls_teacher_wa, ulb_inds)
        ulb_pred_labels = torch.cat([pb[:, -1] for pb in pl_boxes]).int().detach()
        pl_cls_count_pre_filter = torch.bincount(ulb_pred_labels, minlength=4)[1:]

        # Semantic Filtering
        filtering_masks, pl_sem_scores_rect = self._filter_pls_sem_scores(pl_scores, pl_sem_logits)

        # TODO(farzad): Set the scores of pseudo-labels that their argmax is changed after rectification to zero.
        pl_weights = [scores_rect.max(-1, keepdim=True)[0] for scores_rect in pl_sem_scores_rect]
        # pl_weights = [scores.new_ones(scores.shape[0], 1) for scores in pl_sem_scores_rect]
        if 'pl_metrics' in metrics_registry.tags():
            metrics_input = self.get_pl_metrics_input(pl_boxes, pl_sem_scores_rect, pl_weights, filtering_masks, batch_dict_ema['gt_boxes'][ulb_inds])
            metrics_registry.get('pl_metrics').update(**metrics_input)

        pl_boxes = [torch.hstack([boxes[mask], weights[mask]]) for boxes, weights, mask in zip(pl_boxes, pl_weights, filtering_masks)]

        # Comment the following line to use gt boxes for unlabeled data!
        self._fill_with_pseudo_labels(batch_dict, pl_boxes, ulb_inds, lbl_inds)

        pl_cls_count_post_filter = torch.bincount(batch_dict['gt_boxes'][ulb_inds][...,7].view(-1).int().detach(), minlength=4)[1:]
        gt_cls_count = torch.bincount(batch_dict['ori_unlabeled_boxes'][...,-1].view(-1).int().detach(), minlength=4)[1:]

        pl_count_dict = {'avg_num_gts_per_sample': self._arr2dict(gt_cls_count / len(ulb_inds)),  # backward compatibility. Added to stats_utils. Will be removed later.
                         'avg_num_pls_pre_filter_per_sample': self._arr2dict(pl_cls_count_pre_filter / len(ulb_inds)),
                         # backward compatibility. Added to stats_utils. Will be removed later.
                         'avg_num_pls_post_filter_per_sample': self._arr2dict(pl_cls_count_post_filter / len(ulb_inds))}

        # apply student's augs on teacher's pseudo-labels (filtered) only (not points)
        batch_dict = self.apply_augmentation(batch_dict, batch_dict, ulb_inds, key='gt_boxes')
        self.update_metrics_pl(targets_dict=batch_dict)

        for cur_module in self.pv_rcnn.module_list:
            batch_dict = cur_module(batch_dict)
        
        if self.model_cfg['ROI_HEAD'].get('ENABLE_INSTANCE_SUP_LOSS', False) and  self.model_cfg['ROI_HEAD'].get('INSTANCE_CONTRASTIVE_LOSS_MODEL', False) =='Student':

            #1. @Student - Get shared_features over strongly aug GTs
            batch_dict = self.pv_rcnn.roi_head.forward(batch_dict, test_only=True,use_gtboxes=True)

            #2. @Student - Get shared_features over Weakly aug GTs
            with torch.no_grad():
                for cur_module in self.pv_rcnn.module_list:
                    try:
                        batch_dict_wa = cur_module(batch_dict_wa, test_only=True,use_gtboxes=True)
                    except TypeError as e:
                        batch_dict_wa = cur_module(batch_dict_wa)        

        if self.thresh_alg is not None:
            thresh_inputs = dict()

            # pre_gt_dict = self._split_batch(batch_dict, tag='pre_gt_sample')
            # self._gen_pseudo_labels(pre_gt_dict)
            # pls_teacher_pre_gt, _ = self.pv_rcnn_ema.post_processing(pre_gt_dict, no_recall_dict=True)
            # thresh_inputs['conf_scores_pre_gt_wa'] = torch.cat([self.pad_tensor(pl['pred_scores'].unsqueeze(0).unsqueeze(2), max_len=100) for pl in pls_teacher_pre_gt]).detach().clone()
            # thresh_inputs['sem_scores_pre_gt_wa'] = torch.cat([self.pad_tensor(pl['pred_sem_scores_logits'].unsqueeze(0), max_len=100) for pl in pls_teacher_pre_gt]).detach().clone()
            # thresh_inputs['gt_labels_pre_gt_wa'] = self.pad_tensor(pre_gt_dict['gt_boxes'][..., 7:8], max_len=100).detach().clone()

            # pls_std_sa, _ = self.pv_rcnn_ema.post_processing(batch_dict, no_recall_dict=True)
            # thresh_inputs['conf_scores_sa'] = torch.cat([self.pad_tensor(pl['pred_scores'].unsqueeze(0).unsqueeze(2), max_len=100) for pl in pls_std_sa]).detach().clone()
            # thresh_inputs['sem_scores_sa'] = torch.cat([self.pad_tensor(pl['pred_sem_scores_logits'].unsqueeze(0), max_len=100) for pl in pls_std_sa]).detach().clone()
            # thresh_inputs['gt_labels_sa'] = self.pad_tensor(batch_dict['gt_boxes'][..., 7:8], max_len=100).detach().clone()

            pl_scores_lbl = [pls_teacher_wa[i]['pred_scores'] for i in lbl_inds]
            pl_sem_logits_lbl = [pls_teacher_wa[i]['pred_sem_scores_logits'] for i in lbl_inds]
            conf_scores_wa_lbl, sem_scores_wa_lbl = self._get_thresh_alg_inputs(pl_scores_lbl, pl_sem_logits_lbl)
            conf_scores_wa_ulb, sem_scores_wa_ulb = self._get_thresh_alg_inputs(pl_scores, pl_sem_logits)
            thresh_inputs['conf_scores_wa'] = torch.cat([conf_scores_wa_lbl, conf_scores_wa_ulb])
            thresh_inputs['sem_scores_wa'] = torch.cat([sem_scores_wa_lbl, sem_scores_wa_ulb])
            thresh_inputs['gt_labels_wa'] = self.pad_tensor(batch_dict_ema['gt_boxes'][..., 7:8], max_len=100).detach().clone()

            self.thresh_alg.update(**thresh_inputs)

        if self.model_cfg['ROI_HEAD'].get('ENABLE_PROTOTYPING', False):
            # Update the bank with student's features from augmented labeled data
            bank = feature_bank_registry.get('gt_aug_lbl_prototypes')
            sa_gt_lbl_inputs = self._prep_bank_inputs(batch_dict, lbl_inds, bank.num_points_thresh)
            bank.update(**sa_gt_lbl_inputs, iteration=batch_dict['cur_iteration'])

        # For metrics calculation
        self.pv_rcnn.roi_head.forward_ret_dict['unlabeled_inds'] = ulb_inds
        

        disp_dict = {}
        loss_rpn_cls, loss_rpn_box, tb_dict = self.pv_rcnn.dense_head.get_loss()
        loss_point, tb_dict = self.pv_rcnn.point_head.get_loss(tb_dict)
        loss_rcnn_cls, loss_rcnn_box, tb_dict = self.pv_rcnn.roi_head.get_loss(tb_dict)

        loss = 0
        # Use the same reduction method as the baseline model (3diou) by the default
        reduce_loss_fn = getattr(torch, self.model_cfg.REDUCE_LOSS, 'sum')
        loss += reduce_loss_fn(loss_rpn_cls[lbl_inds, ...])
        loss += reduce_loss_fn(loss_rpn_box[lbl_inds, ...]) + reduce_loss_fn(loss_rpn_box[ulb_inds, ...]) * self.unlabeled_weight
        loss += reduce_loss_fn(loss_point[lbl_inds, ...])
        loss += reduce_loss_fn(loss_rcnn_cls[lbl_inds, ...])
        loss += reduce_loss_fn(loss_rcnn_box[lbl_inds, ...])

        if self.unlabeled_supervise_cls:
            loss += reduce_loss_fn(loss_rpn_cls[ulb_inds, ...]) * self.unlabeled_weight
        if self.model_cfg['ROI_HEAD'].get('ENABLE_SOFT_TEACHER', False) or self.model_cfg.get('UNLABELED_SUPERVISE_OBJ', False):
            loss += reduce_loss_fn(loss_rcnn_cls[ulb_inds, ...]) * self.unlabeled_weight
        if self.unlabeled_supervise_refine:
            loss += reduce_loss_fn(loss_rcnn_box[ulb_inds, ...]) * self.unlabeled_weight
        if self.model_cfg['ROI_HEAD'].get('ENABLE_INSTANCE_CONTRASTIVE_LOSS', False): 
            instance_cont_loss,classwise_instance_cont_loss = self._get_sim_instance_contrastive_loss(batch_dict,batch_dict_ema, bank, ulb_inds)
            if instance_cont_loss is not None:
                loss += instance_cont_loss * self.model_cfg['ROI_HEAD']['INSTANCE_CONTRASTIVE_LOSS_WEIGHT']
                tb_dict['instance_cont_loss'] = instance_cont_loss.item() * self.model_cfg['ROI_HEAD']['INSTANCE_CONTRASTIVE_LOSS_WEIGHT']
                for cind,class_name in enumerate(['Car','Pedestrian','Cyclist']):
                    tb_dict[f'classwise_instance_cont_loss_{class_name}'] = classwise_instance_cont_loss[f'{class_name}_Pl']
        if self.model_cfg['ROI_HEAD'].get('ENABLE_MEAN_INSTANCE_CONTRASTIVE_LOSS', False): 
            instance_cont_loss,classwise_instance_cont_loss = self._get_sim_instance_contrastive_loss(batch_dict,batch_dict_ema, bank, ulb_inds,mean_instance=True)
            if instance_cont_loss is not None:
                loss += instance_cont_loss * self.model_cfg['ROI_HEAD']['INSTANCE_CONTRASTIVE_LOSS_WEIGHT']
                tb_dict['instance_cont_loss'] = instance_cont_loss.item() * self.model_cfg['ROI_HEAD']['INSTANCE_CONTRASTIVE_LOSS_WEIGHT']
                for cind,class_name in enumerate(['Car','Pedestrian','Cyclist']):
                    tb_dict[f'classwise_instance_cont_loss_{class_name}'] = classwise_instance_cont_loss[f'{class_name}_Pl']   

        if self.model_cfg['ROI_HEAD'].get('ENABLE_PROTO_SIM_LOSS', False):
            proto_sim_loss,classwise_proto_sim = self._get_sim_instance_contrastive_loss(batch_dict,batch_dict_ema, bank, ulb_inds,proto_sim=True)
            if proto_sim_loss is not None:
                loss += proto_sim_loss * self.model_cfg['ROI_HEAD']['PROTO_SIM_LOSS_WEIGHT']
                tb_dict['proto_sim_loss'] = proto_sim_loss.item()
                for cind,class_name in enumerate(['Car','Pedestrian','Cyclist']):
                    tb_dict[f'classwise_proto_sim_loss_{class_name}'] = classwise_proto_sim[f'{class_name}_Pl']  

        if self.model_cfg['ROI_HEAD'].get('ENABLE_MCONT_LOSS', False):
            mCont_labeled = bank._get_multi_cont_loss()
            if mCont_labeled is not None:
                loss += mCont_labeled['total_loss'] * self.model_cfg['ROI_HEAD']['MCONT_LOSS_WEIGHT'] 
                tb_dict['mCont_loss'] = mCont_labeled['total_loss'].item()
                for cind,class_name in enumerate(['Car','Pedestrian','Cyclist']):
                    tb_dict[f'mCont_{class_name}_proto'] = mCont_labeled['classwise_loss'][cind].item()
                if self.model_cfg.get('STORE_RAW_SIM_IN_PKL', False):
                    self.mcont_dict['logits'].append(mCont_labeled['raw_logits'].tolist())
                    self.mcont_dict['iteration'].append(batch_dict['cur_iteration'])
                    output_dir = os.path.split(os.path.abspath(batch_dict['ckpt_save_dir']))[0]
                    file_path = os.path.join(output_dir, 'mcont_raw_logits.pkl')
                    pickle.dump(self.mcont_dict, open(file_path, 'wb'))

        if self.model_cfg['ROI_HEAD'].get('ENABLE_BATCH_MCONT', False):
            selected_batch_dict = self._clone_gt_boxes_and_feats(batch_dict)
            with torch.no_grad():
                batch_gt_feats = self.pv_rcnn_ema.roi_head.pool_features(selected_batch_dict, use_gtboxes=True)
                batch_size_rcnn = batch_gt_feats.shape[0]
                shared_features = self.pv_rcnn_ema.roi_head.shared_fc_layer(batch_gt_feats.view(batch_size_rcnn, -1, 1))
            batch_gt_feats = shared_features.view(*batch_dict['gt_boxes'].shape[:2], -1)
            batch_gt_feats_lb = batch_gt_feats[batch_dict['labeled_mask'].bool()]  
            batch_gt_labels_lb = batch_dict['gt_boxes'][batch_dict['labeled_mask'].bool()][:,:,-1].long()
            batch_gt_feats_lb = torch.cat([batch_gt_feats_lb,batch_gt_labels_lb.unsqueeze(-1)],dim=-1)
            gathered_tensor = self.gather_tensors(batch_gt_feats_lb)
            gathered_labels = gathered_tensor[:,-1].long()
            non_zero_mask = gathered_labels != 0
            gathered_feats = gathered_tensor[:,:-1][non_zero_mask]
            gathered_labels = gathered_labels[non_zero_mask]
            mCont_labeled_features = bank._get_multi_cont_loss_lb_instances(gathered_feats,gathered_labels)
            mCont_labeled_loss =  (mCont_labeled_features['total_loss'] * self.model_cfg['ROI_HEAD']['MCONT_LOSS_WEIGHT'])

            if dist.is_initialized():
                loss+= (mCont_labeled_loss/2)
                tb_dict['mCont_loss_instance'] = (mCont_labeled_features['total_loss'].item())     
            else:
                loss+= mCont_labeled_loss
                tb_dict['mCont_loss_instance'] = (mCont_labeled_features['total_loss'].item()/2) 

            for cind,class_name in enumerate(['Car','Pedestrian','Cyclist']):
                tb_dict[f'mCont_{class_name}_lb_inst'] = mCont_labeled_features['classwise_loss'][cind].item()
                if self.model_cfg.get('STORE_RAW_SIM_IN_PKL', False):
                    self.mcont_dict['logits'].append(mCont_labeled_features['raw_logits'].tolist())
                    self.mcont_dict['iteration'].append(batch_dict['cur_iteration'])
                    output_dir = os.path.split(os.path.abspath(batch_dict['ckpt_save_dir']))[0]
                    file_path = os.path.join(output_dir, 'mcont_lb_instances_logits.pkl')
                    pickle.dump(self.mcont_dict, open(file_path, 'wb'))

        if self.model_cfg['ROI_HEAD'].get('ENABLE_PROTO_CONTRASTIVE_LOSS', False):
            proto_cont_loss = self._get_proto_contrastive_loss(batch_dict, bank, ulb_inds)
            if proto_cont_loss is not None:
                loss += proto_cont_loss * self.model_cfg['ROI_HEAD']['PROTO_CONTRASTIVE_LOSS_WEIGHT']
                tb_dict['proto_cont_loss'] = proto_cont_loss.item()
        if self.model_cfg['ROI_HEAD'].get('ENABLE_INSTANCE_SUP_LOSS', False):
            #lbl_inst_cont_loss = self._get_instance_contrastive_loss(batch_dict,batch_dict_wa,lbl_inds,ulb_inds)
            if self.model_cfg.ROI_HEAD.INSTANCE_CONTRASTIVE_LOSS_MODEL=='Teacher':
                lbl_inst_cont_loss = self._get_instance_contrastive_loss(batch_dict_sa,batch_dict_ema,lbl_inds,ulb_inds)
            elif self.model_cfg.ROI_HEAD.INSTANCE_CONTRASTIVE_LOSS_MODEL =='Student' :
                lbl_inst_cont_loss, tb_dict = self._get_instance_contrastive_loss(tb_dict,batch_dict,batch_dict_wa,lbl_inds,ulb_inds)
            if lbl_inst_cont_loss is not None:
                loss +=  lbl_inst_cont_loss * self.model_cfg['ROI_HEAD']['INSTANCE_CONTRASTIVE_LOSS_WEIGHT']
        tb_dict_ = self._prep_tb_dict(tb_dict, lbl_inds, lbl_inds, reduce_loss_fn)
        tb_dict_.update(**pl_count_dict)

        if self.model_cfg['ROI_HEAD'].get('ENABLE_ULB_CLS_DIST_LOSS', False):
            roi_head_forward_dict = self.pv_rcnn.roi_head.forward_ret_dict
            ulb_loss_cls_dist, cls_dist_dict = self.pv_rcnn.roi_head.get_ulb_cls_dist_loss(roi_head_forward_dict)
            loss += ulb_loss_cls_dist
            tb_dict_.update(cls_dist_dict)

        if self.model_cfg.get('STORE_SCORES_IN_PKL', False):
            batch_dict['roi_sim_scores'] = self.pv_rcnn.roi_head.forward_ret_dict['roi_sim_scores']
            batch_dict_ema['prefilter_pls'] = dump_stats_prefilter
            self.dump_statistics(batch_dict, batch_dict_ema, ulb_inds,lbl_inds)

        if self.model_cfg['ROI_HEAD'].get('ENABLE_PROTOTYPING', False):
            for tag in feature_bank_registry.tags():
                feature_bank_registry.get(tag).compute()

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

    def _arr2dict(self, array):
        return {cls: array[cind] for cind, cls in enumerate(self.class_names)}

    def _get_proto_contrastive_loss(self, batch_dict, bank, ulb_inds):
        gt_boxes = batch_dict['gt_boxes']
        B, N = gt_boxes.shape[:2]
        sa_pl_feats = self.pv_rcnn.roi_head.pool_features(batch_dict, use_gtboxes=True).view(B * N, -1)
        batch_size_rcnn = sa_pl_feats.shape[0]
        shared_features = self.pv_rcnn.roi_head.shared_fc_layer(sa_pl_feats.view(batch_size_rcnn, -1, 1)) 
        shared_features = shared_features.squeeze(-1)     
        pl_labels = batch_dict['gt_boxes'][..., 7].view(-1).long() - 1
        proto_cont_loss = bank.get_proto_contrastive_loss(shared_features, pl_labels)
        if proto_cont_loss is None:
            return
        nonzero_mask = torch.logical_not(torch.eq(gt_boxes, 0).all(dim=-1))
        ulb_nonzero_mask = nonzero_mask[ulb_inds]
        filter_thresh = self.model_cfg['ROI_HEAD']['PL_PROTO_CONTRASTIVE_THRESH']
        valid_pl = gt_boxes[...,-1][ulb_inds][ulb_nonzero_mask].long().unsqueeze(-1) - 1
        clswise_filter_thresh = torch.tensor(filter_thresh,device=valid_pl.device).unsqueeze(0).repeat(valid_pl.shape[0],1).gather(index=(valid_pl),dim=1).squeeze(-1)
        valid_filtered_pls = (batch_dict['pseudo_scores'][ulb_inds][ulb_nonzero_mask] >= clswise_filter_thresh)
                
        if ulb_nonzero_mask.sum() == 0 or valid_filtered_pls.sum() == 0:
            print(f"No pl instances predicted for strongly augmented frame(s) {batch_dict['frame_id'][ulb_inds.cpu().numpy()]}")
            return
        return proto_cont_loss.view(B, N)[ulb_inds][ulb_nonzero_mask][valid_filtered_pls].mean()


    def _get_sim_instance_contrastive_loss(self,batch_dict,batch_dict_ema,bank,ulb_inds,mean_instance=False,proto_sim=False): #TODO: Deepika: Refactor this function
            """
            Contains the implementation of instance level contrastive losses like proto_sim, simmatch, mean_simmatch, mcont, proto_cont

            Credits: 

            Simmatch: https://arxiv.org/abs/2308.06692
            Protocon: https://arxiv.org/pdf/2303.13556.pdf
            Mcont: https://arxiv.org/pdf/2303.13556.pdf

            Args: 
                    batch_dict: dict containing  WA GTs
                    batch_dict_ema: dict containing SA GTs
                    bank: feature bank object
                    ulb_inds: unlabeled indices
                    mean_instance: bool, if True, mean instance simmatch loss is calculated
                    proto_sim: bool, if True, proto_sim loss is calculated
                Returns:
                    instance_cont_loss: instance contrastive loss
                    classwise_loss: dict containing classwise instance contrastive loss for metrics
                """
            # prepare batch_dict with gt boxes and teacher features
            batch_dict_wa_gt = {'unlabeled_inds': batch_dict['unlabeled_inds'],
                            'labeled_inds': batch_dict['labeled_inds'],
                            'rois': batch_dict['rois'].data.clone(),
                            'roi_scores': batch_dict['roi_scores'].data.clone(),
                            'roi_labels': batch_dict['roi_labels'].data.clone(),
                            'has_class_labels': batch_dict['has_class_labels'],
                            'batch_size': batch_dict['batch_size'],
                            'gt_boxes': batch_dict['gt_boxes'].data.clone(),
                            # using teacher features
                            'point_features': batch_dict_ema['point_features'].data.clone(),
                            'point_coords': batch_dict_ema['point_coords'].data.clone(),
                            'point_cls_scores': batch_dict_ema['point_cls_scores'].data.clone(),
                            
            }
            # reverse the augmentation --> Strong Augmentation to Weak Augmentation
            batch_dict_wa_gt = self.reverse_augmentation(batch_dict_wa_gt, batch_dict, ulb_inds,key='gt_boxes')
            gt_boxes = batch_dict['gt_boxes']
            B, N = gt_boxes.shape[:2]
            gt_labels = gt_boxes[..., -1].view(B,N).long() - 1
            # obtain teacher features on the student generated PLs
            with torch.no_grad():
                batch_gt_feats_wa = self.pv_rcnn_ema.roi_head.pool_features(batch_dict_wa_gt, use_gtboxes=True)
                batch_size_rcnn = batch_gt_feats_wa.shape[0]
                shared_features_wa = self.pv_rcnn_ema.roi_head.shared_fc_layer(batch_gt_feats_wa.view(batch_size_rcnn, -1, 1)).squeeze(-1)
                shared_features_wa = shared_features_wa.view(*batch_dict['gt_boxes'].shape[:2], -1)
            # obtain student features on the student generated PLs
            batch_gt_feats_sa = self.pv_rcnn.roi_head.pool_features(batch_dict, use_gtboxes=True)
            batch_size_rcnn = batch_gt_feats_sa.shape[0]
            shared_features_sa = self.pv_rcnn.roi_head.shared_fc_layer(batch_gt_feats_sa.view(batch_size_rcnn, -1, 1)).squeeze(-1)
            shared_features_sa = shared_features_sa.view(*batch_dict['gt_boxes'].shape[:2], -1)

            assert batch_gt_feats_sa.shape[0] == batch_gt_feats_wa.shape[0], "batch_dict gt features shape mismatch"
            # proto_sim loss
            if proto_sim == True:
                shared_features_wa_ulb = shared_features_wa[ulb_inds]
                shared_features_sa_ulb = shared_features_sa[ulb_inds]
                proto_sim_loss = bank.get_proto_sim_loss(shared_features_wa_ulb,shared_features_sa_ulb)
                if proto_sim_loss is None:
                    return None,None
                ulb_nonzero_mask = torch.logical_not(torch.eq(gt_boxes[ulb_inds], 0).all(dim=-1))

                pseudo_scores = batch_dict['pseudo_scores'][ulb_inds][ulb_nonzero_mask]
                pseudo_conf_thresh = self.model_cfg['ROI_HEAD']['PL_PROTO_SIM_THRESH']
                valid_pl = gt_boxes[ulb_inds][ulb_nonzero_mask][:,-1].long().unsqueeze(-1) 
                clswise_pseudo_thresh = torch.tensor(pseudo_conf_thresh,device=valid_pl.device).unsqueeze(0).repeat(valid_pl.shape[0],1).gather(index=(valid_pl-1),dim=1).squeeze(-1)
                valid_pls = (pseudo_scores >= clswise_pseudo_thresh)
                if ulb_nonzero_mask.sum() == 0 or valid_pls.sum() == 0:
                    print(f"No pl instances predicted for strongly augmented frame(s) {batch_dict['frame_id'][ulb_inds.cpu().numpy()]}")
                    return None,None
                proto_sim_loss_total = proto_sim_loss['total_loss'].view(shared_features_sa_ulb.shape[0], N,3)
                proto_loss = proto_sim_loss_total[ulb_nonzero_mask][valid_pls].sum(-1).mean()
                
                Car_instance_proto_loss =  (proto_sim_loss_total[:,:,0])[ulb_nonzero_mask] 
                Ped_instance_proto_loss =  (proto_sim_loss_total[:,:,1])[ulb_nonzero_mask]
                Cyc_instance_proto_loss =  (proto_sim_loss_total[:,:,2])[ulb_nonzero_mask]
                classwise_loss = {'Car_Pl':{},'Pedestrian_Pl':{},'Cyclist_Pl':{}}
                for cind,class_name in enumerate(['Car','Pedestrian','Cyclist']):
                    classwise_loss[f'{class_name}_Pl'] = { 
                            'Car_proto': Car_instance_proto_loss[gt_labels[ulb_inds][ulb_nonzero_mask]==cind].mean().item() * self.model_cfg['ROI_HEAD']['PROTO_SIM_LOSS_WEIGHT'],
                            'Ped_proto': Ped_instance_proto_loss[gt_labels[ulb_inds][ulb_nonzero_mask]==cind].mean().item() * self.model_cfg['ROI_HEAD']['PROTO_SIM_LOSS_WEIGHT'],
                            'Cyc_proto': Cyc_instance_proto_loss[gt_labels[ulb_inds][ulb_nonzero_mask]==cind].mean().item() * self.model_cfg['ROI_HEAD']['PROTO_SIM_LOSS_WEIGHT'],
                    }
                self.loss_dict['cos_sim_pl_wa'].append(proto_sim_loss['cos_sim_wa'].tolist())
                self.loss_dict['cos_sim_pl_sa'].append(proto_sim_loss['cos_sim_sa'].tolist())
                self.loss_dict['pl_labels'].append(gt_labels[ulb_inds][ulb_nonzero_mask].tolist())
                if self.model_cfg.get('STORE_RAW_SIM_IN_PKL', False):
                    output_dir = os.path.split(os.path.abspath(batch_dict['ckpt_save_dir']))[0]
                    file_path = os.path.join(output_dir, 'cos_sim.pkl')
                    pickle.dump(self.loss_dict, open(file_path, 'wb'))

                return proto_loss,classwise_loss

            # simmatch loss: https://arxiv.org/abs/2308.06692
            if mean_instance == False:
                instance_cont_tuple = bank.get_simmatch_loss(shared_features_wa,shared_features_sa,ulb_inds) # normal_simmatch_loss
                
                if instance_cont_tuple is None:
                    return None,None
                nonzero_mask = torch.logical_not(torch.eq(gt_boxes, 0).all(dim=-1))
                ulb_nonzero_mask = nonzero_mask[ulb_inds]
                if ulb_nonzero_mask.sum() == 0:
                    print(f"No pl instances predicted for strongly augmented frame(s) {batch_dict['frame_id'][ulb_inds.cpu().numpy()]}")
                    return None,None
                loss_labels = instance_cont_tuple[1]
                instance_cont_tuple[0] = instance_cont_tuple[0].view(B, N, -1)
                instance_cont_sum = instance_cont_tuple[0].sum(-1)# calculates sum of all terms of CE for a particular instance
                instance_cont_loss = instance_cont_sum[ulb_inds][ulb_nonzero_mask].mean()# mean of all instances           
                cos_sim_wa = instance_cont_tuple[2].view(B, N, -1)
                cos_sim_sa = instance_cont_tuple[3].view(B, N, -1)
                # metrics update
                Car_instance_proto_loss = instance_cont_tuple[0][:,:,loss_labels==0][ulb_inds][ulb_nonzero_mask].sum(-1)
                Ped_instance_proto_loss =  instance_cont_tuple[0][:,:,loss_labels==1][ulb_inds][ulb_nonzero_mask].sum(-1)
                Cyc_instance_proto_loss = instance_cont_tuple[0][:,:,loss_labels==2][ulb_inds][ulb_nonzero_mask].sum(-1)
                classwise_loss = {'Car_Pl':{},'Pedestrian_Pl':{},'Cyclist_Pl':{}}
                for cind,class_name in enumerate(['Car','Pedestrian','Cyclist']):
                    classwise_loss[f'{class_name}_Pl'] = {
                            'Car_proto': Car_instance_proto_loss[gt_labels[ulb_inds][ulb_nonzero_mask]==cind].mean().item() * self.model_cfg['ROI_HEAD']['INSTANCE_CONTRASTIVE_LOSS_WEIGHT'],
                            'Ped_proto': Ped_instance_proto_loss[gt_labels[ulb_inds][ulb_nonzero_mask]==cind].mean().item() * self.model_cfg['ROI_HEAD']['INSTANCE_CONTRASTIVE_LOSS_WEIGHT'],
                            'Cyc_proto': Cyc_instance_proto_loss[gt_labels[ulb_inds][ulb_nonzero_mask]==cind].mean().item() * self.model_cfg['ROI_HEAD']['INSTANCE_CONTRASTIVE_LOSS_WEIGHT'],
                    }

                if self.model_cfg.get('STORE_RAW_SIM_IN_PKL', False):
                    self.loss_dict['proto_labels'] = loss_labels.tolist()
                    self.loss_dict['pl_labels'].append(gt_labels[ulb_inds][ulb_nonzero_mask].tolist())
                    self.loss_dict['cos_sim_pl_wa'].append(cos_sim_wa[ulb_inds][ulb_nonzero_mask].tolist())
                    self.loss_dict['cos_sim_pl_sa'].append(cos_sim_sa[ulb_inds][ulb_nonzero_mask].tolist())
                    output_dir = os.path.split(os.path.abspath(batch_dict['ckpt_save_dir']))[0]
                    file_path = os.path.join(output_dir, 'cos_sim.pkl')
                    pickle.dump(self.loss_dict, open(file_path, 'wb'))
                return instance_cont_loss,classwise_loss
            # mean instance simmatch loss: consistency across classwise prototypes
            else:
                instance_cont_tuple = bank.get_simmatch_mean_loss(shared_features_wa,shared_features_sa,ulb_inds) # mean_simmatch_loss, instead of logits for each instance, classwise logits are used. 
                if instance_cont_tuple is None:
                    return None,None
                nonzero_mask = torch.logical_not(torch.eq(gt_boxes, 0).all(dim=-1))
                ulb_nonzero_mask = nonzero_mask[ulb_inds]
                if ulb_nonzero_mask.sum() == 0:
                    print(f"No pl instances predicted for strongly augmented frame(s) {batch_dict['frame_id'][ulb_inds.cpu().numpy()]}")
                    return None,None
                
                loss_labels = instance_cont_tuple[1]
                instance_cont_tuple[0] = instance_cont_tuple[0].view(B, N, -1)
                instance_cont_sum = instance_cont_tuple[0].sum(-1) # calculates sum of all terms of CE for a particular instance
                instance_cont_loss = instance_cont_sum[ulb_inds][ulb_nonzero_mask].mean() # mean of all instances
                
                # metrics update
                Car_instance_proto_loss = instance_cont_tuple[0][:,:,0][ulb_inds][ulb_nonzero_mask].mean(-1) 
                Ped_instance_proto_loss =  instance_cont_tuple[0][:,:,1][ulb_inds][ulb_nonzero_mask].mean(-1)
                Cyc_instance_proto_loss = instance_cont_tuple[0][:,:,2][ulb_inds][ulb_nonzero_mask].mean(-1)
                classwise_loss = {'Car_Pl':{},'Pedestrian_Pl':{},'Cyclist_Pl':{}}
                for cind,class_name in enumerate(['Car','Pedestrian','Cyclist']):
                    classwise_loss[f'{class_name}_Pl'] = {
                            'Car_proto': Car_instance_proto_loss * self.model_cfg['ROI_HEAD']['INSTANCE_CONTRASTIVE_LOSS_WEIGHT'],
                            'Ped_proto': Ped_instance_proto_loss * self.model_cfg['ROI_HEAD']['INSTANCE_CONTRASTIVE_LOSS_WEIGHT'],
                            'Cyc_proto': Cyc_instance_proto_loss * self.model_cfg['ROI_HEAD']['INSTANCE_CONTRASTIVE_LOSS_WEIGHT'],
                    }
                self.loss_dict['cos_sim_pl_wa'].append(instance_cont_tuple[2].tolist())
                self.loss_dict['cos_sim_pl_sa'].append(instance_cont_tuple[3].tolist())
                if self.model_cfg.get('STORE_RAW_SIM_IN_PKL', False):
                    output_dir = os.path.split(os.path.abspath(batch_dict['ckpt_save_dir']))[0]
                    file_path = os.path.join(output_dir, 'cos_sim.pkl')
                    pickle.dump(self.loss_dict, open(file_path, 'wb'))
                
                return instance_cont_loss,classwise_loss

    def gather_tensors(self,tensor):
            """
            Returns the gathered tensor to all GPUs in DDP else returns the tensor as such
            dist.gather_all needs the gathered tensors to be of same size.
            We get the sizes of the tensors first, zero pad them to match the size
            Then gather and filter the padding

            Args:
                tensor: tensor to be gathered
                
            """

            assert tensor.ndim == 3,"features should be of shape N,1,256"
            tensor = tensor.view(-1,257)
            
            if not dist.is_initialized():
                return tensor
                # Determine sizes first
            WORLD_SIZE = dist.get_world_size()
            local_size = torch.tensor(tensor.size(), device=tensor.device)
            all_sizes = [torch.zeros_like(local_size) for _ in range(WORLD_SIZE)]
            
            dist.barrier() 
            dist.all_gather(all_sizes,local_size)
            dist.barrier()

            print(f'all_sizes {all_sizes}')
            # make zero-padded version https://stackoverflow.com/questions/71433507/pytorch-python-distributed-multiprocessing-gather-concatenate-tensor-arrays-of
            max_length = max([size[0] for size in all_sizes])
            
            diff = max_length - local_size[0].item()
            if diff:
                pad_size =[diff.item()] #pad with zeros 
                if local_size.ndim >= 1:
                    pad_size.extend(dimension.item() for dimension in local_size[1:])
                padding = torch.zeros(pad_size, device=tensor.device, dtype=tensor.dtype)
                tensor = torch.cat((tensor,padding),)

            all_tensors_padded = [torch.zeros_like(tensor) for _ in range(WORLD_SIZE)]

            dist.barrier()
            dist.all_gather(all_tensors_padded,tensor)
            dist.barrier()

            gathered_tensor = torch.cat(all_tensors_padded)
            non_zero_mask = torch.any(gathered_tensor!=0,dim=-1).squeeze()
            gathered_tensor = gathered_tensor[non_zero_mask]
            return gathered_tensor

    def _align_instance_pairs(self, batch_dict,batch_dict_pair,indices):
        
        embed_size = 256 if not self.model_cfg['ROI_HEAD']['PROJECTOR'] else 256 # if possible, 128
        shared_ft_sa = batch_dict['shared_features_gt'].view(batch_dict['batch_size'],-1,embed_size)[indices]
        shared_ft_wa = batch_dict_pair['shared_features_gt'].view(batch_dict['batch_size'],-1,embed_size)[indices]
        device = shared_ft_sa.device
        labels_sa = batch_dict['gt_boxes'][:,:,7][indices].view(-1)
        labels_wa = batch_dict_pair['gt_boxes'][:,:,7][indices].view(-1)
        instance_idx_sa = batch_dict['instance_idx'][indices].view(-1)
        instance_idx_wa = batch_dict_pair['instance_idx'][indices].view(-1)
        shared_ft_sa = shared_ft_sa.view(-1,embed_size)
        shared_ft_wa = shared_ft_wa.view(-1,embed_size)
        
        # strip off the extra GTs
        prefinal_mask_sa = labels_sa!=0
        prefinal_mask_wa = labels_wa!=0

        instance_idx_sa = instance_idx_sa[prefinal_mask_sa]
        instance_idx_wa = instance_idx_wa[prefinal_mask_wa]

        meta_data = {'to_mask':''}
        valid_instances = np.intersect1d(instance_idx_sa.cpu().numpy(),instance_idx_wa.cpu().numpy()) #
        valid_instances = torch.tensor(valid_instances,device=device)
       
        '''intersect_mask, to remove instances from A which are not in B and VICE VERSA '''
        # intersect_mask = torch.isin(instance_idx,valid_instances) #small
        # intersect_mask_pair = torch.isin(instance_idx_pair,valid_instances)
        intersect_mask_sa = torch.tensor([idx in valid_instances for idx in instance_idx_sa], device=device, dtype=torch.bool)
        intersect_mask_wa = torch.tensor([idx in valid_instances for idx in instance_idx_wa], device=device, dtype=torch.bool)

        instance_idx_sa = instance_idx_sa[intersect_mask_sa]
        instance_idx_wa = instance_idx_wa[intersect_mask_wa]

        labels_sa = (labels_sa[prefinal_mask_sa])[intersect_mask_sa]
        labels_wa =(labels_wa[prefinal_mask_wa])[intersect_mask_wa]

        shared_ft_sa = (shared_ft_sa[prefinal_mask_sa])[intersect_mask_sa]
        shared_ft_wa = (shared_ft_wa[prefinal_mask_wa])[intersect_mask_wa]

        # remove duplicates
        if len(labels_sa) <= len(labels_wa):
            tmp = copy.deepcopy(instance_idx_sa) #small
            tmp_big = instance_idx_wa #big
            meta_data['to_mask'] = 'ft_pair'
        else:
            tmp = instance_idx_wa
            tmp_big = instance_idx_sa
            meta_data['to_mask'] = 'ft'

        '''Edge case - Handle more labeled indices in batch_dict_pair's dataloader batch than batch_dict's dataloader(or vice-versa)'''        
        # final_mask = torch.zeros_like(tmp_big)
        final_mask = []
        # args2 =[]
        for idx, item in enumerate(tmp_big,0):
            if item in tmp:
                final_mask.append(idx)
                tmp[torch.where(tmp==item)[0][0]] = -1
                

        final_mask = torch.tensor(final_mask, device=device)

        final_mask = final_mask.long()
        if meta_data['to_mask'] == 'ft':
            instance_idx_sa = tmp_big[final_mask]   
            labels_sa = labels_sa[final_mask]
            shared_ft_sa = shared_ft_sa[final_mask]
        
        elif meta_data['to_mask'] == 'ft_pair':
            instance_idx_wa = tmp_big[final_mask]   
            labels_wa = labels_wa[final_mask]
            shared_ft_wa = shared_ft_wa[final_mask]
        
        ## sort the GTs
        sorted_sa = instance_idx_sa.sort()[-1].long()
        sorted_wa = instance_idx_wa.sort()[-1].long()

        instance_idx_sa = instance_idx_sa[sorted_sa]
        instance_idx_wa = instance_idx_wa[sorted_wa]

        labels_sa = labels_sa[sorted_sa]
        labels_wa =labels_wa[sorted_wa]
        shared_ft_sa = shared_ft_sa[sorted_sa]
        shared_ft_wa = shared_ft_wa[sorted_wa]

        return labels_sa,labels_wa,instance_idx_sa,instance_idx_wa,shared_ft_sa, shared_ft_wa


    def _get_instance_contrastive_loss(self, tb_dict,batch_dict,batch_dict_pair,lbl_inds,ulb_inds,temperature=1.0,base_temperature=1.0):
        '''
        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: roi_labels[B,N].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        '''
        start_epoch = self.model_cfg['ROI_HEAD'].get(
            'INSTANCE_CONTRASTIVE_LOSS_START_EPOCH', 0)
        stop_epoch = self.model_cfg['ROI_HEAD'].get(
            'INSTANCE_CONTRASTIVE_LOSS_STOP_EPOCH', 60)
        tb_dict = {} if tb_dict is None else tb_dict
        # To examine effects of stopping supervised contrastive loss
        if not start_epoch<=batch_dict['cur_epoch']<stop_epoch:
            return
        temperature = self.model_cfg['ROI_HEAD'].get('TEMPERATURE', 1.0)
        
        if  self.model_cfg['ROI_HEAD'].get('INSTANCE_IDX',None)=="Labeled": # Apply SupConLoss over labeled indices
            indices = lbl_inds
        else:
            indices = ulb_inds
        labels_sa, labels_wa, instance_idx_sa, instance_idx_wa, embed_ft_sa, embed_ft_wa = \
            self._align_instance_pairs(batch_dict, batch_dict_pair,indices)
        batch_size_labeled = labels_sa.shape[0]
        device = embed_ft_sa.device
        labels = torch.cat((labels_sa,labels_sa), dim=0)

        assert torch.equal(instance_idx_sa, instance_idx_wa)
        assert torch.equal(labels_sa, labels_wa)   # Problem : Fails for Ulb! Same instance id , diff label for strong and weak aug ulb 
        # Record stats
        batch_dict['lbl_inst_freq'] =  torch.bincount(labels_sa.view(-1).int().detach(),minlength = 4).tolist()[1:]       #Record freq of each class in batch
        batch_dict['positive_pairs_duped'] = [(2*n-1) * 2*n for n in batch_dict['lbl_inst_freq']]
        batch_dict['negative_pairs_duped'] = [sum(batch_dict['lbl_inst_freq']) - k  for k in batch_dict['lbl_inst_freq']]
        batch_dict['negative_pairs_duped'] = [4*k*i for k,i in zip(batch_dict['negative_pairs_duped'],batch_dict['lbl_inst_freq'])]


        combined_embed_features = torch.cat([embed_ft_sa.unsqueeze(1), embed_ft_wa.unsqueeze(1)], dim=1) # B*N,num_pairs,channel_dim
        num_pairs = combined_embed_features.shape[1]
        assert num_pairs == 2  # contrast_count = 2

        '''Create Contrastive Mask'''
        labels_sa = labels_sa.contiguous().view(-1, 1)
        mask = torch.eq(labels_sa, labels_sa.T).float().to(device) # (B*N, B*N)
        mask = mask.repeat(num_pairs, num_pairs)        # Tiling mask from N,N -> 2N, 2N)
        logits_mask = torch.scatter(torch.ones_like(mask),1,torch.arange(batch_size_labeled * num_pairs).view(-1, 1).to(device),0)    # mask-out self-contrast cases
        mask = mask * logits_mask

        '''
        plt.imshow(mask.cpu().numpy(), cmap='viridis')
        plt.colorbar()
        plt.savefig("mask.png")
        plt.clf()
        '''
        contrast_feature = torch.cat(torch.unbind(combined_embed_features, dim=1), dim=0) 
        contrast_feature = F.normalize(contrast_feature.view(-1,combined_embed_features.shape[-1])) # normalized features before masking. original code does it earlier : https://github.com/HobbitLong/SupContrast/blob/ae5da977b0abd4bdc1a6fd4ec4ba2c3655a1879f/networks/resnet_big.py#L185C51-L185C51
        # compute logits
        anchor_dot_contrast = torch.div(torch.matmul(contrast_feature, contrast_feature.T),temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        exp_logits = torch.exp(logits) * logits_mask # compute log_prob
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)         # compute mean of log-likelihood over positive

        #unscaled_loss
        unscaled_instloss_car = mean_log_prob_pos[[labels==1]].mean()
        unscaled_instloss_ped = mean_log_prob_pos[[labels==2]].mean()
        unscaled_instloss_cyc = mean_log_prob_pos[[labels==3]].mean()

        # batch_dict['unscaled_instloss_car'] = unscaled_instloss_car.unsqueeze(-1)
        # batch_dict['unscaled_instloss_ped'] = unscaled_instloss_ped.unsqueeze(-1)
        # batch_dict['unscaled_instloss_cyc'] = unscaled_instloss_cyc.unsqueeze(-1)

        instance_loss = - ( temperature/ base_temperature) * mean_log_prob_pos # base_temperature only scales the loss, temperature sharpens / smoothes the loss
        #scaled_loss
        instloss_car = instance_loss[labels==1].mean()
        instloss_ped = instance_loss[labels==2].mean()
        instloss_cyc = instance_loss[labels==3].mean()
        instloss_all = instance_loss.mean()

        # batch_dict['instloss_car'] = instloss_car.unsqueeze(-1).tolist()
        # batch_dict['instloss_ped'] = instloss_ped.unsqueeze(-1).tolist()
        # batch_dict['instloss_cyc'] = instloss_cyc.unsqueeze(-1).tolist()

        inst_tb_dict = {
            'unscaled_instloss_car': unscaled_instloss_car.unsqueeze(-1),
            'unscaled_instloss_ped': unscaled_instloss_ped.unsqueeze(-1),
            'unscaled_instloss_cyc': unscaled_instloss_cyc.unsqueeze(-1),
            'instloss_car': instloss_car.unsqueeze(-1),
            'instloss_cyc': instloss_cyc.unsqueeze(-1),
            'instloss_ped': instloss_ped.unsqueeze(-1),
            'instloss_all' : instloss_all.unsqueeze(-1),
        }
        tb_dict.update(inst_tb_dict)

        if instance_loss is None:
            return
        instance_loss = instance_loss.mean()
        return instance_loss, tb_dict


    @staticmethod
    def _prep_tb_dict(tb_dict, lbl_inds, ulb_inds, reduce_loss_fn):
        tb_dict_ = {}
        ignore_keys = ['proto_cont_loss','instance_cont_loss','classwise_instance_cont_loss_Car','classwise_instance_cont_loss_Pedestrian','classwise_instance_cont_loss_Cyclist',
                        'mCont_loss','mCont_Car_lb','mCont_Pedestrian_lb','mCont_Cyclist_lb','mCont_loss_instance','mCont_Car_lb_inst','mCont_Pedestrian_lb_inst','mCont_Cyclist_lb_inst',
                        'classwise_proto_sim_loss_Car','classwise_proto_sim_loss_Pedestrian','classwise_proto_sim_loss_Cyclist','proto_sim_loss']
        for key in tb_dict.keys():
            if key == 'proto_cont_loss':
                tb_dict_[key] = tb_dict[key]
            elif "instloss" in key:
                tb_dict_[key] = tb_dict[key]
            elif key in ignore_keys:
                tb_dict_[key] = tb_dict[key]            
            elif 'loss' in key or 'acc' in key or 'point_pos_num' in key:
                tb_dict_[f"{key}_labeled"] = reduce_loss_fn(tb_dict[key][lbl_inds, ...])
                tb_dict_[f"{key}_unlabeled"] = reduce_loss_fn(tb_dict[key][ulb_inds, ...])
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

    def dump_statistics(self, batch_dict, unlabeled_inds):
        # Store different types of scores over all itrs and epochs and dump them in a pickle for offline modeling
        # TODO (shashank) : Can be optimized later to save computational time, currently takes about 0.002sec
        batch_roi_labels = self.pv_rcnn.roi_head.forward_ret_dict['roi_labels'][unlabeled_inds]
        batch_roi_labels = [roi_labels.clone().detach() for roi_labels in batch_roi_labels]

        batch_rois = self.pv_rcnn.roi_head.forward_ret_dict['rois'][unlabeled_inds]
        batch_rois = [rois.clone().detach() for rois in batch_rois]

        batch_ori_gt_boxes = self.pv_rcnn.roi_head.forward_ret_dict['ori_unlabeled_boxes']
        batch_ori_gt_boxes = [ori_gt_boxes.clone().detach() for ori_gt_boxes in batch_ori_gt_boxes]

        gt_labels = batch_dict['gt_boxes'][:,:,7][labeled_inds].view(-1)
        

        batch_ori_gt_boxes_ema = batch_dict['ori_unlabeled_boxes_ema']
        batch_ori_gt_boxes_ema = [ori_gt_boxes_ema.clone().detach() for ori_gt_boxes_ema in batch_ori_gt_boxes_ema]

        batch_pls = batch_dict_ema['prefilter_pls']['gt_boxes_ema'][unlabeled_inds]
        batch_pls = [pls.clone().detach() for pls in batch_pls]
        for i in range(len(batch_pls)):
            valid_pl_boxes_mask = torch.logical_not(torch.all(batch_pls[i] == 0, dim=-1))
            valid_pls = batch_pls[i][valid_pl_boxes_mask]
            valid_pl_labels = batch_pls[i][valid_pl_boxes_mask][:, -1].int() 

            valid_gt_boxes_pl_mask = torch.logical_not(torch.all(batch_ori_gt_boxes_ema[i] == 0, dim=-1))
            valid_gt_boxes_pl = batch_ori_gt_boxes_ema[i][valid_gt_boxes_pl_mask]
            valid_gt_pl_labels = batch_ori_gt_boxes_ema[i][valid_gt_boxes_pl_mask][:, -1].int()
            num_pls = valid_pl_boxes_mask.sum()
            num_gt_pls = valid_gt_boxes_pl_mask.sum()
            cur_unlabeled_ind = unlabeled_inds[i]

            if num_pls > 0 and num_gt_pls > 0:
                # Find IoU between Student's PL v/s Teacher's GTs
                overlap = iou3d_nms_utils.boxes_iou3d_gpu(valid_pls[:, 0:7], valid_gt_boxes_pl[:, 0:7])
                pls_iou_max, assigned_gt_inds = overlap.max(dim=1)
                self.val_dict['iou_pl_gt'].extend(pls_iou_max.tolist())
                self.val_dict['assigned_gt_pl_labels'].extend(valid_gt_pl_labels[assigned_gt_inds].tolist())

                assert batch_dict_ema['prefilter_pls']['rcnn_scores_ema_prefilter'][cur_unlabeled_ind].shape[0] == valid_pl_boxes_mask.shape[0]
                assert batch_dict_ema['prefilter_pls']['pseudo_sem_scores_multiclass'][cur_unlabeled_ind].shape[0] == valid_pl_boxes_mask.shape[0]
                assert batch_dict_ema['prefilter_pls']['pseudo_sim_scores_emas_prefilter'][cur_unlabeled_ind].shape[0] == valid_pl_boxes_mask.shape[0]
                
                self.val_dict['rcnn_scores_pl'].extend((batch_dict_ema['prefilter_pls']['rcnn_scores_ema_prefilter'][cur_unlabeled_ind][valid_pl_boxes_mask]).tolist())
                self.val_dict['pseudo_sem_scores_pl'].extend((batch_dict_ema['prefilter_pls']['pseudo_sem_scores_multiclass'][cur_unlabeled_ind][valid_pl_boxes_mask]).tolist())
                self.val_dict['pseudo_sim_scores_pl'].extend((batch_dict_ema['prefilter_pls']['pseudo_sim_scores_emas_prefilter'][cur_unlabeled_ind][valid_pl_boxes_mask]).tolist())
                self.val_dict['pl_iteration'].extend((torch.ones_like(pls_iou_max) * batch_dict['cur_iteration']).tolist())
                self.val_dict['pseudo_instance_sim_scores_pl'].extend((batch_dict_ema['prefilter_pls']['pseudo_instance_sim_scores_emas_prefilter'][cur_unlabeled_ind][valid_pl_boxes_mask]).tolist())  

        for i in range(len(batch_rois)):
            valid_rois_mask = torch.logical_not(torch.all(batch_rois[i] == 0, dim=-1))
            valid_rois = batch_rois[i][valid_rois_mask]
            valid_roi_labels = batch_roi_labels[i][valid_rois_mask]
            valid_roi_labels -= 1  # Starting class indices from zero

            valid_gt_boxes_mask = torch.logical_not(torch.all(batch_ori_gt_boxes[i] == 0, dim=-1))
            valid_gt_boxes = batch_ori_gt_boxes[i][valid_gt_boxes_mask]
            valid_gt_boxes[:, -1] -= 1  # Starting class indices from zero
            num_gts = valid_gt_boxes_mask.sum()
            num_preds = valid_rois_mask.sum()

            cur_unlabeled_ind = unlabeled_inds[i]
            if num_gts > 0 and num_preds > 0:
                # Find IoU between Student's ROI v/s Original GTs
                overlap = iou3d_nms_utils.boxes_iou3d_gpu(valid_rois[:, 0:7], valid_gt_boxes[:, 0:7])
                preds_iou_max, assigned_gt_inds = overlap.max(dim=1)
                self.val_dict['iou_roi_gt'].extend(preds_iou_max.tolist())

                cur_iou_roi_pl = self.pv_rcnn.roi_head.forward_ret_dict['gt_iou_of_rois'][cur_unlabeled_ind]
                self.val_dict['iou_roi_pl'].extend(cur_iou_roi_pl.tolist())

                cur_pred_score = torch.sigmoid(batch_dict['batch_cls_preds'][cur_unlabeled_ind]).squeeze()
                self.val_dict['pred_scores'].extend(cur_pred_score.tolist())

                # if 'rcnn_cls_score_teacher' in self.pv_rcnn.roi_head.forward_ret_dict:
                #     cur_teacher_pred_score = self.pv_rcnn.roi_head.forward_ret_dict['rcnn_cls_score_teacher'][
                #         cur_unlabeled_ind]
                #     self.val_dict['teacher_pred_scores'].extend(cur_teacher_pred_score.tolist())

                #     cur_weight = self.pv_rcnn.roi_head.forward_ret_dict['rcnn_cls_weights'][cur_unlabeled_ind]
                #     self.val_dict['weights'].extend(cur_weight.tolist())

                cur_roi_score = torch.sigmoid(self.pv_rcnn.roi_head.forward_ret_dict['roi_scores'][cur_unlabeled_ind])
                self.val_dict['roi_scores'].extend(cur_roi_score.tolist())
                cur_instance_sim_score = self.pv_rcnn.roi_head.forward_ret_dict['roi_instance_sim_scores'][cur_unlabeled_ind]
                self.val_dict['roi_instance_sim_scores'].extend(cur_instance_sim_score.tolist())
                
                # cur_pcv_score = self.pv_rcnn.roi_head.forward_ret_dict['pcv_scores'][cur_unlabeled_ind]
                # self.val_dict['pcv_scores'].extend(cur_pcv_score.tolist())

                # # cur_num_points_roi = self.pv_rcnn.roi_head.forward_ret_dict['num_points_in_roi'][cur_unlabeled_ind]
                # # self.val_dict['num_points_in_roi'].extend(cur_num_points_roi.tolist())

                cur_roi_label = self.pv_rcnn.roi_head.forward_ret_dict['roi_labels'][cur_unlabeled_ind].squeeze()
                self.val_dict['class_labels'].extend(cur_roi_label.tolist())

                cur_iteration = torch.ones_like(preds_iou_max) * (batch_dict['cur_iteration'])
                self.val_dict['iteration'].extend(cur_iteration.tolist())

                start_epoch = self.model_cfg['ROI_HEAD'].get('INSTANCE_CONTRASTIVE_LOSS_START_EPOCH', 0)
                stop_epoch = self.model_cfg['ROI_HEAD'].get('INSTANCE_CONTRASTIVE_LOSS_STOP_EPOCH', 60)
                if self.model_cfg['ROI_HEAD']['ENABLE_INSTANCE_SUP_LOSS'] and start_epoch<=batch_dict['cur_epoch']<stop_epoch:
                    bincount_values = batch_dict['lbl_inst_freq']
                    # cumu_values = [a + b for a, b in zip(self.val_dict['lbl_inst_freq'][-3:], bincount_values)]
                    self.val_dict['lbl_inst_freq'].extend(bincount_values)
                    self.val_dict['positive_pairs_duped'].extend(batch_dict['positive_pairs_duped'])
                    self.val_dict['negative_pairs_duped'].extend(batch_dict['negative_pairs_duped'])

                    # self.val_dict['unscaled_instloss_car'].extend(batch_dict['unscaled_instloss_car'])
                    # self.val_dict['unscaled_instloss_ped'].extend(batch_dict['unscaled_instloss_ped'])
                    # self.val_dict['unscaled_instloss_cyc'].extend(batch_dict['unscaled_instloss_cyc'])

                    # self.val_dict['instloss_car'].extend(batch_dict['instloss_car'])
                    # self.val_dict['instloss_ped'].extend(batch_dict['instloss_ped'])
                    # self.val_dict['instloss_cyc'].extend(batch_dict['instloss_cyc'])

        # replace old pickle data (if exists) with updated one
        # if (batch_dict['cur_epoch']) == batch_dict['total_epochs']:
        output_dir = os.path.split(os.path.abspath(batch_dict['ckpt_save_dir']))[0]
        file_path = os.path.join(output_dir, 'scores.pkl')
        pickle.dump(self.val_dict, open(file_path, 'wb'))

    @staticmethod
    def get_pl_metrics_input(rois, sem_scores, weights, filtering_masks, gts):
        metrics_input = defaultdict(list)
        bs = len(rois)
        for i in range(bs):
            smpl_mask = filtering_masks[i]
            if smpl_mask.sum() == 0:
                # TODO: fix stats for no preds case
                smpl_rois = gts[i].new_zeros((1, 8)).float().to(gts[i].device)  # dummy rois
                smpl_scores = gts[i].new_zeros((1, 3)).float().to(gts[i].device)
                smpl_weights = gts[i].new_zeros((1, 1)).float().to(gts[i].device)
            else:
                smpl_rois = rois[i][smpl_mask].clone().detach()
                smpl_scores = sem_scores[i][smpl_mask].clone().detach()
                smpl_weights = weights[i][smpl_mask].clone().detach()

            metrics_input['rois'].append(smpl_rois)
            metrics_input['roi_scores'].append(smpl_scores)
            metrics_input['ground_truths'].append(gts[i].clone().detach())
            metrics_input['roi_weights'].append(smpl_weights)
        return metrics_input

    @staticmethod
    def _unpack_preds(pred_dicts, ulb_inds):
        pseudo_boxes = []
        pseudo_scores = []
        pseudo_sem_scores = []
        pseudo_sem_scores_logits_list = []
        pseudo_labels = []
        for ind in ulb_inds:
            pseudo_score = pred_dicts[ind]['pred_scores']
            pseudo_box = pred_dicts[ind]['pred_boxes']
            pseudo_label = pred_dicts[ind]['pred_labels']
            pseudo_sem_score = pred_dicts[ind]['pred_sem_scores']
            pseudo_sem_score_logits = pred_dicts[ind]['pred_sem_scores_logits']
            if len(pseudo_label) == 0:
                pseudo_boxes.append(pseudo_label.new_zeros((1, 7)).float())
                pseudo_sem_scores.append(pseudo_label.new_zeros((1,)).float())
                pseudo_sem_scores_logits_list.append(pseudo_label.new_zeros((1, 3)).float())
                pseudo_scores.append(pseudo_label.new_zeros((1,)).float())
                pseudo_labels.append(pseudo_label.new_zeros((1,)).float())
                pseudo_sem_scores_multiclass.append(pseudo_label.new_zeros((1,3)).float())
                pseudo_sim_scores.append(pseudo_label.new_zeros((1,3)).float())
                pseudo_instance_sim_scores.append(pseudo_label.new_zeros((1,3)).float())
                continue

            pseudo_boxes.append(pseudo_box)
            pseudo_sem_scores.append(pseudo_sem_score)
            pseudo_scores.append(pseudo_score)
            pseudo_labels.append(pseudo_label)
            pseudo_sem_scores_logits_list.append(pseudo_sem_score_logits)

        return pseudo_boxes, pseudo_labels, pseudo_scores, pseudo_sem_scores, pseudo_sem_scores_logits_list

    def _filter_pls_conf_scores(self, pls_dict, ulb_inds):
        pseudo_boxes = []
        pseudo_scores = []
        pseudo_sem_scores = []
        sem_scores_logits_list = []
        for boxs, labels, scores, sem_scores, sem_scores_logits in zip(*self._unpack_preds(pls_dict, ulb_inds)):
            if torch.eq(boxs, 0).all():
                pseudo_boxes.append(torch.cat([boxs, labels.view(-1, 1).float()], dim=1))
                pseudo_scores.append(scores)
                pseudo_sem_scores.append(sem_scores)
                sem_scores_logits_list.append(sem_scores_logits)
            else:
                conf_thresh = torch.tensor(self.thresh, device=labels.device).expand_as(sem_scores_logits).gather(dim=1, index=(labels - 1).unsqueeze(-1)).squeeze()
                mask = scores > conf_thresh
                pseudo_boxes.append(torch.cat([boxs[mask], labels[mask].view(-1, 1).float()], dim=1))
                pseudo_sem_scores.append(sem_scores[mask])
                sem_scores_logits_list.append(sem_scores_logits[mask])
                pseudo_scores.append(scores[mask])
        return pseudo_boxes, pseudo_scores, pseudo_sem_scores, sem_scores_logits_list
    def _filter_pls_sem_scores(self, batch_conf_scores, batch_sem_logtis):
        masks = []
        scores_list = []
        for conf_scores, sem_logits in zip(batch_conf_scores, batch_sem_logtis):
            if self.thresh_alg:
                mask, scores = self.thresh_alg.get_mask(conf_scores, sem_logits)
            else:  # 3dioumatch baseline
                labels = torch.argmax(sem_logits, dim=1)
                scores = torch.sigmoid(sem_logits)
                max_scores = scores.max(dim=-1)[0]
                sem_conf_thresh = torch.tensor(self.sem_thresh, device=sem_logits.device).unsqueeze(0).expand_as(sem_logits).gather(dim=1, index=labels.unsqueeze(-1)).squeeze()
                mask = max_scores > sem_conf_thresh

            masks.append(mask)
            scores_list.append(scores)
        return masks, scores_list

    @staticmethod
    def _fill_with_pseudo_labels(batch_dict, pseudo_boxes, unlabeled_inds, labeled_inds, key=None):
        key = 'gt_boxes' if key is None else key
        max_box_num = batch_dict['gt_boxes'].shape[1]

        # Expand the gt_boxes to have the same shape as the pseudo_boxes
        gt_scores = torch.zeros((batch_dict['gt_boxes'].shape[0], max_box_num, 1), device=batch_dict['gt_boxes'].device)
        batch_dict['gt_boxes'] = torch.cat([batch_dict['gt_boxes'], gt_scores], dim=-1)
        # Make sure that scores of labeled boxes are always 1, except for the padding rows which should remain zero.
        valid_inds_lbl = torch.logical_not(torch.eq(batch_dict['gt_boxes'][labeled_inds], 0).all(dim=-1)).nonzero().long()
        batch_dict['gt_boxes'][valid_inds_lbl[:, 0], valid_inds_lbl[:, 1], 8] = 1

        # Ignore the count of pseudo boxes if filled with default values(zeros) when no preds are made
        max_pseudo_box_num = max(
            [torch.logical_not(torch.all(ps_box == 0, dim=-1)).sum().item() for ps_box in pseudo_boxes])

        if max_box_num >= max_pseudo_box_num:
            for i, pseudo_box in enumerate(pseudo_boxes):
                diff = max_box_num - pseudo_box.shape[0]
                if diff > 0:
                    pseudo_box = torch.cat([pseudo_box, torch.zeros((diff, pseudo_box.shape[-1]), device=pseudo_box.device)], dim=0)
                batch_dict[key][unlabeled_inds[i]] = pseudo_box
        else:
            ori_boxes = batch_dict['gt_boxes']
            ori_ins_ids = batch_dict['instance_idx']
            new_boxes = torch.zeros((ori_boxes.shape[0], max_pseudo_box_num, ori_boxes.shape[-1]), device=ori_boxes.device)
            new_ins_idx = torch.full((ori_boxes.shape[0], max_pseudo_box_num), fill_value=-1, device=ori_boxes.device)
            for idx in labeled_inds:
                diff = max_pseudo_box_num - ori_boxes[idx].shape[0]
                new_box = torch.cat([ori_boxes[idx], torch.zeros((diff, ori_boxes.shape[-1]), device=ori_boxes[idx].device)], dim=0)
                new_boxes[idx] = new_box
                new_ins_idx[idx] = torch.cat([ori_ins_ids[idx], -torch.ones((diff,), device=ori_boxes[idx].device)], dim=0)
            for i, pseudo_box in enumerate(pseudo_boxes):

                diff = max_pseudo_box_num - pseudo_box.shape[0]
                if diff > 0:
                    pseudo_box = torch.cat([pseudo_box, torch.zeros((diff, pseudo_box.shape[-1]), device=pseudo_box.device)], dim=0)
                new_boxes[unlabeled_inds[i]] = pseudo_box
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
            # TODO(farzad) check this
            ema_param.data.mul_(alpha).add_((1 - alpha) * param.data)
        self.accumulated_itr = 0

    def load_params_from_file(self, filename, logger, to_cpu=False):
        if not os.path.isfile(filename):
            raise FileNotFoundError

        logger.info('==> Loading parameters from checkpoint %s to %s' % (filename, 'CPU' if to_cpu else 'GPU'))
        loc_type = torch.device('cpu') if to_cpu else None
        checkpoint = torch.load(filename, map_location=loc_type)
        model_state_disk = checkpoint['model_state']

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

        state_dict = self.state_dict()
        state_dict.update(update_model_state)
        self.load_state_dict(state_dict)

        for key in state_dict:
            if key not in update_model_state:
                logger.info('Not updated weight %s: %s' % (key, str(state_dict[key].shape)))

        logger.info('==> Done (loaded %d/%d)' % (len(update_model_state), len(self.state_dict())))
# #gt_cls_count = torch.bincount(batch_dict['ori_unlabeled_boxes'][...,-1].view(-1).int().detach()).tolist()[1:]
# #labels
# #


#         # pl_count_dict = {
#         #     f'pl_iter_count_{cls}': {
#         #         'gt': gt_cls_count[cind],
#         #         'pl_pre_filter': pl_cls_count_pre_filter[cind],
#         #         'pl_post_filter': pl_cls_count_post_filter[cind],
#         #     }
#         #     for cind, cls in enumerate(self.class_names)
#         # }

#         # tb_dict_ = self._prep_tb_dict(tb_dict, lbl_inds, ulb_inds, reduce_loss_fn)
#         # tb_dict_.update(**pl_count_dict)

#     def update_metrics_pred(self, targets_dict,pseudo_labels,mask_type='cls',bank=None):
#         pseudo_boxes, pseudo_labels, pseudo_score, pseudo_sem_score, pseudo_box_var, pseudo_score_var,pseudo_sem_score_multiclass,pseudo_sim_score,pseudo_instance_sim_score = self._unpack_predictions(pseudo_labels, targets_dict['unlabeled_inds'])
#         pseudo_boxes = [torch.cat([pseudo_box, pseudo_label.view(-1, 1).float()], dim=1) \
#                             for (pseudo_box, pseudo_label) in zip(pseudo_boxes, pseudo_labels)] # add label to boxes
#         # pseudo_sem_scores_multiclass = [pseudo_sem_score_multiclass]
#         # pseudo_sim_scores = torch.cat(pseudo_sim_score, dim=0).unsqueeze(0)
#         self._fill_with_pseudo_labels_prefilter(targets_dict, pseudo_boxes, pseudo_sem_score_multiclass, pseudo_sim_score, pseudo_score,pseudo_instance_sim_score,targets_dict['unlabeled_inds'], targets_dict['labeled_inds']) #TODO: check if this is correct
#         targets_dict['gt_boxes_emas_prefilter'] = targets_dict['gt_boxes'].clone()
#         targets_dict['pseudo_sem_scores_multiclass_emas_prefilter'] = targets_dict['pseudo_sem_scores_multiclass'].clone()
#         targets_dict['pseudo_sim_scores_emas_prefilter'] = targets_dict['pseudo_sim_scores'].clone()
#         targets_dict['rcnn_scores_ema_prefilter'] = targets_dict['pseudo_scores'].clone()
#         targets_dict['pseudo_instance_sim_scores_emas_prefilter'] = targets_dict['pseudo_instance_sim_scores'].clone()

#         self.apply_augmentation(targets_dict, targets_dict, targets_dict['unlabeled_inds'], key='gt_boxes')
#         metrics_input = defaultdict(list)
#         for i, uind in enumerate(targets_dict['unlabeled_inds']):
#             # mask = (targets_dict['reg_valid_mask'][uind] > 0) if mask_type == 'reg' else (
#             #             targets_dict['rcnn_cls_labels'][uind] >= 0)
#             # if mask.sum() == 0:
#             #     # print(f'Warning: No {mask_type} rois for unlabeled index {uind}')
#             #     continue

#             # (Proposals) PLs are passed in as ROIs
#             rois = targets_dict['gt_boxes'][uind].detach().clone()
#             roi_labels = targets_dict['gt_boxes'][...,-1][uind].unsqueeze(-1).clone().detach()
#             roi_scores_multiclass = targets_dict['pseudo_sem_scores_multiclass'][uind].clone().detach()
#             roi_sim_scores_multiclass = targets_dict['pseudo_sim_scores'][uind].clone().detach()
#             roi_instance_sim_scores_multiclass = targets_dict['pseudo_instance_sim_scores'][uind].detach().clone()
#             metrics_input['roi_instance_sim_scores'].append(roi_instance_sim_scores_multiclass)
#             metrics_input['rois'].append(rois)
#             metrics_input['roi_scores'].append(roi_scores_multiclass)
#             metrics_input['roi_sim_scores'].append(roi_sim_scores_multiclass)

#             # (Real labels) GT info: Original GTs are passed in as GTs
#             gt_labeled_boxes = targets_dict['ori_unlabeled_boxes'][i]
#             metrics_input['ground_truths'].append(gt_labeled_boxes)
#             metrics_input['roi_weights'] = None
#             metrics_input['roi_iou_wrt_pl'] = None
#             metrics_input['roi_target_scores'] = None

#             bs_id = targets_dict['points'][:, 0] == uind
#             points = targets_dict['points'][bs_id, 1:].detach().clone()
#             metrics_input['points'].append(points)
#         if len(metrics_input['rois']) == 0:
#             # print(f'Warning: No {mask_type} rois for any unlabeled index')
#             return
#         tag = f'pl_gt_metrics_before_filtering_{mask_type}'
#         metrics_registry.get(tag).update(**metrics_input)
#         return {
#             'gt_boxes_ema': targets_dict['gt_boxes_emas_prefilter'],
#             'pseudo_sem_scores_multiclass': targets_dict['pseudo_sem_scores_multiclass_emas_prefilter'],
#             'pseudo_sim_scores_emas_prefilter': targets_dict['pseudo_sim_scores_emas_prefilter'],
#             'pseudo_instance_sim_scores_emas_prefilter': targets_dict['pseudo_instance_sim_scores_emas_prefilter'],
#             'rcnn_scores_ema_prefilter' : targets_dict['rcnn_scores_ema_prefilter']
#         }

#     def update_metrics_pl(self,targets_dict, mask_type='cls'):
#         metrics_input = defaultdict(list)
#         for i, uind in enumerate(targets_dict['unlabeled_inds']):
#             # mask = (targets_dict['reg_valid_mask'][uind] > 0) if mask_type == 'reg' else (
#             #             targets_dict['rcnn_cls_labels'][uind] >= 0)
#             # if mask.sum() == 0:
#             #     # print(f'Warning: No {mask_type} rois for unlabeled index {uind}')
#             #     continue

#             # (Proposals) ROI info
#             rois = targets_dict['gt_boxes'][uind].detach().clone()
#             roi_labels = targets_dict['gt_boxes'][...,-1][uind].unsqueeze(-1).detach().clone()
#             roi_scores_multiclass = targets_dict['pseudo_sem_scores_multiclass'][uind].detach().clone()
#             roi_instance_sim_scores_multiclass = targets_dict['pseudo_instance_sim_scores'][uind].detach().clone()
#             roi_sim_scores_multiclass = targets_dict['pseudo_sim_scores'][uind].detach().clone()
#             roi_instance_sim_scores_multiclass = targets_dict['pseudo_instance_sim_scores'][uind].detach().clone()
#             metrics_input['rois'].append(rois)
#             metrics_input['roi_scores'].append(roi_scores_multiclass)
#             metrics_input['roi_sim_scores'].append(roi_sim_scores_multiclass)
#             metrics_input['roi_instance_sim_scores'].append(roi_instance_sim_scores_multiclass)

#             # (Real labels) GT info
#             gt_labeled_boxes = targets_dict['ori_unlabeled_boxes'][i]
#             metrics_input['ground_truths'].append(gt_labeled_boxes)
#             metrics_input['roi_weights'] = None
#             metrics_input['roi_iou_wrt_pl'] = None
#             metrics_input['roi_target_scores'] = None

#             bs_id = targets_dict['points'][:, 0] == uind
#             points = targets_dict['points'][bs_id, 1:].clone().detach()
#             metrics_input['points'].append(points)
#         if len(metrics_input['rois']) == 0:
#             # print(f'Warning: No {mask_type} rois for any unlabeled index')
#             return
#         tag = f'pl_gt_metrics_after_filtering_{mask_type}'
#         metrics_registry.get(tag).update(**metrics_input)
        