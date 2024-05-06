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
        vals_to_store = ['iou_roi_pl', 'iou_roi_gt', 'obj_scores','gt_boxes','assigned_gt_inds','assigned_iou_class',
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
            if self.model_cfg['ROI_HEAD'].get('ENABLE_INSTANCE_SUP_LOSS', False):
                batch_dict_aug = self.apply_augmentation(batch_dict, batch_dict, labeled_inds, key='gt_boxes')
                with torch.no_grad():
                    for cur_module in self.module_list:
                        batch_dict_aug = cur_module(batch_dict_aug)
                lbl_inst_cont_loss, tb_dict = self._get_instance_contrastive_loss(tb_dict,batch_dict,batch_dict_aug,labeled_inds)

        
            loss, tb_dict, disp_dict = self.get_training_loss(batch_dict)
            if self.model_cfg['ROI_HEAD'].get('ENABLE_INSTANCE_SUP_LOSS', False) and lbl_inst_cont_loss is not None:
                loss +=  lbl_inst_cont_loss * self.model_cfg['ROI_HEAD']['INSTANCE_CONTRASTIVE_LOSS_WEIGHT']

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

    def _get_instance_contrastive_loss(self, tb_dict,batch_dict,batch_dict_aug,lbl_inds,temperature=1.0,base_temperature=1.0):
        '''
        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: roi_labels[B,N].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        '''
        # start_epoch = self.model_cfg['ROI_HEAD'].get('INSTANCE_CONTRASTIVE_LOSS_START_EPOCH', 0)
        # stop_epoch = self.model_cfg['ROI_HEAD'].get('INSTANCE_CONTRASTIVE_LOSS_STOP_EPOCH', 60)
        tb_dict = {} if tb_dict is None else tb_dict
        # # To examine effects of stopping supervised contrastive loss
        # if not start_epoch<=batch_dict['cur_epoch']<stop_epoch:
        #     return
        temperature = self.model_cfg['ROI_HEAD'].get('TEMPERATURE', 1.0)

        labels_sa, labels_wa, instance_idx_sa, instance_idx_wa, embed_ft_sa, embed_ft_wa = self._align_instance_pairs(batch_dict, batch_dict_aug,lbl_inds)
        batch_size_labeled = labels_sa.shape[0]
        device = embed_ft_sa.device
        labels = torch.cat((labels_sa,labels_sa), dim=0)

        assert torch.equal(instance_idx_sa, instance_idx_wa)
        assert torch.equal(labels_sa, labels_wa) 
        combined_embed_features = torch.cat([embed_ft_sa.unsqueeze(1), embed_ft_wa.unsqueeze(1)], dim=1) # B*N,num_pairs,channel_dim
        num_pairs = combined_embed_features.shape[1]
        assert num_pairs == 2  # contrast_count = 2

        '''Create Contrastive Mask'''
        labels_sa = labels_sa.contiguous().view(-1, 1)
        mask = torch.eq(labels_sa, labels_sa.T).float().to(device) # (B*N, B*N)
        mask = mask.repeat(num_pairs, num_pairs)        # Tiling mask from N,N -> 2N, 2N)
        logits_mask = torch.scatter(torch.ones_like(mask),1,torch.arange(batch_size_labeled * num_pairs).view(-1, 1).to(device),0)    # mask-out self-contrast cases
        mask = mask * logits_mask
        contrast_feature = torch.cat(torch.unbind(combined_embed_features, dim=1), dim=0) 
        contrast_feature = F.normalize(contrast_feature.view(-1,combined_embed_features.shape[-1])) # normalized features before masking. original code does it earlier : https://github.com/HobbitLong/SupContrast/blob/ae5da977b0abd4bdc1a6fd4ec4ba2c3655a1879f/networks/resnet_big.py#L185C51-L185C51
        anchor_dot_contrast = torch.div(torch.matmul(contrast_feature, contrast_feature.T),temperature)
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        exp_logits = torch.exp(logits) * logits_mask # compute log_prob
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)   
        
        #Role of temperature scaling in SupConLoss ... rescaling instance_loss by 0.1 to see results
        instance_loss = - ( temperature/ base_temperature) * mean_log_prob_pos # base_temperature only scales the loss, temperature sharpens / smoothes the loss
        #scaled_loss
        instloss_car = instance_loss[labels==1].mean()
        instloss_ped = instance_loss[labels==2].mean()
        instloss_cyc = instance_loss[labels==3].mean()
        instloss_all = instance_loss.mean()

        inst_tb_dict = {
            'inst_loss_car': instloss_car.unsqueeze(-1),
            'inst_loss_cyc': instloss_cyc.unsqueeze(-1),
            'inst_loss_ped': instloss_ped.unsqueeze(-1),
            'inst_loss_total' : instloss_all.unsqueeze(-1),
        }
        tb_dict.update(inst_tb_dict)

        if instance_loss is None:
            return
        instance_loss = instance_loss.mean()
        return instance_loss, tb_dict


    def _align_instance_pairs(self, batch_dict, batch_dict_aug, indices):
        
        embed_size = 256 if not self.model_cfg['ROI_HEAD']['PROJECTOR'] else 256 # if possible, 128
        shared_ft_sa = batch_dict['shared_features_gt'].view(batch_dict['batch_size'],-1,embed_size)[indices]
        shared_ft_wa = batch_dict_aug['shared_features_gt'].view(batch_dict['batch_size'],-1,embed_size)[indices]
        device = shared_ft_sa.device
        labels_sa = batch_dict['gt_boxes'][:,:,7][indices].view(-1)
        labels_wa = batch_dict_aug['gt_boxes'][:,:,7][indices].view(-1)
        instance_idx_sa = batch_dict['instance_idx'][indices].view(-1)
        instance_idx_wa = batch_dict_aug['instance_idx'][indices].view(-1)
        shared_ft_sa = shared_ft_sa.view(-1,embed_size)
        shared_ft_wa = shared_ft_wa.view(-1,embed_size)
        
        prefinal_mask_sa = labels_sa!=0
        prefinal_mask_wa = labels_wa!=0

        instance_idx_sa = instance_idx_sa[prefinal_mask_sa]
        instance_idx_wa = instance_idx_wa[prefinal_mask_wa]

        meta_data = {'to_mask':''}
        valid_instances = np.intersect1d(instance_idx_sa.cpu().numpy(),instance_idx_wa.cpu().numpy()) #
        valid_instances = torch.tensor(valid_instances,device=device)
       
        '''intersect_mask, to remove instances from A which are not in B and VICE VERSA '''
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

        '''Edge case - Handle more labeled indices in batch_dict_aug's dataloader batch than batch_dict's dataloader(or vice-versa)'''        
        final_mask = []
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
        
        sorted_sa = instance_idx_sa.sort()[-1].long()
        sorted_wa = instance_idx_wa.sort()[-1].long()

        instance_idx_sa = instance_idx_sa[sorted_sa]
        instance_idx_wa = instance_idx_wa[sorted_wa]

        labels_sa = labels_sa[sorted_sa]
        labels_wa =labels_wa[sorted_wa]
        shared_ft_sa = shared_ft_sa[sorted_sa]
        shared_ft_wa = shared_ft_wa[sorted_wa]

        return labels_sa,labels_wa,instance_idx_sa,instance_idx_wa,shared_ft_sa, shared_ft_wa


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
                # Find IoU between ROI v/s Original GTs
                overlap = iou3d_nms_utils.boxes_iou3d_gpu(valid_rois[:, 0:7], valid_gt_boxes[:, 0:7])
                preds_iou_max, assigned_gt_inds = overlap.max(dim=1)
                self.val_dict['iou_roi_gt'].extend(preds_iou_max.tolist())
                self.val_dict['assigned_gt_inds'].extend(assigned_gt_inds.tolist())
                assigned_iou_class = []
                for ind in assigned_gt_inds:
                    assigned_iou_class.append(valid_gt_boxes[ind][-1].cpu())
                self.val_dict['assigned_iou_class'].extend(assigned_iou_class)
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
        file_path = os.path.join(output_dir, 'new_ohnegtsmpl.pkl')
        pickle.dump(self.val_dict, open(file_path, 'wb'))

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
