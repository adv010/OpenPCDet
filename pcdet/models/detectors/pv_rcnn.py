from .detector3d_template import Detector3DTemplate
import os
import torch.distributed as dist 
import pickle

class PVRCNN(Detector3DTemplate):
    def __init__(self, model_cfg, num_class, dataset):
        super().__init__(model_cfg=model_cfg, num_class=num_class, dataset=dataset)
        self.module_list = self.build_networks()
        self.shared_pkl = {'ens': []}

    def forward(self, batch_dict):
        for cur_module in self.module_list:
            batch_dict = cur_module(batch_dict)

        temp = {}
        cur_gt_boxes = batch_dict['gt_boxes']
        #cur_shared_features_gt = batch_dict['shared_features_gt']
        cur_cls_preds = batch_dict['batch_cls_preds']
        k = cur_gt_boxes.__len__() - 1
        while k >= 0 and cur_gt_boxes[k].sum() == 0:
            k -= 1
        cur_gt_boxes = cur_gt_boxes[:k + 1]
        temp['pooled_features_gt'] = (batch_dict['pooled_features_gt'])[:k + 1].clone().detach().cpu().numpy()
        # temp['instance_idx'] = (batch_dict['instance_idx'])[:k + 1].clone().detach().cpu().numpy()
        temp['gt_classes'] = ((batch_dict['gt_boxes'])[..., -1].int())[:k + 1].clone().detach().cpu().numpy()
        temp['gt_boxes'] = cur_gt_boxes.clone().detach().cpu().numpy()
        self.shared_pkl['ens'].append(temp)

        output_dir = os.path.split(os.path.abspath(batch_dict['ckpt_save_dir']))[0]
        
        if dist.is_initialized():
            rank = os.getenv('RANK')
            file_path = os.path.join(output_dir, f'prototype_infos_fully_sup{rank}.pkl')
        else:
            file_path = os.path.join(output_dir, 'prototype_infos_fully_sup.pkl')
        pickle.dump(self.shared_pkl, open(file_path, 'wb'))

        if self.training:
            loss, tb_dict, disp_dict = self.get_training_loss()

            ret_dict = {
                'loss': loss
            }
            return ret_dict, tb_dict, disp_dict
        else:
            pred_dicts, recall_dicts = self.post_processing(batch_dict)
            return pred_dicts, recall_dicts, {}

    def get_training_loss(self):
        disp_dict = {}
        loss_rpn, tb_dict = self.dense_head.get_loss()
        loss_point, tb_dict = self.point_head.get_loss(tb_dict)
        loss_rcnn, tb_dict = self.roi_head.get_loss(tb_dict)

        loss = loss_rpn + loss_point + loss_rcnn
        return loss, tb_dict, disp_dict
