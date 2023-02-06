import numpy as np
import torch.nn as nn

from .anchor_head_template import AnchorHeadTemplate


class AnchorHeadSingle(AnchorHeadTemplate):
    def __init__(self, model_cfg, input_channels, num_class, class_names, grid_size, point_cloud_range,
                 predict_boxes_when_training=True, **kwargs):
        super().__init__(
            model_cfg=model_cfg, num_class=num_class, class_names=class_names, grid_size=grid_size, point_cloud_range=point_cloud_range,
            predict_boxes_when_training=predict_boxes_when_training
        )

        self.num_anchors_per_location = sum(self.num_anchors_per_location)

        self.conv_cls = nn.Conv2d(
            input_channels, self.num_anchors_per_location * self.num_class,
            kernel_size=1
        )
        self.conv_box = nn.Conv2d(
            input_channels, self.num_anchors_per_location * self.box_coder.code_size,
            kernel_size=1
        )

        if self.model_cfg.get('USE_DIRECTION_CLASSIFIER', None) is not None:
            self.conv_dir_cls = nn.Conv2d(
                input_channels,
                self.num_anchors_per_location * self.model_cfg.NUM_DIR_BINS,
                kernel_size=1
            )
        else:
            self.conv_dir_cls = None
        self.init_weights()

        self.print_loss_when_eval = False

    def init_weights(self):
        pi = 0.01
        nn.init.constant_(self.conv_cls.bias, -np.log((1 - pi) / pi))
        nn.init.normal_(self.conv_box.weight, mean=0, std=0.001)

    def forward(self, data_dict, disable_gt_roi_when_pseudo_labeling=False):
        spatial_features_2d = data_dict['spatial_features_2d']

        cls_preds = self.conv_cls(spatial_features_2d)
        box_preds = self.conv_box(spatial_features_2d)

        cls_preds = cls_preds.permute(0, 2, 3, 1).contiguous()  # [N, H, W, C] #torch.Size([2, 200, 176, 18])
        box_preds = box_preds.permute(0, 2, 3, 1).contiguous()  # [N, H, W, C] #torch.Size([2, 200, 176, 42])

        self.forward_ret_dict['cls_preds'] = cls_preds
        self.forward_ret_dict['box_preds'] = box_preds

        if self.conv_dir_cls is not None:
            dir_cls_preds = self.conv_dir_cls(spatial_features_2d)
            dir_cls_preds = dir_cls_preds.permute(0, 2, 3, 1).contiguous()
            self.forward_ret_dict['dir_cls_preds'] = dir_cls_preds
        else:
            dir_cls_preds = None


# Keep disable_gt_roi flag true during Teacher. During Student, we won't be disabling gt roi. We have GTs on account of PLs from teacher.
        '''
Before entering this code block, we have 70400 anchors of each class = 211200 anchors in total. But these anchors won't have any GTs to map to.
We create what are called targets for these anchors. Targets can be thought of as GTs modified and mapped at anchor-level , which are handled by target_assigner.
We map anchors to gts basis their IoU with GTs. For anchors with IoU > certain threshold, we set reg_weights = 1
We get a targets_dict which contains : (target class labels for each anchor, target boxes for each anchor, reg_weight for each anchor)

        '''
        if (self.training or self.print_loss_when_eval) and not disable_gt_roi_when_pseudo_labeling: # disable_gt_roi_when_pseudo_labeling flag is disabled during teacher run
            targets_dict = self.assign_targets(
                gt_boxes=data_dict['gt_boxes']
            )
            self.forward_ret_dict.update(targets_dict)

        if not self.training or self.predict_boxes_when_training:
            batch_cls_preds, batch_box_preds = self.generate_predicted_boxes(
                batch_size=data_dict['batch_size'],
                cls_preds=cls_preds, box_preds=box_preds, dir_cls_preds=dir_cls_preds
            )
            '''The function generate_predicted_boxes takes as input the predicted class probabilities (cls_preds), predicted bounding box coordinates (box_preds), and predicted orientations (dir_cls_preds) for a batch of data. The function returns the predicted class probabilities for each anchor (batch_cls_preds) and the decoded bounding box coordinates with orientation information for each anchor (batch_box_preds)'''
            data_dict['batch_cls_preds'] = batch_cls_preds #torch.Size([2, 211200, 3])
            data_dict['batch_box_preds'] = batch_box_preds #torch.Size([2, 211200, 7])
            data_dict['cls_preds_normalized'] = False

        return data_dict
