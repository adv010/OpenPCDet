import torch.nn.functional as F
import torch.nn as nn
import torch
from torch.nn import BatchNorm1d
import random
from ...ops.pointnet2.pointnet2_stack import pointnet2_modules as pointnet2_stack_modules
from ...utils import common_utils
from .roi_head_template import RoIHeadTemplate
from torch.nn.utils import weight_norm
import numpy as np
from sklearn.metrics import precision_score


class PVRCNNHead(RoIHeadTemplate):
    def __init__(self, input_channels, model_cfg, num_class=1,
                 predict_boxes_when_training=True, **kwargs):
        super().__init__(num_class=num_class, model_cfg=model_cfg,
                         predict_boxes_when_training=predict_boxes_when_training)
        self.model_cfg = model_cfg

        self.roi_grid_pool_layer, num_c_out = pointnet2_stack_modules.build_local_aggregation_module(
            input_channels=input_channels, config=self.model_cfg.ROI_GRID_POOL
        )

        GRID_SIZE = self.model_cfg.ROI_GRID_POOL.GRID_SIZE
        pre_channel = GRID_SIZE * GRID_SIZE * GRID_SIZE * num_c_out

        shared_fc_list = []
        for k in range(0, self.model_cfg.SHARED_FC.__len__()):
            shared_fc_list.extend([
                nn.Conv1d(pre_channel, self.model_cfg.SHARED_FC[k], kernel_size=1, bias=False),
                nn.BatchNorm1d(self.model_cfg.SHARED_FC[k]),
                nn.ReLU()
            ])
            pre_channel = self.model_cfg.SHARED_FC[k]

            if k != self.model_cfg.SHARED_FC.__len__() - 1 and self.model_cfg.DP_RATIO > 0:
                shared_fc_list.append(nn.Dropout(self.model_cfg.DP_RATIO))

        self.shared_fc_layer = nn.Sequential(*shared_fc_list)

        self.cls_layers = self.make_fc_layers(
            input_channels=pre_channel, output_channels=self.num_class, fc_list=self.model_cfg.CLS_FC
        )
        self.reg_layers = self.make_fc_layers(
            input_channels=pre_channel,
            output_channels=self.box_coder.code_size * self.num_class,
            fc_list=self.model_cfg.REG_FC
        )
        self.print_loss_when_eval = False
        self.init_weights(weight_init='xavier')

    def init_weights(self, weight_init='xavier'):
        if weight_init == 'kaiming':
            init_func = nn.init.kaiming_normal_
        elif weight_init == 'xavier':
            init_func = nn.init.xavier_normal_
        elif weight_init == 'normal':
            init_func = nn.init.normal_
        else:
            raise NotImplementedError

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv1d):
                if weight_init == 'normal':
                    init_func(m.weight, mean=0, std=0.001)
                else:
                    init_func(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        nn.init.normal_(self.reg_layers[-1].weight, mean=0, std=0.001)

    def roi_grid_pool(self, batch_dict):
        """
        Args:
            batch_dict:
                batch_size:
                rois: (B, num_rois, 7 + C)
                point_coords: (num_points, 4)  [bs_idx, x, y, z]
                point_features: (num_points, C)
                point_cls_scores: (N1 + N2 + N3 + ..., 1)
                point_part_offset: (N1 + N2 + N3 + ..., 3)
        Returns:

        """
        batch_size = batch_dict['batch_size']
        rois = batch_dict['rois']
        point_coords = batch_dict["point_coords"]
        point_features = batch_dict["point_features"]
        point_cls_scores = batch_dict["point_cls_scores"]

        point_features = point_features * point_cls_scores.view(-1, 1)

        global_roi_grid_points, local_roi_grid_points = self.get_global_grid_points_of_roi(
            rois, grid_size=self.model_cfg.ROI_GRID_POOL.GRID_SIZE
        )  # (BxN, 6x6x6, 3)
        global_roi_grid_points = global_roi_grid_points.view(batch_size, -1, 3)  # (B, Nx6x6x6, 3)

        xyz = point_coords[:, 1:4]
        xyz_batch_cnt = xyz.new_zeros(batch_size).int()
        batch_idx = point_coords[:, 0]
        for k in range(batch_size):
            xyz_batch_cnt[k] = (batch_idx == k).sum()

        new_xyz = global_roi_grid_points.view(-1, 3)
        new_xyz_batch_cnt = xyz.new_zeros(batch_size).int().fill_(global_roi_grid_points.shape[1])
        pooled_points, pooled_features = self.roi_grid_pool_layer(
            xyz=xyz.contiguous(),
            xyz_batch_cnt=xyz_batch_cnt,
            new_xyz=new_xyz,
            new_xyz_batch_cnt=new_xyz_batch_cnt,
            features=point_features.contiguous(),
        )  # (M1 + M2 ..., C)

        pooled_features = pooled_features.view(
            -1, self.model_cfg.ROI_GRID_POOL.GRID_SIZE ** 3,
            pooled_features.shape[-1]
        )  # (BxN, 6x6x6, C)
        return pooled_features

    def get_global_grid_points_of_roi(self, rois, grid_size):
        rois = rois.view(-1, rois.shape[-1])
        batch_size_rcnn = rois.shape[0]

        local_roi_grid_points = self.get_dense_grid_points(rois, batch_size_rcnn, grid_size)  # (B, 6x6x6, 3)
        global_roi_grid_points = common_utils.rotate_points_along_z(
            local_roi_grid_points.clone(), rois[:, 6]
        ).squeeze(dim=1)
        global_center = rois[:, 0:3].clone()
        global_roi_grid_points += global_center.unsqueeze(dim=1)
        return global_roi_grid_points, local_roi_grid_points

    @staticmethod
    def get_dense_grid_points(rois, batch_size_rcnn, grid_size):
        faked_features = rois.new_ones((grid_size, grid_size, grid_size))
        dense_idx = faked_features.nonzero()  # (N, 3) [x_idx, y_idx, z_idx]
        dense_idx = dense_idx.repeat(batch_size_rcnn, 1, 1).float()  # (B, 6x6x6, 3)

        local_roi_size = rois.view(batch_size_rcnn, -1)[:, 3:6]
        roi_grid_points = (dense_idx + 0.5) / grid_size * local_roi_size.unsqueeze(dim=1) \
                          - (local_roi_size.unsqueeze(dim=1) / 2)  # (B, 6x6x6, 3)
        return roi_grid_points

    def forward(self, batch_dict, test_only=False):
        """
        :param input_data: input dict
        :return:
        """

        nms_config = self.model_cfg.NMS_CONFIG['TRAIN' if self.training and not test_only else 'TEST']
        # proposal_layer doesn't continue if the rois are already in the batch_dict.
        # However, for labeled data proposal layer should continue!
        targets_dict = self.proposal_layer(batch_dict, nms_config=nms_config)
        # should not use gt_roi for pseudo label generation
        if (self.training or self.print_loss_when_eval) and not test_only:
            targets_dict = self.assign_targets(batch_dict)
            batch_dict['rois'] = targets_dict['rois']
            batch_dict['roi_labels'] = targets_dict['roi_labels']
            # batch_dict['roi_scores'] = targets_dict['roi_scores']
            # batch_dict['roi_scores_logits'] = targets_dict['roi_scores_logits']
            # batch_dict['rcnn_cls_labels'] = targets_dict['rcnn_cls_labels']
            # (batch_dict['rcnn_cls_labels'] == -1).any().item() and print('rcnn_cls_labels has -1')
            # batch_dict['gt_iou_of_rois'] = targets_dict['gt_iou_of_rois']

        pooled_features = self.roi_grid_pool(batch_dict)  # (BxN, 6x6x6, C)
        grid_size = self.model_cfg.ROI_GRID_POOL.GRID_SIZE
        batch_size_rcnn = pooled_features.shape[0]
        pooled_features = pooled_features.permute(0, 2, 1). \
            contiguous().view(batch_size_rcnn, -1, grid_size, grid_size, grid_size)  # (BxN, C, 6, 6, 6)

        batch_size_rcnn = pooled_features.shape[0]
        shared_features = self.shared_fc_layer(pooled_features.view(batch_size_rcnn, -1, 1))
        rcnn_cls = self.cls_layers(shared_features).transpose(1, 2).contiguous().squeeze(dim=1)  # (B, 1 or 2)
        rcnn_reg = self.reg_layers(shared_features).transpose(1, 2).contiguous().squeeze(dim=1)  # (B, C)

        if not self.training or self.predict_boxes_when_training:
            batch_cls_preds, batch_box_preds = self.generate_predicted_boxes(
                batch_size=batch_dict['batch_size'], rois=batch_dict['rois'], cls_preds=rcnn_cls, box_preds=rcnn_reg
            )
            # note that the rpn batch_cls_preds and batch_box_preds are being overridden here by rcnn preds
            batch_dict['batch_cls_preds'] = batch_cls_preds
            batch_dict['batch_box_preds'] = batch_box_preds
            batch_dict['cls_preds_normalized'] = False

        if self.training or self.print_loss_when_eval:
            targets_dict['rcnn_cls'] = rcnn_cls
            targets_dict['rcnn_reg'] = rcnn_reg
            self.forward_ret_dict = targets_dict

        return batch_dict


class PVRCNNHeadWithGridTransformer(PVRCNNHead):
    def __init__(self, input_channels, model_cfg, num_class=1, **kwargs):
        super().__init__(input_channels, model_cfg, num_class=num_class, **kwargs)
        self.grid_transformer_encoder = GridTransformerEncoder(
            feature_dim=self.model_cfg.GRID_TRANSFORMER_ENCODER.FEATURE_DIM,
            num_heads=self.model_cfg.GRID_TRANSFORMER_ENCODER.NUM_HEADS,
            num_layers=self.model_cfg.GRID_TRANSFORMER_ENCODER.NUM_LAYERS,
            hidden_dim=self.model_cfg.GRID_TRANSFORMER_ENCODER.HIDDEN_DIM
        )
        self.cls_layers = nn.Sequential(nn.Linear(256, 256),
                                        BatchNorm1d(256),
                                        nn.ReLU(),
                                        nn.Linear(256, num_class))
        self.reg_layers = nn.Sequential(nn.Linear(256, 256),
                                        BatchNorm1d(256),
                                        nn.ReLU(),
                                        nn.Linear(256, self.box_coder.code_size * num_class))
        self.print_loss_when_eval = False
        self.init_weights(weight_init='xavier')

    def forward(self, batch_dict, test_only=False):
        nms_config = self.model_cfg.NMS_CONFIG['TRAIN' if self.training and not test_only else 'TEST']
        targets_dict = self.proposal_layer(batch_dict, nms_config=nms_config)
        if (self.training or self.print_loss_when_eval) and not test_only:
            targets_dict = self.assign_targets(batch_dict)
            batch_dict['rois'] = targets_dict['rois']
            batch_dict['roi_labels'] = targets_dict['roi_labels']

        pooled_features = self.roi_grid_pool(batch_dict)  # (BxN, 6x6x6, C)
        grid_size = self.model_cfg.ROI_GRID_POOL.GRID_SIZE
        batch_size_rcnn = pooled_features.shape[0]
        pooled_features = pooled_features.permute(0, 2, 1). \
            contiguous().view(batch_size_rcnn, -1, grid_size, grid_size, grid_size)  # (BxN, C, 6, 6, 6)

        refined_grid_features = self.grid_transformer_encoder(pooled_features)
        shared_features = self.shared_fc_layer(refined_grid_features.contiguous().view(batch_size_rcnn, -1, 1)).squeeze(-1)
        rcnn_cls = self.cls_layers(shared_features)
        rcnn_reg = self.reg_layers(shared_features)

        if not self.training or self.predict_boxes_when_training:
            batch_cls_preds, batch_box_preds = self.generate_predicted_boxes(
                batch_size=batch_dict['batch_size'], rois=batch_dict['rois'], cls_preds=rcnn_cls, box_preds=rcnn_reg
            )
            # note that the rpn batch_cls_preds and batch_box_preds are being overridden here by rcnn preds
            batch_dict['batch_cls_preds'] = batch_cls_preds
            batch_dict['batch_box_preds'] = batch_box_preds
            batch_dict['cls_preds_normalized'] = False

        if self.training or self.print_loss_when_eval:
            targets_dict['rcnn_cls'] = rcnn_cls
            targets_dict['rcnn_reg'] = rcnn_reg
            self.forward_ret_dict = targets_dict


class GridTransformerEncoder(nn.Module):
    def __init__(self, feature_dim, num_heads, num_layers, hidden_dim):
        super().__init__()
        self.flatten_dim = 6 * 6 * 6  # Flattened grid size (6x6x6 = 216)
        self.positional_encoding = nn.Parameter(torch.randn(1, self.flatten_dim, feature_dim))
        # self.cls_token = nn.Parameter(torch.randn(1, 1, feature_dim))  # (1, 1, C)

        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=feature_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim,
            activation="relu"
        )
        self.transformer = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)

    def forward(self, pooled_features):
        B_N, C, G1, G2, G3 = pooled_features.shape

        pooled_features = pooled_features.view(B_N, C, -1)  # (B * N, C, 216)
        pooled_features = pooled_features.permute(0, 2, 1)  # (B * N, 216, C)
        # cls_token = self.cls_token.expand(B_N, -1, -1)  # (B * N, 1, C)
        # features_with_cls = torch.cat([cls_token, pooled_features], dim=1)  # (B * N, 217, C)
        features_with_pos = pooled_features + self.positional_encoding  # (B * N, 216, C)
        refined_features = self.transformer(features_with_pos)  # (B * N, 217, C)
        # cls_output = refined_features[:, 0, :]  # (B * N, C)
        # refined_grid_features = refined_features[:, 1:, :]  # (B * N, 216, C)
        refined_grid_features = refined_features.permute(0, 2, 1)  # (B * N, C, 216)
        refined_grid_features = refined_grid_features.view(B_N, C, G1, G2, G3)  # (B * N, C, 6, 6, 6)

        return refined_grid_features


class PVRCNNHeadWithProjector(PVRCNNHead):
    def __init__(self, input_channels, model_cfg, num_class=1, **kwargs):
        super().__init__(input_channels, model_cfg, num_class=num_class, **kwargs)
        self.cfgs = model_cfg
        self.proj_size = model_cfg.PROJECTION_SIZE
        input_size = model_cfg.SHARED_FC[-1]
        self.projector = weight_norm(nn.Linear(input_size, self.proj_size, bias=False))
        self.projector.weight_g.data.fill_(1)
        if model_cfg.DINO_LOSS.NORM_LAST_LAYER:
            self.projector.weight_g.requires_grad = False

        self.mask_token = nn.Parameter(torch.randn(1, 128))  # (1, C)

    def get_masked_feats(self, batch_dict, crop_size=4):
        pooled_features = self.roi_grid_pool(batch_dict)  # (BxN, 6x6x6, C)
        grid_size = self.cfgs.ROI_GRID_POOL.GRID_SIZE
        batch_size_rcnn = pooled_features.shape[0]
        pooled_features = pooled_features.permute(0, 2, 1). \
            contiguous().view(batch_size_rcnn, -1, grid_size, grid_size, grid_size)  # (BxN, C, 6, 6, 6)

        if crop_size and crop_size < grid_size:
            # Random Sub-Grid Crop
            start_x = random.randint(0, grid_size - crop_size)
            start_y = random.randint(0, grid_size - crop_size)
            start_z = random.randint(0, grid_size - crop_size)

            cropped_features = pooled_features[
                               :,  # Keep batch
                               :,  # Keep feature dimension
                               start_x:start_x + crop_size,
                               start_y:start_y + crop_size,
                               start_z:start_z + crop_size
                               ]  # (BxN, C, crop_size, crop_size, crop_size)

            # TODO(farzad): requires clone()?
            padded_features = self.mask_token.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand_as(pooled_features).clone()  # (BxN, C, 6, 6, 6)

            padded_features[:, :,
            start_x:start_x + crop_size,
            start_y:start_y + crop_size,
            start_z:start_z + crop_size
            ] = cropped_features

            pooled_features = padded_features  # (BxN, C, 6, 6, 6)

        pooled_features_flat = pooled_features.view(batch_size_rcnn, -1, 1)  # (BxN, Cx6x6x6, 1)
        shared_features = self.shared_fc_layer(pooled_features_flat).squeeze(-1)  # (BxN, D)
        if self.cfgs.DINO_LOSS.NORM_LAST_LAYER:
            shared_features = F.normalize(shared_features, p=2, dim=-1)
        proj_feats = self.projector(shared_features)
        return proj_feats

    def get_proj_feats(self, batch_dict):
        pooled_features = self.roi_grid_pool(batch_dict)  # (BxN, 6x6x6, C)
        grid_size = self.cfgs.ROI_GRID_POOL.GRID_SIZE
        batch_size_rcnn = pooled_features.shape[0]
        pooled_features = pooled_features.permute(0, 2, 1). \
            contiguous().view(batch_size_rcnn, -1, grid_size, grid_size, grid_size)  # (BxN, C, 6, 6, 6)
        # TODO(farzad): ablation: remove the shared_fc_layer
        shared_features = self.shared_fc_layer(pooled_features.view(batch_size_rcnn, -1, 1)).squeeze(-1)
        if self.cfgs.DINO_LOSS.NORM_LAST_LAYER:
            shared_features = F.normalize(shared_features, p=2, dim=-1)
        proj_feats = self.projector(shared_features)
        return proj_feats


class PVRCNNHeadWithSemCls(PVRCNNHead):
    def __init__(self, input_channels, model_cfg, num_class=1, **kwargs):
        super().__init__(input_channels, model_cfg, num_class=num_class, **kwargs)
        self.sem_cls_layer = None

    def get_box_sem_cls_layer_loss(self, forward_ret_dict):
        batch_size = forward_ret_dict['rcnn_cls_labels'].shape[0]

        sem_cls_preds = forward_ret_dict['rcnn_sem_cls']
        sem_cls_targets = forward_ret_dict['roi_labels'] - 1
        # obj_cls_targets = forward_ret_dict['rcnn_cls_labels'].view(-1)
        weights_lbl = forward_ret_dict['gt_iou_of_rois'].chunk(2)[0].view(-1)
        weights_ulb = torch.zeros_like(weights_lbl)
        weights = torch.cat([weights_lbl, weights_ulb])
        weights = (weights >= 0.55).float()
        sem_cls_targets_gt = forward_ret_dict['gt_of_rois'][..., -1].view(-1) - 1
        # lbl_gt_iou_rois = forward_ret_dict['gt_iou_of_rois'].chunk(2)[0].view(-1)
        # ulb_roi_scores = torch.sigmoid(forward_ret_dict['roi_scores'].chunk(2)[1].view(-1))
        # weights = torch.cat([lbl_gt_iou_rois, ulb_roi_scores])
        sem_cls_preds = sem_cls_preds.view(-1, 3)

        sem_cls_targets = sem_cls_targets.view(-1)
        batch_loss_cls = F.cross_entropy(sem_cls_preds, sem_cls_targets, reduction='none')

        # cls_valid_mask = (obj_cls_targets >= 0).float()
        batch_loss_cls = batch_loss_cls.reshape(batch_size, -1)
        # cls_valid_mask = cls_valid_mask.reshape(batch_size, -1)
        weights = weights.reshape(batch_size, -1)
        loss_sem_cls = (batch_loss_cls * weights).sum(-1) / torch.clamp(weights.sum(-1), min=1.0)

        precision = precision_score(sem_cls_targets.view(-1).cpu().numpy(),
                                    sem_cls_preds.max(dim=-1)[1].view(-1).cpu().numpy(),
                                    sample_weight=weights.view(-1).cpu().numpy(), average=None, labels=range(3),
                                    zero_division=np.nan)
        target_precision = precision_score(sem_cls_targets_gt.cpu().numpy(), sem_cls_targets.view(-1).cpu().numpy(),
                                           sample_weight=weights.view(-1).cpu().numpy(), average=None, labels=range(3),
                                           zero_division=np.nan)

        tb_dict = {
            'loss_sem_cls': loss_sem_cls,
            'rcnn_sem_cls_precision': self._arr2dict(precision),
            'rcnn_sem_cls_target_precision': self._arr2dict(target_precision),
        }
        return loss_sem_cls, tb_dict

    def _arr2dict(self, array, ignore_zeros=False, ignore_nan=False):
        def should_include(value):
            return not ((ignore_zeros and value == 0) or (ignore_nan and np.isnan(value)))

        classes = ['Bg', 'Fg'] if array.shape[-1] == 2 else ['Car', 'Pedestrian', 'Cyclist']
        return {cls: array[cind] for cind, cls in enumerate(classes) if should_include(array[cind])}

    def forward(self, batch_dict, test_only=False):
        raise NotImplementedError