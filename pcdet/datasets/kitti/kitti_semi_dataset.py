import copy
import pickle
from pathlib import Path
from . import kitti_utils
import io
import numpy as np
from ...utils import box_utils, calibration_kitti, common_utils, object3d_kitti
from ..semi_dataset import SemiDatasetTemplate
import hdbscan
import types
import torch
import matplotlib.pyplot as plt
import os

def split_kitti_semi_data(dataset_cfg, data_splits, logger, root_path=None):
    root_path = dataset_cfg.DATA_PATH if root_path is None else root_path
    root_path = Path(root_path)
    logger.info('Loading kitti dataset')

    train_info_path = dataset_cfg.INFO_PATH["train"][0]
    train_info_path = root_path.resolve() / train_info_path
    kitti_train_infos = []
    with open(train_info_path, 'rb') as f:
        infos = pickle.load(f)
        kitti_train_infos.extend(infos)

    test_info_path = dataset_cfg.INFO_PATH["test"][0]
    test_info_path = root_path.resolve() / test_info_path
    kitti_test_infos = []
    with open(test_info_path, 'rb') as f:
        infos = pickle.load(f)
        kitti_test_infos.extend(infos)

    train_split_lbl_path = root_path / "ImageSets" / (data_splits['train'] + ".txt")
    with open(train_split_lbl_path, "r") as f:
        sample_index_list_lbl = [int(x.split(" ")[1]) for x in f.readlines()]

    kitti_pretrain_infos = [kitti_train_infos[i] for i in sample_index_list_lbl]
    kitti_labeled_infos = copy.deepcopy(kitti_pretrain_infos)
    kitti_unlabeled_infos = [kitti_train_infos[i] for i in range(len(kitti_train_infos)) if i not in sample_index_list_lbl]

    logger.info('Total samples for kitti pre-training dataset: %d' % (len(kitti_pretrain_infos)))
    logger.info('Total samples for kitti labeled dataset: %d' % (len(kitti_labeled_infos)))
    logger.info('Total samples for kitti unlabeled dataset: %d' % (len(kitti_unlabeled_infos)))
    logger.info('Total samples for kitti testing dataset: %d' % (len(kitti_test_infos)))
    return kitti_pretrain_infos, kitti_labeled_infos, kitti_unlabeled_infos, kitti_test_infos


class KittiSemiDataset(SemiDatasetTemplate):
    def __init__(self, dataset_cfg, class_names, infos=None, training=True, root_path=None, logger=None, repeat=1):
        super().__init__(dataset_cfg=dataset_cfg, class_names=class_names, training=training, root_path=root_path, logger=logger)
        self.split = self.dataset_cfg.DATA_SPLIT[self.mode]
        self.root_split_path = self.root_path / ('training' if self.split != 'test' else 'testing')
        self.repeat = repeat

        split_dir = self.root_path / 'ImageSets' / (self.split + '.txt')
        self.sample_id_list = [x.strip() for x in open(split_dir).readlines()] if split_dir.exists() else None

        self.kitti_infos = infos
        self.groundremoval_hyperparameters_kitti = {
            'Step0__M': 0,
            'Step1__xyradius_threshold': 50.00,  # meters
            # KITTI’s ground is often around -1.65 m. Here we allow a ±1.0 m window.
            'Step1__zmin_threshold': -1.65 - 1.0,  # approximately -2.65 m
            'Step1__zmax_threshold': -1.65 + 1.0,  # approximately -0.65 m
            'Step2__min_sample_points': 100,      # KITTI point clouds are generally sparser than nuScenes, so we lower the minimum number of sample points for global plane fitting.
            'Step2__residual_threshold': 0.15,    # slightly looser threshold (meters)
            'Step2__max_trials': 20,            # For cone-based plane fitting, we keep a similar dmax threshold.
            'Step3__dmax_thres': 0.30,            # meters
            'Step3__num_cones': 8,            # Number of cones can remain similar.
            'Step3__min_number_cone_points': 300, # reduced from 500
            'Step3__min_sample_points': 100,      # reduced from 250
            'Step3__residual_threshold': 0.05,    # meters (kept tight for local plane fit)
            'Step3__max_trials': 10, # Reduced from 20
        }   
        self.sc_hyperparams = {
            'Step1__sky_threshold': 2.5,
            'Step1__range_threshold': 60,
            'Step1__x_range_threshold': 70.4,
            'Step1__y_range_threshold': 40,
            'Step2__clustersize_threshold': 5,
            'Step2__cluster_selection_epsilon': 0.5,
            'Step3__num_cones': 8,
            'Step4__length_max_threshold': 6,
            'Step4__width_max_threshold': 3,
            'Step4__height_min_threshold': 1.2,
            'Step4__height_above_ground_max_threshold': 2.0,
            'Step4__length_width_max_ratio_threshold': 2.0,
            'Step4__area_min_threshold': 1.0,
            'Step0__M': 0  # Single frame processing.
        }

    # # MODEST algorithm from UNION - Modified with available road_plane for grounding
    # def fit_bounding_box(self, cluster_reference: np.ndarray, road_plane: np.ndarray):
    #     """
    #     Fit a BEV bounding box around cluster points (in NumPy) and adjust the vertical placement using a provided road_plane.
        
    #     Args:
    #         cluster_reference (np.ndarray): LiDAR point cloud of the cluster in the reference frame with shape (N, 4).
    #                                         The first 3 columns are [x, y, z].
    #         road_plane (np.ndarray): Array of 4 elements [a, b, c, d] defining the ground plane
    #                                 via a*x + b*y + c*z + d = 0.
        
    #     Returns:
    #         T_reference_bbox (np.ndarray): A 4x4 homogeneous transformation matrix that maps points
    #                                     from the bounding box frame to the reference frame.
    #         bboxdimensions (List[float]): List containing the bounding box dimensions [length, width, height].
    #     """
    #     # --- Step 1: Determine the best rotation angle using a closeness metric.
    #     delta = 1  # degrees
    #     d0 = 1e-2
    #     max_beta = -float('inf')
    #     choose_angle = None

    #     for angle in np.arange(0, 90 + delta, delta):
    #         angle_rad = angle * np.pi / 180.0
    #         R_local_reference = np.array([[np.cos(angle_rad), np.sin(angle_rad)],
    #                                     [-np.sin(angle_rad), np.cos(angle_rad)]], dtype=np.float32)
    #         cluster_local = (R_local_reference @ cluster_reference[:, :2].T).T
    #         min_x = np.min(cluster_local[:, 0])
    #         max_x = np.max(cluster_local[:, 0])
    #         min_y = np.min(cluster_local[:, 1])
    #         max_y = np.max(cluster_local[:, 1])
    #         Dx = np.minimum(cluster_local[:, 0] - min_x, max_x - cluster_local[:, 0])
    #         Dy = np.minimum(cluster_local[:, 1] - min_y, max_y - cluster_local[:, 1])
    #         beta = np.minimum(Dx, Dy)
    #         beta = np.maximum(beta, d0)
    #         beta_sum = np.sum(1.0 / beta)
    #         if beta_sum > max_beta:
    #             max_beta = beta_sum
    #             choose_angle = angle_rad

    #     # --- Step 2: Recompute rotation using the chosen angle.
    #     R_local_reference = np.array([[np.cos(choose_angle), np.sin(choose_angle)],
    #                                 [-np.sin(choose_angle), np.cos(choose_angle)]], dtype=np.float32)
    #     cluster_local = (R_local_reference @ cluster_reference[:, :2].T).T
    #     min_x = np.min(cluster_local[:, 0])
    #     max_x = np.max(cluster_local[:, 0])
    #     min_y = np.min(cluster_local[:, 1])
    #     max_y = np.max(cluster_local[:, 1])

    #     if (max_x - min_x) < (max_y - min_y):
    #         choose_angle = choose_angle + np.pi / 2.0
    #         R_local_reference = np.array([[np.cos(choose_angle), np.sin(choose_angle)],
    #                                     [-np.sin(choose_angle), np.cos(choose_angle)]], dtype=np.float32)
    #         cluster_local = (R_local_reference @ cluster_reference[:, :2].T).T
    #         min_x = np.min(cluster_local[:, 0])
    #         max_x = np.max(cluster_local[:, 0])
    #         min_y = np.min(cluster_local[:, 1])
    #         max_y = np.max(cluster_local[:, 1])

    #     # --- Step 3: Compute the bounding box corners in the rotated (local) frame.
    #     corners_local = np.array([[max_x, min_y],
    #                             [min_x, min_y],
    #                             [min_x, max_y],
    #                             [max_x, max_y]], dtype=np.float32)
    #     corners_reference = (R_local_reference.T @ corners_local.T).T

    #     # --- Step 4: Compute the horizontal center from corners.
    #     center_x = np.mean(corners_reference[:, 0])
    #     center_y = np.mean(corners_reference[:, 1])
    #     # Compute ground height from the road plane:
    #     # Solve a*x + b*y + c*z + d = 0  =>  z = -(a*x + b*y + d)/c
    #     a, b, c, d = road_plane
    #     ground_z = -(a * center_x + b * center_y + d) / c

    #     # --- Step 5: Compute the bounding box height using the cluster's maximum z.
    #     bboxheight = np.max(cluster_reference[:, 2]) - ground_z
    #     # Set the vertical center such that the bottom of the box is at ground_z.
    #     center_z = ground_z + bboxheight / 2.0

    #     bboxcenter_reference = np.array([center_x, center_y, center_z], dtype=np.float32)

    #     # --- Step 6: Assemble the 4x4 homogeneous transformation matrix.
    #     T_reference_bbox = np.eye(4, dtype=np.float32)
    #     T_reference_bbox[:3, 3] = bboxcenter_reference
    #     T_reference_bbox[:2, 0] = R_local_reference.T[:, 0]
    #     T_reference_bbox[:2, 1] = R_local_reference.T[:, 1]

    #     # Compute horizontal dimensions:
    #     bboxlength = np.linalg.norm(corners_reference[1] - corners_reference[0])
    #     bboxwidth  = np.linalg.norm(corners_reference[3] - corners_reference[0])
    #     bboxdimensions = [float(bboxlength), float(bboxwidth), float(bboxheight)]

    #     return T_reference_bbox, bboxdimensions



    def spatial_clustering_adapted(self, pc_lidar: np.ndarray,
                                ground_mask: np.ndarray,
                                apply_filters: bool = True) -> dict:
        """
        Spatial clustering of LiDAR point cloud using a simplified approach.
        Args:
            pc_lidar (np.ndarray), ground_mask (np.ndarray), apply_filters (bool): If True, also remove sky and far-away points;
        Returns:
            dict: A dictionary mapping each cluster label to points in cluster
        """
        if apply_filters:
            sky_thres   = self.sc_hyperparams.get('Step1__sky_threshold', 3.0)
            range_thres = self.sc_hyperparams.get('Step1__range_threshold', 50.0)

            boolall_sky = pc_lidar[:, 2] >= sky_thres

            # Mark far-away points (radial distance > range_thres).
            radial_distance = np.linalg.norm(pc_lidar[:, :2], axis=1)
            boolall_outrange = radial_distance > range_thres

            boolall_outlier = ground_mask | boolall_sky | boolall_outrange
        else:
            boolall_outlier = ground_mask

        # Indices of inlier points
        ids_inlier = np.where(~boolall_outlier)[0]
        inlier_pc = pc_lidar[ids_inlier].copy()

        clustersize_thres = self.sc_hyperparams.get('Step2__clustersize_threshold', 10)
        cluster_selection_epsilon = self.sc_hyperparams.get('Step2__cluster_selection_epsilon', 0.5)

        clusterer = hdbscan.HDBSCAN(min_cluster_size=clustersize_thres,
                                    metric='euclidean',
                                    cluster_selection_epsilon=cluster_selection_epsilon)

        # Fit on the inlier points (using first 3 dims for 3D position)
        cluster_labels = clusterer.fit_predict(inlier_pc[:, :3])
        unique_labels = np.unique(cluster_labels)

        if pc_lidar.shape[1] > 3:
            points = pc_lidar[:, :3]
        else:
            points = pc_lidar

        full_labels = -99 * np.ones(points.shape[0], dtype=int)  # Initialize
        full_labels[ids_inlier] = cluster_labels                 # Assign labels to inlier points

        # # Path to save figure
        # save_path = '/mnt/data/adat01/adv_OpenPCDet/UNION_figs'
        # file_name = 'clusters_3d_viz_ulb.png'
        # os.makedirs(save_path, exist_ok=True)

        # fig = plt.figure(figsize=(10, 8))
        # ax = fig.add_subplot(111, projection='3d')

        # # Plot ground points in red
        # ground_pts = points[ground_mask]
        # ax.scatter(ground_pts[:, 0], ground_pts[:, 1], ground_pts[:, 2],
        #         c='r', s=1, label='Ground')

        # # We'll color each cluster uniquely, noise in black
        # cmap = plt.get_cmap("tab20")
        # non_ground_mask = ~ground_mask

        # # Loop over each label in the non-ground set
        # labels_non_ground = np.unique(full_labels[non_ground_mask])
        # for label in labels_non_ground:
        #     if label == -99:
        #         # Means it wasn't labeled (outlier in filtering)
        #         continue

        #     cluster_mask = (full_labels == label)
        #     cluster_points = points[cluster_mask]

        #     if label == -1:
        #         # Noise
        #         color = 'k'
        #         label_name = 'Noise'
        #     else:
        #         color = cmap(label % 20)
        #         label_name = f'Cluster {label}'

        #     ax.scatter(cluster_points[:, 0],
        #             cluster_points[:, 1],
        #             cluster_points[:, 2],
        #             c=[color], s=1, label=label_name)

        # ax.set_xlabel('X')
        # ax.set_ylabel('Y')
        # ax.set_zlabel('Z')
        # ax.set_title('3D Cluster Visualization')
        # ax.legend(loc='upper right')

        # # Save figure
        # full_save_path = os.path.join(save_path, file_name)
        # plt.savefig(full_save_path)
        # plt.close(fig)

        cluster_dict = {}
        for label in unique_labels:
            if label == -1:  # skip noise
                continue 
            cluster_indices = ids_inlier[cluster_labels == label]
            cluster_dict[label] = cluster_indices

        return cluster_dict


    def ground_point_removal_only(self, pc_lidar, calib, road_plane, hyperparameters=None):
        if hyperparameters is None:
            hyperparameters = self.groundremoval_hyperparameters_kitti
        dmax_thres = hyperparameters.get('Step3__dmax_thres', 0.30)
        # Transform LiDAR points to camera coordinates.
        pc_cam = calib.lidar_to_rect(pc_lidar[:, :3])  # (N,3)
        # Unpack the road plane parameters (assumed in camera coords).
        a, b, c, d = road_plane
        expected_y = (-d - a * pc_cam[:, 0] - c * pc_cam[:, 2]) / b     # The plane equation is: a*x + b*y + c*z + d = 0  ->  y = -(d + a*x + c*z)/b
        diff_y = np.abs(pc_cam[:, 1] - expected_y)    # Compute the absolute difference between the actual y and the expected ground y.
        boolall_ground = diff_y <= dmax_thres
        return boolall_ground


    def set_split(self, split):
        super().__init__(dataset_cfg=self.dataset_cfg, class_names=self.class_names, training=self.training,
                         root_path=self.root_path, logger=self.logger)
        self.split = split
        self.root_split_path = self.root_path / ('training' if self.split != 'test' else 'testing')

        split_dir = self.root_path / 'ImageSets' / (self.split + '.txt')
        self.sample_id_list = [x.strip() for x in open(split_dir).readlines()] if split_dir.exists() else None

    def get_lidar(self, idx):
        lidar_file = self.root_split_path / 'velodyne' / ('%s.bin' % idx)
        assert lidar_file.exists()
        points = np.fromfile(str(lidar_file), dtype=np.float32).reshape(-1, 4)

        return points

    def get_image(self, idx):
        """
        Loads image for a sample
        Args:
            idx: int, Sample index
        Returns:
            image: (H, W, 3), RGB Image
        """
        img_file = self.root_split_path / 'image_2' / ('%s.png' % idx)
        assert img_file.exists()
        image = io.imread(img_file)
        image = image.astype(np.float32)
        image /= 255.0
        return image

    def get_image_shape(self, idx):
        from skimage import io
        img_file = self.root_split_path / 'image_2' / ('%s.png' % idx)
        assert img_file.exists()
        return np.array(io.imread(img_file).shape[:2], dtype=np.int32)

    def get_label(self, idx):
        label_file = self.root_split_path / 'label_2' / ('%s.txt' % idx)
        assert label_file.exists()
        return object3d_kitti.get_objects_from_label(label_file)

    def get_depth_map(self, idx):
        """
        Loads depth map for a sample
        Args:
            idx: str, Sample index
        Returns:
            depth: (H, W), Depth map
        """
        depth_file = self.root_split_path / 'depth_2' / ('%s.png' % idx)
        assert depth_file.exists()
        depth = io.imread(depth_file)
        depth = depth.astype(np.float32)
        depth /= 256.0
        return depth
    
    def get_calib(self, idx):
        calib_file = self.root_split_path / 'calib' / ('%s.txt' % idx)
        assert calib_file.exists()
        calibrated_res = calibration_kitti.Calibration(calib_file)

        return calibrated_res
        
    def get_road_plane(self, idx):
        plane_file = self.root_split_path / 'planes' / ('%s.txt' % idx)
        if not plane_file.exists():
            return None

        with open(plane_file, 'r') as f:
            lines = f.readlines()
            
        lines = [float(i) for i in lines[3].split()]
        plane = np.asarray(lines)

        # Ensure normal is always facing up, this is in the rectified camera coordinate
        if plane[1] > 0:
            plane = -plane

        norm = np.linalg.norm(plane[0:3])
        plane = plane / norm
        return plane
        
    @staticmethod
    def get_fov_flag(pts_rect, img_shape, calib, margin=0):
        """
        Args:
            pts_rect:
            img_shape:
            calib:
            margin:
        Returns:

        """
        pts_img, pts_rect_depth = calib.rect_to_img(pts_rect)
        val_flag_1 = np.logical_and(pts_img[:, 0] >= 0 - margin, pts_img[:, 0] < img_shape[1] + margin)
        val_flag_2 = np.logical_and(pts_img[:, 1] >= 0 - margin, pts_img[:, 1] < img_shape[0] + margin)
        val_flag_merge = np.logical_and(val_flag_1, val_flag_2)
        pts_valid_flag = np.logical_and(val_flag_merge, pts_rect_depth >= 0)

        return pts_valid_flag

    def generate_prediction_dicts(self, batch_dict, pred_dicts, class_names, output_path=None):
        """
        Args:
            batch_dict:
                frame_id:
            pred_dicts: list of pred_dicts
                pred_boxes: (N, 7), Tensor
                pred_scores: (N), Tensor
                pred_labels: (N), Tensor
            class_names:
            output_path:

        Returns:

        """
        def get_template_prediction(num_samples):
            ret_dict = {
                'name': np.zeros(num_samples), 'truncated': np.zeros(num_samples),
                'occluded': np.zeros(num_samples), 'alpha': np.zeros(num_samples),
                'bbox': np.zeros([num_samples, 4]), 'dimensions': np.zeros([num_samples, 3]),
                'location': np.zeros([num_samples, 3]), 'rotation_y': np.zeros(num_samples),
                'score': np.zeros(num_samples), 'boxes_lidar': np.zeros([num_samples, 7])
            }
            return ret_dict

        def generate_single_sample_dict(batch_index, box_dict):
            pred_scores = box_dict['pred_scores'].cpu().numpy()
            pred_boxes = box_dict['pred_boxes'].cpu().numpy()
            pred_labels = box_dict['pred_labels'].cpu().numpy()
            pred_dict = get_template_prediction(pred_scores.shape[0])
            if pred_scores.shape[0] == 0:
                return pred_dict

            calib = batch_dict['calib'][batch_index]
            image_shape = batch_dict['image_shape'][batch_index].cpu().numpy()

            if self.dataset_cfg.get('SHIFT_COOR', None):
                #print ("*******WARNING FOR SHIFT_COOR:", self.dataset_cfg.SHIFT_COOR)
                pred_boxes[:, 0:3] -= self.dataset_cfg.SHIFT_COOR

            # TODO(farzad): This is new and should be checked.
            # BOX FILTER
            if self.dataset_cfg.get('TEST', None) and self.dataset_cfg.TEST.BOX_FILTER['FOV_FILTER']:
                box_preds_lidar_center = pred_boxes[:, 0:3]
                pts_rect = calib.lidar_to_rect(box_preds_lidar_center)
                fov_flag = self.get_fov_flag(pts_rect, image_shape, calib, margin=5)
                pred_boxes = pred_boxes[fov_flag]
                pred_labels = pred_labels[fov_flag]
                pred_scores = pred_scores[fov_flag]
            
            pred_boxes_camera = box_utils.boxes3d_lidar_to_kitti_camera(pred_boxes, calib)
            pred_boxes_img = box_utils.boxes3d_kitti_camera_to_imageboxes(
                pred_boxes_camera, calib, image_shape=image_shape
            )

            pred_dict['name'] = np.array(class_names)[pred_labels - 1]
            pred_dict['alpha'] = -np.arctan2(-pred_boxes[:, 1], pred_boxes[:, 0]) + pred_boxes_camera[:, 6]
            pred_dict['bbox'] = pred_boxes_img
            pred_dict['dimensions'] = pred_boxes_camera[:, 3:6]
            pred_dict['location'] = pred_boxes_camera[:, 0:3]
            pred_dict['rotation_y'] = pred_boxes_camera[:, 6]
            pred_dict['score'] = pred_scores
            pred_dict['boxes_lidar'] = pred_boxes

            return pred_dict

        annos = []
        for index, box_dict in enumerate(pred_dicts):
            frame_id = batch_dict['frame_id'][index]

            single_pred_dict = generate_single_sample_dict(index, box_dict)
            single_pred_dict['frame_id'] = frame_id
            annos.append(single_pred_dict)

            if output_path is not None:
                cur_det_file = output_path / ('%s.txt' % frame_id)
                with open(cur_det_file, 'w') as f:
                    bbox = single_pred_dict['bbox']
                    loc = single_pred_dict['location']
                    dims = single_pred_dict['dimensions']  # lhw -> hwl

                    for idx in range(len(bbox)):
                        print('%s -1 -1 %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f'
                              % (single_pred_dict['name'][idx], single_pred_dict['alpha'][idx],
                                 bbox[idx][0], bbox[idx][1], bbox[idx][2], bbox[idx][3],
                                 dims[idx][1], dims[idx][2], dims[idx][0], loc[idx][0],
                                 loc[idx][1], loc[idx][2], single_pred_dict['rotation_y'][idx],
                                 single_pred_dict['score'][idx]), file=f)

        return annos

    def evaluation(self, det_annos, class_names, **kwargs):
        if 'annos' not in self.kitti_infos[0].keys():
            return None, {}

        from .kitti_object_eval_python import eval as kitti_eval

        eval_det_annos = copy.deepcopy(det_annos)
        eval_gt_annos = [copy.deepcopy(info['annos']) for info in self.kitti_infos]
        ap_result_str, ap_dict = kitti_eval.get_official_eval_result(eval_gt_annos, eval_det_annos, class_names)

        return ap_result_str, ap_dict

    def __len__(self):
        if self._merge_all_iters_to_one_epoch:
            return len(self.kitti_infos) * self.total_epochs

        if self.training:
            return len(self.kitti_infos) * self.repeat
        else:
            return len(self.kitti_infos)

    def pre_getitem(self, index):
        if self._merge_all_iters_to_one_epoch:
            index = index % len(self.kitti_infos)

        index = index % len(self.kitti_infos)

        info = copy.deepcopy(self.kitti_infos[index])

        sample_idx = info['point_cloud']['lidar_idx']

        calib = self.get_calib(sample_idx)
        get_item_list = self.dataset_cfg.get('GET_ITEM_LIST', ['points'])

        input_dict = {
            'db_flag': "kitti",
            'frame_id': sample_idx,
            'calib': calib,
        }

        if 'annos' in info:
            annos = info['annos']
            annos = common_utils.drop_info_with_name(annos, name='DontCare')
            loc, dims, rots = annos['location'], annos['dimensions'], annos['rotation_y']
            gt_names = annos['name']
            gt_boxes_camera = np.concatenate([loc, dims, rots[..., np.newaxis]], axis=1).astype(np.float32)
            gt_boxes_lidar = box_utils.boxes3d_kitti_camera_to_lidar(gt_boxes_camera, calib)

            if self.dataset_cfg.get('SHIFT_COOR', None):
                gt_boxes_lidar[:, 0:3] += self.dataset_cfg.SHIFT_COOR

            input_dict.update({
                'gt_names': gt_names,
                'gt_boxes': gt_boxes_lidar
            })
            if "gt_boxes2d" in get_item_list:
                input_dict['gt_boxes2d'] = annos["bbox"]
            if self.dataset_cfg.get('REMOVE_ORIGIN_GTS', None) and self.training:
                input_dict['points'] = box_utils.remove_points_in_boxes3d(input_dict['points'], input_dict['gt_boxes'])
                mask = np.zeros(gt_boxes_lidar.shape[0], dtype=np.bool_)
                input_dict['gt_boxes'] = input_dict['gt_boxes'][mask]
                input_dict['gt_names'] = input_dict['gt_names'][mask]

            road_plane = self.get_road_plane(sample_idx)
            if road_plane is not None:
                input_dict['road_plane'] = road_plane

        if "points" in get_item_list:
            points = self.get_lidar(sample_idx)

            img_shape = info['image']['image_shape']
            if self.dataset_cfg.FOV_POINTS_ONLY:
                pts_rect = calib.lidar_to_rect(points[:, 0:3])
                fov_flag = self.get_fov_flag(pts_rect, img_shape, calib)
                points = points[fov_flag]
            if self.dataset_cfg.get('SHIFT_COOR', None):
                points[:, 0:3] += np.array(self.dataset_cfg.SHIFT_COOR, dtype=np.float32)
            input_dict['points'] = points

        if "images" in get_item_list:
            input_dict['images'] = self.get_image(sample_idx)

        if "depth_maps" in get_item_list:
            input_dict['depth_maps'] = self.get_depth_map(sample_idx)

        if "calib_matricies" in get_item_list:
            input_dict["trans_lidar_to_cam"], input_dict["trans_cam_to_img"] = kitti_utils.calib_to_matricies(calib)

        return input_dict, img_shape

    def __getitem__(self, index):
        input_dict, img_shape = self.pre_getitem(index)
        data_dict = self.prepare_data(data_dict=input_dict)
        data_dict['image_shape'] = img_shape
        return data_dict


class KittiLabeledDataset(KittiSemiDataset):
    def __init__(self, dataset_cfg, class_names, infos=None, training=True, root_path=None, logger=None, repeat=1):
        assert training is True
        super().__init__(dataset_cfg=dataset_cfg, class_names=class_names, infos=infos, training=training,
                         root_path=root_path, logger=logger, repeat=repeat)
        self.labeled_data_for = dataset_cfg.LABELED_DATA_FOR

    def __getitem__(self, index):
        input_dict, img_shape = self.pre_getitem(index)
        pc_lidar = input_dict['points']
        calib = input_dict['calib']
        all_ground_mask = self.ground_point_removal_only(pc_lidar, calib, input_dict['road_plane'])

        clusters = self.spatial_clustering_adapted(
            pc_lidar=pc_lidar, 
            ground_mask=all_ground_mask
        )
        # Inject the ground removal outputs into the input dictionary.
        input_dict['ground_mask'] = all_ground_mask  # Optionally as numpy array.
        input_dict['clusters'] = clusters #Access boxes using clusters[i].box_7d
        
        teacher_dict, student_dict = self.prepare_data_ssl(input_dict, prepare_for=self.labeled_data_for)

        if teacher_dict is not None:
            teacher_dict['image_shape'] = img_shape
        if student_dict is not None:
            student_dict['image_shape'] = img_shape

        return tuple([teacher_dict, student_dict])


class KittiUnlabeledDataset(KittiSemiDataset):
    def __init__(self, dataset_cfg, class_names, infos=None, training=True, root_path=None, logger=None, repeat=1):
        assert training is True
        super().__init__(dataset_cfg=dataset_cfg, class_names=class_names, infos=infos, training=training,
                         root_path=root_path, logger=logger, repeat=repeat)
        self.unlabeled_data_for = dataset_cfg.UNLABELED_DATA_FOR

    def __getitem__(self, index):
        if self._merge_all_iters_to_one_epoch:
            index = index % len(self.kitti_infos)

        info = copy.deepcopy(self.kitti_infos[index])

        sample_idx = info['point_cloud']['lidar_idx']
        
        calib = self.get_calib(sample_idx)
        get_item_list = self.dataset_cfg.get('GET_ITEM_LIST', ['points'])

        input_dict = {
            'db_flag': "kitti",
            'frame_id': sample_idx,
            'calib': calib,
        }
        # if 'annos' in info:
        #     annos = info['annos']
        #     annos = common_utils.drop_info_with_name(annos, name='DontCare')
        #     loc, dims, rots = annos['location'], annos['dimensions'], annos['rotation_y']
        #     gt_names = annos['name']
        #     gt_boxes_camera = np.concatenate([loc, dims, rots[..., np.newaxis]], axis=1).astype(np.float32)
        #     gt_boxes_lidar = box_utils.boxes3d_kitti_camera_to_lidar(gt_boxes_camera, calib)
        #     if self.dataset_cfg.get('SHIFT_COOR', None):
        #         gt_boxes_lidar[:, 0:3] += self.dataset_cfg.SHIFT_COOR

        #     input_dict.update({
        #         'gt_names': gt_names,
        #         'gt_boxes': gt_boxes_lidar,
        #         'ulb': True
        #     })

        #     if "gt_boxes2d" in get_item_list:
        #         input_dict['gt_boxes2d'] = annos["bbox"]

        #     if self.dataset_cfg.get('REMOVE_ORIGIN_GTS', None) and self.training:
        #         input_dict['points'] = box_utils.remove_points_in_boxes3d(input_dict['points'], input_dict['gt_boxes'])
        #         mask = np.zeros(gt_boxes_lidar.shape[0], dtype=np.bool_)
        #         input_dict['gt_boxes'] = input_dict['gt_boxes'][mask]
        #         input_dict['gt_names'] = input_dict['gt_names'][mask]

        #     road_plane = self.get_road_plane(sample_idx)
        #     if road_plane is not None:
        #         input_dict['road_plane'] = road_plane

        if "points" in get_item_list:
            points = self.get_lidar(sample_idx)
            
            img_shape = info['image']['image_shape']
            if self.dataset_cfg.FOV_POINTS_ONLY:
                pts_rect = calib.lidar_to_rect(points[:, 0:3])
                fov_flag = self.get_fov_flag(pts_rect, img_shape, calib)
                points = points[fov_flag]
            if self.dataset_cfg.get('SHIFT_COOR', None):
                points[:, 0:3] += np.array(self.dataset_cfg.SHIFT_COOR, dtype=np.float32)
            input_dict['points'] = points

        if "images" in get_item_list:
            input_dict['images'] = self.get_image(sample_idx)

        if "depth_maps" in get_item_list:
            input_dict['depth_maps'] = self.get_depth_map(sample_idx)

        if "calib_matricies" in get_item_list:
            input_dict["trans_lidar_to_cam"], input_dict["trans_cam_to_img"] = kitti_utils.calib_to_matricies(calib)
         
        road_plane = self.get_road_plane(sample_idx) # Using unlabeled road_plane [In Camera Frame-of]

        pc_lidar =  input_dict['points']

        all_ground_mask = self.ground_point_removal_only(pc_lidar, calib, road_plane)

        # Call the spatial clustering function:
        clusters = self.spatial_clustering_adapted(
            pc_lidar=pc_lidar, 
            ground_mask=all_ground_mask
            )

        input_dict['ground_mask'] = all_ground_mask  # Optionally as numpy array.
        input_dict['clusters'] = clusters
        teacher_dict, student_dict = self.prepare_data_ssl(input_dict, prepare_for=self.unlabeled_data_for)
        
        if teacher_dict is not None:
            teacher_dict['image_shape'] = img_shape
        if student_dict is not None:
            student_dict['image_shape'] = img_shape

        return tuple([teacher_dict, student_dict])