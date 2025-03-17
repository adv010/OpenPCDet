from collections import defaultdict
from pathlib import Path
import copy
import numpy as np
import torch.utils.data as torch_data

from ..utils import common_utils, box_utils
from .augmentor.data_augmentor import DataAugmentor
from .augmentor.ssl_data_augmentor import SSLDataAugmentor
from .processor.data_processor import DataProcessor
from .processor.point_feature_encoder import PointFeatureEncoder
import hdbscan
import torch
from sklearn.linear_model import RANSACRegressor
class SemiDatasetTemplate(torch_data.Dataset):
    def __init__(self, dataset_cfg=None, class_names=None, training=True, root_path=None, logger=None):
        super().__init__()
        self.dataset_cfg = dataset_cfg
        self.training = training
        self.class_names = class_names
        self.logger = logger
        self.root_path = Path(root_path) if root_path is not None else Path(self.dataset_cfg.DATA_PATH)
        self.logger = logger
        if self.dataset_cfg is None or class_names is None:
            return

        self.point_cloud_range = np.array(self.dataset_cfg.POINT_CLOUD_RANGE, dtype=np.float32)
        self.point_feature_encoder = PointFeatureEncoder(
            self.dataset_cfg.POINT_FEATURE_ENCODING,
            point_cloud_range=self.point_cloud_range
        )

        self.data_augmentor = DataAugmentor(
            self.root_path, self.dataset_cfg.DATA_AUGMENTOR, self.class_names,
            logger=self.logger) if self.training else None

        self.data_processor = DataProcessor(
            self.dataset_cfg.DATA_PROCESSOR, point_cloud_range=self.point_cloud_range, training=self.training,
            num_point_features=self.point_feature_encoder.num_point_features
        )
        self.teacher_augmentor = SSLDataAugmentor(
            self.root_path, self.dataset_cfg.TEACHER_AUGMENTOR, self.class_names,
            logger=self.logger) if self.training else None

        self.student_augmentor = SSLDataAugmentor(
            self.root_path, self.dataset_cfg.STUDENT_AUGMENTOR, self.class_names,
            logger=self.logger) if self.training else None

        self.grid_size = self.data_processor.grid_size
        self.voxel_size = self.data_processor.voxel_size
        self.total_epochs = 0
        self._merge_all_iters_to_one_epoch = False

        if hasattr(self.data_processor, "depth_downsample_factor"):
            self.depth_downsample_factor = self.data_processor.depth_downsample_factor
        else:
            self.depth_downsample_factor = None

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
            'Step2__clustersize_threshold': 3,
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


    def ground_point_removal_only(self, pc, road_plane, hyperparameters=None):
        if hyperparameters is None:
            hyperparameters = self.groundremoval_hyperparameters_kitti
        # KITTI LiDAR frame: (X forward, Y left, Z up)
        pc_cam = pc  # (N, 4)
        a, b, c, d = road_plane
        expected_z = (-d - a * pc_cam[:, 0] - b * pc_cam[:, 1]) / c  
        # Compute absolute height difference
        diff_z = np.abs(pc_cam[:, 2] - expected_z)    
        threshold = np.mean(diff_z)
        boolall_ground = diff_z <= threshold
        return boolall_ground

    def ground_point_removal_ransac_sklearn(self, pc, distance_threshold=0.2):
        """
        Uses sklearn's RANSACRegressor to estimate the ground plane and remove ground points.
        
        Args:
            pc (numpy.ndarray): (N, 4) point cloud in KITTI LiDAR coordinates (X forward, Y left, Z up, intensity).
            distance_threshold (float): Maximum residual for a point to be considered an inlier.

        Returns:
            bool_ground_mask (numpy.ndarray): Boolean mask (True for ground points, False for non-ground).
            plane_parameters (numpy.ndarray): Normalized ground plane parameters (a, b, c, d).
        """
        # Extract X, Y, Z coordinates
        X_train = pc[:, [0, 1]]  # Use (x, y) as features
        y_train = pc[:, 2]       # Use z as the target (height)

        # Fit RANSAC model for ground plane estimation
        ransac = RANSACRegressor(min_samples=3, residual_threshold=distance_threshold)
        ransac.fit(X_train, y_train)

        # Get plane parameters
        a, b = ransac.estimator_.coef_  # Coefficients for x and y
        c = -1                           # Normalized z coefficient
        d = ransac.estimator_.intercept_  # Intercept term

        # Store in NumPy array
        plane_parameters = np.array([a, b, c, d])

        # Normalize the plane normal (a, b, c)
        plane_parameters = -plane_parameters / np.linalg.norm(plane_parameters[:3])

        # Compute expected Z values for each point
        expected_z = (plane_parameters[0] * pc[:, 0] +
                    plane_parameters[1] * pc[:, 1] +
                    plane_parameters[3]) / -plane_parameters[2]

        # Compute absolute difference in Z height
        diff_z = np.abs(pc[:, 2] - expected_z)

        # Apply thresholding to identify ground points
        bool_ground_mask = diff_z <= distance_threshold

        return bool_ground_mask, plane_parameters

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

        # Fit on the inlier points
        cluster_labels = clusterer.fit_predict(inlier_pc[:, :3])
        unique_labels = np.unique(cluster_labels)

        if pc_lidar.shape[1] > 3:
            points = pc_lidar[:, :3]
        else:
            points = pc_lidar

        full_labels = -99 * np.ones(points.shape[0], dtype=int)  # Initialize
        full_labels[ids_inlier] = cluster_labels                 # Assign labels to inlier points

        cluster_dict = {}
        for label in unique_labels:
            if label == -1:  # skip noise
                continue 
            cluster_indices = ids_inlier[cluster_labels == label]
            cluster_dict[label] = cluster_indices

        # # Flatten all cluster indices into a single list
        # all_indices = [idx for indices in cluster_dict.values() for idx in indices]

        # # Convert to a set and check uniqueness
        # are_unique = len(all_indices) == len(set(all_indices))
        # print("Are all clusters unique (no shared indices)?", are_unique)


        # from collections import Counter

        # # Count occurrences of each index
        # index_counts = Counter(all_indices)

        # # Find max shared occurrences (should be >1 if indices are shared)
        # max_shared = max(index_counts.values()) if not are_unique else 0
        # print("Max points shared between any two clusters:", max_shared)

        # tolerance = 100  # Allowable error range
        # expected_total = len(full_labels) + ground_mask.sum()
        ## total_clustered_points = sum(len(indices) for indices in cluster_dict.values())

        # is_within_tolerance = abs(total_clustered_points - expected_total) <= tolerance

        # print("Total clustered points:", total_clustered_points)
        # print("Expected total (PC points + ground points):", expected_total)
        # print(f"Does sum of clustered points match within ±{tolerance}?", is_within_tolerance)

        return cluster_dict, full_labels

    def fit_bounding_box(self, cluster_reference: np.ndarray, road_plane: np.ndarray):
        """
        Fit a KITTI-format bounding box to a cluster.
        
        Returns:
            list: KITTI-style bounding box [x, y, z, dx, dy, dz, yaw].
        """
        delta = 1  # degrees
        max_beta = -float('inf')
        choose_angle = None

        for angle in np.arange(0, 90 + delta, delta):
            angle_rad = np.radians(angle)
            R_local_reference = np.array([[np.cos(angle_rad), np.sin(angle_rad)],
                                        [-np.sin(angle_rad), np.cos(angle_rad)]], dtype=np.float32)
            cluster_local = (R_local_reference @ cluster_reference[:, :2].T).T
            min_x, max_x = np.min(cluster_local[:, 0]), np.max(cluster_local[:, 0])
            min_y, max_y = np.min(cluster_local[:, 1]), np.max(cluster_local[:, 1])

            Dx = np.minimum(cluster_local[:, 0] - min_x, max_x - cluster_local[:, 0])
            Dy = np.minimum(cluster_local[:, 1] - min_y, max_y - cluster_local[:, 1])
            beta = np.minimum(Dx, Dy)
            beta_sum = np.sum(1.0 / np.maximum(beta, 1e-2))
            if beta_sum > max_beta:
                max_beta = beta_sum
                choose_angle = angle_rad

        # Rotate using best angle
        R_local_reference = np.array([[np.cos(choose_angle), np.sin(choose_angle)],
                                    [-np.sin(choose_angle), np.cos(choose_angle)]], dtype=np.float32)
        cluster_local = (R_local_reference @ cluster_reference[:, :2].T).T
        min_x, max_x = np.min(cluster_local[:, 0]), np.max(cluster_local[:, 0])
        min_y, max_y = np.min(cluster_local[:, 1]), np.max(cluster_local[:, 1])

        corners_local = np.array([[max_x, min_y], [min_x, min_y], [min_x, max_y], [max_x, max_y]], dtype=np.float32)
        corners_reference = (R_local_reference.T @ corners_local.T).T

        # Compute center and height (fixed)
        center_x, center_y = np.mean(corners_reference[:, 0]), np.mean(corners_reference[:, 1])
        ground_z = np.median(cluster_reference[:, 2])  # Use median Z for stability
        bboxheight = np.max(cluster_reference[:, 2]) - ground_z
        center_z = ground_z + bboxheight / 2.0

        # Compute correct length, width
        bboxlength = np.linalg.norm(corners_reference[1] - corners_reference[0])
        bboxwidth  = np.linalg.norm(corners_reference[3] - corners_reference[0])

        # Ensure length > width
        if bboxwidth > bboxlength:
            bboxlength, bboxwidth = bboxwidth, bboxlength  # Swap if needed
            choose_angle += np.pi / 2.0  # Adjust yaw accordingly

        return [center_x, center_y, center_z, bboxlength, bboxwidth, bboxheight, choose_angle]

    @property
    def mode(self):
        return 'train' if self.training else 'test'

    def __getstate__(self):
        d = dict(self.__dict__)
        del d['logger']
        return d

    def __setstate__(self, d):
        self.__dict__.update(d)

    @staticmethod
    def generate_prediction_dicts(batch_dict, pred_dicts, class_names, output_path=None):
        """
        To support a custom dataset, implement this function to receive the predicted results from the model, and then
        transform the unified normative coordinate to your required coordinate, and optionally save them to disk.

        Args:
            batch_dict: dict of original data from the dataloader
            pred_dicts: dict of predicted results from the model
                pred_boxes: (N, 7), Tensor
                pred_scores: (N), Tensor
                pred_labels: (N), Tensor
            class_names:
            output_path: if it is not None, save the results to this path
        Returns:

        """

    def merge_all_iters_to_one_epoch(self, merge=True, epochs=None):
        if merge:
            self._merge_all_iters_to_one_epoch = True
            self.total_epochs = epochs
        else:
            self._merge_all_iters_to_one_epoch = False

    def __len__(self):
        raise NotImplementedError

    def __getitem__(self, index):
        """
        To support a custom dataset, implement this function to load the raw data (and labels), then transform them to
        the unified normative coordinate and call the function self.prepare_data() to process the data and send them
        to the model.

        Args:
            index:

        Returns:

        """
        raise NotImplementedError

    def prepare_data(self, data_dict):
        """
        Args:
            data_dict:
                points: (N, 3 + C_in)
                gt_boxes: optional, (N, 7 + C) [x, y, z, dx, dy, dz, heading, ...]
                gt_names: optional, (N), string
                ...

        Returns:
            data_dict:
                frame_id: string
                points: (N, 3 + C_in)
                gt_boxes: optional, (N, 7 + C) [x, y, z, dx, dy, dz, heading, ...]
                gt_names: optional, (N), string
                use_lead_xyz: bool
                voxels: optional (num_voxels, max_points_per_voxel, 3 + C)
                voxel_coords: optional (num_voxels, 3)
                voxel_num_points: optional (num_voxels)
                ...
        """
        if self.training:
            assert 'gt_boxes' in data_dict, 'gt_boxes should be provided for training'
            gt_boxes_mask = np.array([n in self.class_names for n in data_dict['gt_names']], dtype=np.bool_)

            data_dict = self.data_augmentor.forward(
                data_dict={
                    **data_dict,
                    'gt_boxes_mask': gt_boxes_mask
                }
            )
            if len(data_dict['gt_boxes']) == 0:
                new_index = np.random.randint(self.__len__())
                return self.__getitem__(new_index)

        if data_dict.get('gt_boxes', None) is not None:
            selected = common_utils.keep_arrays_by_name(data_dict['gt_names'], self.class_names)
            data_dict['gt_boxes'] = data_dict['gt_boxes'][selected]
            data_dict['gt_names'] = data_dict['gt_names'][selected]
            gt_classes = np.array([self.class_names.index(n) + 1 for n in data_dict['gt_names']], dtype=np.int32)
            gt_boxes = np.concatenate((data_dict['gt_boxes'], gt_classes.reshape(-1, 1).astype(np.float32)), axis=1)
            data_dict['gt_boxes'] = gt_boxes

        data_dict = self.point_feature_encoder.forward(data_dict)

        data_dict = self.data_processor.forward(
            data_dict=data_dict
        )
        data_dict.pop('gt_names', None)

        return data_dict

    def prepare_data_ssl(self, input_data_dict, prepare_for, road_plane):
        
        if 'gt_boxes' in input_data_dict:
            gt_boxes_mask = np.array([n in self.class_names for n in input_data_dict['gt_names']], dtype=np.bool_)
            input_data_dict = {
                **input_data_dict,
                'gt_boxes_mask': gt_boxes_mask
            }

        teacher_data_dict = self.teacher_augmentor.forward(
            copy.deepcopy(input_data_dict)) if 'teacher' in prepare_for else None
        student_data_dict = self.student_augmentor.forward(
            copy.deepcopy(input_data_dict)) if 'student' in prepare_for else None

        for i, data_dict in enumerate([input_data_dict, teacher_data_dict, student_data_dict]):
            if data_dict is None:
                continue

            if 'gt_boxes' in data_dict:
                if len(data_dict['gt_boxes']) == 0:
                    new_index = np.random.randint(self.__len__())
                    return self.__getitem__(new_index)

                selected = common_utils.keep_arrays_by_name(data_dict['gt_names'], self.class_names)
                data_dict['gt_boxes'] = data_dict['gt_boxes'][selected]
                data_dict['gt_names'] = data_dict['gt_names'][selected]
                gt_classes = np.array([self.class_names.index(n) + 1 for n in data_dict['gt_names']], dtype=np.int32)
                gt_boxes = np.concatenate((data_dict['gt_boxes'], gt_classes.reshape(-1, 1).astype(np.float32)), axis=1)
                data_dict['gt_boxes'] = gt_boxes

            data_dict = self.point_feature_encoder.forward(data_dict)

            data_dict = self.data_processor.forward(
                data_dict=data_dict
            )

            if i==0:
                continue # Skip for data_dict
            else:
                pc_lidar = data_dict['points']
                if road_plane is not None:
                    best_fit_plane = road_plane
                all_ground_mask = self.ground_point_removal_only(pc_lidar, best_fit_plane)
                ransac_ground_mask, plane_parameters  = self.ground_point_removal_ransac_sklearn (pc_lidar)
                clusters, cluster_labels= self.spatial_clustering_adapted(
                    pc_lidar=pc_lidar, 
                    ground_mask=ransac_ground_mask
                )
                data_dict['ground_mask'] = all_ground_mask  
                data_dict['ransac_ground_mask'] = ransac_ground_mask
                data_dict['clusters'] = clusters
                cluster_boxes = []  # Store KITTI-style boxes

                for cluster_id, point_indices in clusters.items():
                    cluster_points = pc_lidar[point_indices]
                    if cluster_points.shape[0] == 0:
                        continue
                    kitti_box = self.fit_bounding_box(cluster_points, road_plane)
                    kitti_box=torch.tensor(kitti_box)
                    kitti_box[-1] = common_utils.limit_period(kitti_box[-1], offset=0.5, period=2 * np.pi )
                    cluster_boxes.append(kitti_box.numpy())  # Append KITTI-style box
                
                data_dict['cluster_boxes'] = cluster_boxes
            
            data_dict.pop('gt_names', None)

        return (teacher_data_dict, student_data_dict) if teacher_data_dict or student_data_dict else input_data_dict

    @staticmethod
    def collate_batch(batch_list, _unused=False):

        def collate_single_batch(batch_list):
            data_dict = defaultdict(list)
            for cur_sample in batch_list:
                if isinstance(cur_sample, dict):
                    for key, val in cur_sample.items():
                        data_dict[key].append(val)
                else:
                    raise Exception('batch samples must be dict')

            batch_size = len(batch_list)
            ret = {}
            for key, val in data_dict.items():
                try:
                    if key in ['voxels', 'voxel_num_points']:
                        ret[key] = np.concatenate(val, axis=0)
                    elif key in ['points', 'voxel_coords']:
                        coors = []
                        for i, coor in enumerate(val):
                            coor_pad = np.pad(coor, ((0, 0), (1, 0)), mode='constant', constant_values=i)
                            coors.append(coor_pad)
                        ret[key] = np.concatenate(coors, axis=0)
                    elif key in ['gt_boxes']:
                        max_gt = max([len(x) for x in val])
                        batch_gt_boxes3d = np.zeros((batch_size, max_gt, val[0].shape[-1]), dtype=np.float32)
                        for k in range(batch_size):
                            batch_gt_boxes3d[k, :val[k].__len__(), :] = val[k]
                        ret[key] = batch_gt_boxes3d
                    elif key in ['augmentation_list', 'augmentation_params']:
                        ret[key] = val
                    elif key in ['ground_mask']:
                        ret[key] = val
                    else:
                        ret[key] = np.stack(val, axis=0)
                except:
                    print('Error in collate_batch: key=%s' % key)
                    raise TypeError

            ret['batch_size'] = batch_size
            return ret

        if isinstance(batch_list[0], dict):
            return collate_single_batch(batch_list)
        elif isinstance(batch_list[0], tuple):
            if batch_list[0][0] is None:
                teacher_batch = None
            else:
                teacher_batch_list = [sample[0] for sample in batch_list]
                teacher_batch = collate_single_batch(teacher_batch_list)
            if batch_list[0][1] is None:
                student_batch = None
            else:
                student_batch_list = [sample[1] for sample in batch_list]
                student_batch = collate_single_batch(student_batch_list)
            return teacher_batch, student_batch
        else:
            raise Exception('batch samples must be dict or tuple')
