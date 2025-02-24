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


    def fit_bounding_box(self, cluster_reference: torch.Tensor):
        """
        Dummy implementation of fit_bounding_box.
        
        Args:
            cluster_reference (torch.Tensor): Point cloud for the cluster (N,4). We assume the first 3 columns are x, y, z.
        
        Returns:
            T_reference_bbox (torch.Tensor): A 4x4 identity matrix (dummy transformation).
            bboxdimensions (List[float]): Bounding box dimensions computed as the range (max-min) along each axis.
        """
        if cluster_reference.shape[0] == 0:
            T_reference_bbox = torch.eye(4, dtype=cluster_reference.dtype, device=cluster_reference.device)
            bboxdimensions = [0.0, 0.0, 0.0]
        else:
            mins = torch.min(cluster_reference[:, :3], dim=0).values
            maxs = torch.max(cluster_reference[:, :3], dim=0).values
            dims = (maxs - mins).tolist()  # [length, width, height] as dummy dimensions
            T_reference_bbox = torch.eye(4, dtype=cluster_reference.dtype, device=cluster_reference.device)
            bboxdimensions = dims
     
        return T_reference_bbox, bboxdimensions


    def spatial_clustering(self, pc_lidar: torch.Tensor,
                        boolall_ground: torch.Tensor,
                        Ts_coneplane_lidar: np.ndarray,
                        original_lengths: dict,
                        hyperparameters: dict) -> dict:
        """
        Cluster inlier points of a point cloud spatially for a single sweep.
        
        Args:
            pc_lidar (torch.Tensor): LiDAR point cloud in LiDAR frame, shape (N,4).
            boolall_ground (torch.Tensor): Boolean mask indicating ground points, shape (N,).
            Ts_coneplane_lidar (np.ndarray): NumPy array containing cone-plane homogeneous transformation
                                            matrices with shape (num_cones, 4, 4).
            original_lengths (dict): Dictionary with the indices boundaries for the original sweep (e.g. {0: [0, N]}).
            hyperparameters (dict): Hyperparameters for spatial clustering.
        
        Returns:
            cluster_dict (dict): Dictionary containing cluster information including fitted bounding boxes.
        """
        # --- Step 0: For multi-frame setups, M would denote extra frames. Here M=0.
        M = hyperparameters.get('Step0__M', 0)
        
        # --- Step 1: Filter out points considered as ground, sky, or too far away.
        sky_thres     = hyperparameters['Step1__sky_threshold']  # meters
        range_thres   = hyperparameters['Step1__range_threshold']  # meters
        x_range_thres = hyperparameters.get('Step1__x_range_threshold', None)
        y_range_thres = hyperparameters.get('Step1__y_range_threshold', None)
        
        # Use the global plane transformation (first cone) as a reference.
        # Convert the first cone-plane from numpy to torch.
        T_globalplane_lidar = torch.from_numpy(Ts_coneplane_lidar[0]).float()
        
        # Transform the point cloud using T_globalplane (to align with plane reference)
        pc_globalplane = (T_globalplane_lidar @ pc_lidar.T).T
        
        # Identify sky points: points with height (z in transformed frame) above the sky threshold.
        boolall_sky = pc_globalplane[:, 2] >= sky_thres
        
        # Determine points that are out of range.
        if range_thres is not None:
            boolall_outrange = torch.linalg.norm(pc_lidar[:, :2], ord=2, axis=1) > range_thres
        else:
            boolall_outrange = (torch.abs(pc_lidar[:, 0]) > x_range_thres) | (torch.abs(pc_lidar[:, 1]) > y_range_thres)
        
        # Combine masks: a point is an outlier if it is ground, sky, or out-of-range.
        boolall_outlier = boolall_ground | boolall_sky | boolall_outrange
        idsall_inlier = torch.where(~boolall_outlier)[0]
        
        inlier_pc_lidar = pc_lidar[idsall_inlier].clone()
        
        # --- Step 2: Cluster the inlier points using HDBSCAN.
        clustersize_thres = hyperparameters['Step2__clustersize_threshold']
        cluster_selection_epsilon = hyperparameters['Step2__cluster_selection_epsilon']
        
        total_num_frames = len(original_lengths)  # For single sweep, this is 1.
        
        hdbscan_clusterer = hdbscan.HDBSCAN(min_cluster_size=clustersize_thres,
                                            metric='euclidean',
                                            cluster_selection_epsilon=cluster_selection_epsilon)
        # We cluster using only the first three coordinates (x, y, z)
        hdbscan_cluster_labels = torch.IntTensor(hdbscan_clusterer.fit_predict(inlier_pc_lidar[:, :3]))
        
        # --- Step 3: Fit bounding boxes to each cluster using a MODEST-like approach.
        num_cones = hyperparameters['Step3__num_cones']
        fov_cone = 360 / num_cones  # degrees per cone
        
        # For single-sweep, create a timestamp tensor that marks all points with the same frame (e.g., frame 0)
        timestamp_tensor = torch.zeros(inlier_pc_lidar.shape[0], dtype=torch.int32)
        
        cluster_dict = {}
        # Skip label 0 as it typically represents noise in HDBSCAN.
        for label in torch.unique(hdbscan_cluster_labels)[1:].tolist():
            # Get indices of the cluster
            idsall_cluster = idsall_inlier[torch.where(hdbscan_cluster_labels == label)[0]]
            
            # Fit a bounding box using the provided fit_bounding_box function.
            #TODO : fit_bounding_box -> dummy method implemented. What is to be done with this?
            T_lidar_bbox, bboxdimensions = self.fit_bounding_box(pc_lidar[idsall_cluster, :])
            
            # Transform inlier points into the bounding box frame.
            inlier_pc_bbox = (torch.linalg.inv(T_lidar_bbox) @ inlier_pc_lidar.T).T
            boolall_insidebbox = (
                (torch.abs(inlier_pc_bbox[:, 0]) <= bboxdimensions[0] / 2) &
                (torch.abs(inlier_pc_bbox[:, 1]) <= bboxdimensions[1] / 2) &
                (inlier_pc_bbox[:, 2] >= 0) &
                (inlier_pc_bbox[:, 2] <= bboxdimensions[2])
            )
            idsall_insidebbox = idsall_inlier[boolall_insidebbox]
            
            # Determine cone index based on the position of the bounding box center.
            # Compute an angle from the translation part of T_lidar_bbox.
            cone_idx = int((((180 / np.pi * torch.atan2(T_lidar_bbox[1, 3], T_lidar_bbox[0, 3]) + 360) % 360) / fov_cone) % num_cones)
            # Convert the corresponding cone-plane to torch.
            T_coneplane_lidar = torch.from_numpy(Ts_coneplane_lidar[cone_idx]).float()
            
            # Calculate the height of the bounding box center above ground.
            height_above_ground = (T_coneplane_lidar @ T_lidar_bbox)[2, 3].item()
            
            # Adjust the bounding box so it "touches" the ground.
            touchground_T_lidar_bbox = T_lidar_bbox.clone()
            touchground_T_lidar_bbox[2, 3] += -height_above_ground
            touchground_bboxdimensions = bboxdimensions.copy()
            touchground_bboxdimensions[2] += height_above_ground
            
            # Create a simple namespace to hold cluster data.
            cluster = types.SimpleNamespace()
            cluster.avg_number_points = len(idsall_cluster) / total_num_frames
            cluster.T_lidar_bbox = T_lidar_bbox.numpy()
            cluster.bboxdimensions = bboxdimensions
            cluster.yaw_radians = torch.atan2(T_lidar_bbox[1, 0], T_lidar_bbox[0, 0]).item()
            cluster.touchground_T_lidar_bbox = touchground_T_lidar_bbox.numpy()
            cluster.touchground_bboxdimensions = touchground_bboxdimensions
            cluster.touchground_yaw_radians = torch.atan2(touchground_T_lidar_bbox[1, 0], touchground_T_lidar_bbox[0, 0]).item()
            cluster.height_above_ground = height_above_ground
            cluster.idsall_aggregated = idsall_cluster.tolist()
            # For a single sweep, all points are in frame 0.
            cluster.idsall_frame = {0: idsall_cluster.tolist()}
            cluster.idsall_aggregated2 = idsall_insidebbox.tolist()
            cluster.idsall_frame2 = {0: idsall_insidebbox.tolist()}
            
            cluster_dict[label] = cluster
            
        # --- Step 4: Filter clusters based on size and shape thresholds.
        length_max_threshold = hyperparameters['Step4__length_max_threshold']
        width_max_threshold = hyperparameters['Step4__width_max_threshold']
        height_min_threshold = hyperparameters['Step4__height_min_threshold']
        height_above_ground_max_threshold = hyperparameters['Step4__height_above_ground_max_threshold']
        length_width_max_ratio_threshold = hyperparameters['Step4__length_width_max_ratio_threshold']
        area_min_threshold = hyperparameters['Step4__area_min_threshold']
        
        sorted_cluster_list = sorted(cluster_dict.values(), key=lambda c: c.avg_number_points)
        filtered_cluster_list = []
        for cluster in sorted_cluster_list:
            bboxlength, bboxwidth, bboxheight = cluster.bboxdimensions
            height_above_ground = cluster.height_above_ground
            
            if bboxlength > length_max_threshold:
                continue
            elif bboxwidth > width_max_threshold:
                continue
            elif bboxheight < height_min_threshold:
                continue
            elif height_above_ground > height_above_ground_max_threshold:
                continue
            elif bboxlength / bboxwidth > length_width_max_ratio_threshold:
                continue
            elif bboxlength * bboxwidth < area_min_threshold:
                continue
            else:
                # Ensure the cluster has at least one point from the current frame.
                if len(cluster.idsall_frame.get(0, [])) > 0:
                    filtered_cluster_list.append(cluster)
        
        # Re-index clusters into a dictionary.
        cluster_dict = dict(zip(range(len(filtered_cluster_list)), filtered_cluster_list))
        
        return cluster_dict


    def get_T_plane_reference(self, road_plane: np.ndarray) -> np.ndarray:
        """
        Compute a 4x4 homogeneous transformation matrix that aligns the provided road_plane
        to a canonical reference frame where the road becomes flat (i.e., the plane lies at z=0).
        
        Args:
            road_plane (np.ndarray): 1D array of shape (4,) representing the plane parameters
                                    [a, b, c, d] for the plane equation:
                                    a*x + b*y + c*z + d = 0.
                                    The road_plane is assumed to be normalized such that
                                    the normal vector [a, b, c] is unit length.
        
        Returns:
            T (np.ndarray): A 4x4 homogeneous transformation matrix. When applied to a point in homogeneous
                            coordinates, this transformation rotates and translates the point so that the road_plane
                            becomes horizontal (with z=0).
        """
        # Extract the normal (n) and offset (d)
        n = road_plane[:3].astype(np.float32)
        d = float(road_plane[3])
        
        # Ensure the normal is unit length.
        norm_n = np.linalg.norm(n)
        if norm_n < 1e-6:
            raise ValueError("Normal vector is too small!")
        n = n / norm_n
        d = d / norm_n
        
        # Our target normal is [0, 0, 1] (so that after transformation the ground is horizontal)
        target = np.array([0, 0, 1], dtype=np.float32)
        
        # Compute the rotation matrix that rotates n to target using Rodrigues' formula.
        v = np.cross(n, target)
        s = np.linalg.norm(v)
        c = np.dot(n, target)
        
        if s < 1e-6:
            # The normal is already aligned (or opposite) to the target.
            R = np.eye(3, dtype=np.float32)
        else:
            # Skew-symmetric matrix for v.
            vx = np.array([[0, -v[2], v[1]],
                        [v[2], 0, -v[0]],
                        [-v[1], v[0], 0]], dtype=np.float32)
            R = np.eye(3, dtype=np.float32) + vx + np.dot(vx, vx) * ((1 - c) / (s ** 2))
        
        # Compute a point on the plane (the closest point to the origin): p0 = -d * n.
        p0 = -d * n
        # Rotate this point.
        p0_rot = R.dot(p0)
        
        # We want the transformed plane to lie at z=0.
        # Compute the translation t so that the z-coordinate of R*p0 becomes zero.
        t = np.array([0, 0, -p0_rot[2]], dtype=np.float32)
        
        # Assemble the 4x4 homogeneous transformation matrix.
        T = np.eye(4, dtype=np.float32)
        T[:3, :3] = R
        T[:3, 3] = t
        return T

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
        if 'road_plane' in input_dict: 
            road_plane = input_dict['road_plane']  # e.g., [a, b, c, d]
        else:
            raise ValueError("No road_plane provided for sample {}".format(input_dict['frame_id']))

        # Convert points (a numpy array) to torch.Tensor for processing
        pc = torch.from_numpy(input_dict['points']).float()  # shape: (N, 4)

        # Compute the transformation matrix directly using the provided road_plane.
        # (Assume get_T_plane_reference is a utility function that computes a 4x4 matrix from the plane parameters.)
        T_globalplane = self.get_T_plane_reference(road_plane)  # shape: (4,4)

        plane_tensor = torch.tensor(road_plane, dtype=torch.float32)
        norm_val = torch.linalg.norm(plane_tensor[:3])
        distances = (pc[:, 0]*plane_tensor[0] + pc[:, 1]*plane_tensor[1] + 
                        pc[:, 2]*plane_tensor[2] + plane_tensor[3]) / norm_val

        # Define a threshold (e.g., 0.15 m) for ground points.
        ground_thresh = 0.15
        ground_mask = distances.abs() < ground_thresh  # Boolean tensor: True if point is ground.

        # For spatial clustering, we want to work with the same single-sweep data.
        # Create a dummy "original_lengths" dictionary.
        original_lengths = {0: [0, pc.shape[0]]}

        # Define clustering hyperparameters ( M=0 , only single-sweep)
        sc_hyperparams = {
            'Step1__sky_threshold': 2.5,
            'Step1__range_threshold': 50,
            'Step2__clustersize_threshold': 10,
            'Step2__cluster_selection_epsilon': 0.5,
            'Step3__num_cones': 12,
            'Step4__length_max_threshold': 6,
            'Step4__width_max_threshold': 3,
            'Step4__height_min_threshold': 1.2,
            'Step4__height_above_ground_max_threshold': 2.5,
            'Step4__length_width_max_ratio_threshold': 2.0,
            'Step4__area_min_threshold': 1.5,
            'Step0__M': 0  # Single frame processing.
        }

        # For spatial clustering, if your algorithm still divides the FOV into cones,
        # simply replicate T_globalplane for the number of cones:
        num_cones = sc_hyperparams['Step3__num_cones']
        Ts_coneplane = np.repeat(np.expand_dims(T_globalplane, axis=0), num_cones, axis=0)
        # Call the spatial clustering function:
        clusters = self.spatial_clustering(
            pc_lidar=pc, 
            boolall_ground=ground_mask, 
            Ts_coneplane_lidar=Ts_coneplane, 
            original_lengths=original_lengths, 
            hyperparameters=sc_hyperparams
        )

        # Inject the ground removal outputs into the input dictionary.
        input_dict['ground_mask'] = ground_mask.numpy()  # Optionally as numpy array.
        input_dict['clusters'] = clusters
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
        
        road_plane = np.array([-0.04675316, -0.99886225, 0.00939886, 1.58625293]) # Static road_plane for unlabeled_samples

        # Convert points (a numpy array) to torch.Tensor for processing
        pc = torch.from_numpy(input_dict['points']).float()  # shape: (N, 4)

        # Compute the transformation matrix directly using the provided road_plane.
        # (Assume get_T_plane_reference is a utility function that computes a 4x4 matrix from the plane parameters.)
        T_globalplane = self.get_T_plane_reference(road_plane)  # shape: (4,4)

        plane_tensor = torch.tensor(road_plane, dtype=torch.float32)
        norm_val = torch.linalg.norm(plane_tensor[:3])
        distances = (pc[:, 0]*plane_tensor[0] + pc[:, 1]*plane_tensor[1] + 
                        pc[:, 2]*plane_tensor[2] + plane_tensor[3]) / norm_val

        # Define a threshold (e.g., 0.15 m) for ground points.
        ground_thresh = 0.15
        ground_mask = distances.abs() < ground_thresh  # Boolean tensor: True if point is ground.

        # For spatial clustering, we want to work with the same single-sweep data.
        # Create a dummy "original_lengths" dictionary.
        original_lengths = {0: [0, pc.shape[0]]}

        # Define clustering hyperparameters ( M=0 , only single-sweep)
        sc_hyperparams = {
            'Step1__sky_threshold': 2.5,
            'Step1__range_threshold': 50,
            'Step2__clustersize_threshold': 10,
            'Step2__cluster_selection_epsilon': 0.5,
            'Step3__num_cones': 12,
            'Step4__length_max_threshold': 6,
            'Step4__width_max_threshold': 3,
            'Step4__height_min_threshold': 1.2,
            'Step4__height_above_ground_max_threshold': 2.5,
            'Step4__length_width_max_ratio_threshold': 2.0,
            'Step4__area_min_threshold': 1.5,
            'Step0__M': 0  # Single frame processing.
        }

        # For spatial clustering, if your algorithm still divides the FOV into cones,
        # simply replicate T_globalplane for the number of cones:
        num_cones = sc_hyperparams['Step3__num_cones']
        Ts_coneplane = np.repeat(np.expand_dims(T_globalplane, axis=0), num_cones, axis=0)
        # Call the spatial clustering function:
        clusters = self.spatial_clustering(
            pc_lidar=pc, 
            boolall_ground=ground_mask, 
            Ts_coneplane_lidar=Ts_coneplane, 
            original_lengths=original_lengths, 
            hyperparameters=sc_hyperparams
        )
        # Inject the ground removal outputs into the input dictionary.
        input_dict['ground_mask'] = ground_mask.numpy()  # Optionally as numpy array.
        input_dict['clusters'] = clusters
        teacher_dict, student_dict = self.prepare_data_ssl(input_dict, prepare_for=self.labeled_data_for)
        
        if teacher_dict is not None:
            teacher_dict['image_shape'] = img_shape
        if student_dict is not None:
            student_dict['image_shape'] = img_shape

        return tuple([teacher_dict, student_dict])