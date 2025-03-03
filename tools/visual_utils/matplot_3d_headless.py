import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from io import BytesIO
from PIL import Image

box_colormap = [
    (0, 1, 0),  # Green
    (0, 1, 1),  # Cyan
    (1, 1, 0),  # Yellow
]

classes = {0: 'Car', 1: 'Pedestrian', 2: 'Cyclist'}

class Matplotlib3DRenderer:
    """3D visualization using Matplotlib."""

    def __init__(self):
        self.fig = plt.figure(figsize=(10, 10))
        self.ax = self.fig.add_subplot(111, projection='3d')

    def clear_scene(self):
        """Clears the current scene."""
        self.ax.clear()
        self.ax.set_xlabel("X")
        self.ax.set_ylabel("Y")
        self.ax.set_zlabel("Z")
        self.ax.set_box_aspect([1, 1, 0.5])
        self.ax.grid(False)

    def plot_points(self, points, keypoints, color='gray', color2='yellow'):
        """Plots a 3D point cloud."""
        self.ax.scatter(points[:, 0], points[:, 1], points[:, 2], c=color, s=1, alpha=0.4)
        self.ax.scatter(keypoints[:, 0], keypoints[:, 1], keypoints[:, 2], c=color2, s=1.02, alpha=0.8)

    def get_bbox_corners(self, box):
        """Computes the 8 corner points for a given 3D bounding box."""
        x, y, z, dx, dy, dz, yaw = box
        R = np.array([
            [np.cos(yaw), -np.sin(yaw), 0],
            [np.sin(yaw), np.cos(yaw), 0],
            [0, 0, 1]
        ])
        l, w, h = dx / 2, dy / 2, dz / 2
        corners = np.array([
            [-l, -w, -h], [l, -w, -h], [l, w, -h], [-l, w, -h],
            [-l, -w, h], [l, -w, h], [l, w, h], [-l, w, h]
        ])
        rotated_corners = (R @ corners.T).T + np.array([x, y, z])
        return rotated_corners

    def plot_bbox(self, box, color='b'):
        """Plots a single 3D bounding box with rotation alignment."""
        corners = self.get_bbox_corners(box)
        edges = [(0, 1), (1, 2), (2, 3), (3, 0),
                 (4, 5), (5, 6), (6, 7), (7, 4),
                 (0, 4), (1, 5), (2, 6), (3, 7)]
        for i, j in edges:
            self.ax.plot([corners[i, 0], corners[j, 0]],
                         [corners[i, 1], corners[j, 1]],
                         [corners[i, 2], corners[j, 2]],
                         color=color, linewidth=2, alpha=0.8)

    def set_camera(self, points, bboxes):
        """Dynamically adjusts the camera based on point cloud and bounding boxes."""
        if len(points) > 0:
            min_vals = points.min(axis=0)
            max_vals = points.max(axis=0)
        else:
            min_vals = max_vals = np.array([0, 0, 0])

        if bboxes is not None and len(bboxes) > 0:
            for box in bboxes:
                min_vals = np.minimum(min_vals, box[:3] - box[3:6] / 2)
                max_vals = np.maximum(max_vals, box[:3] + box[3:6] / 2)

        self.ax.set_xlim(min_vals[0], max_vals[0])
        self.ax.set_ylim(min_vals[1], max_vals[1])
        self.ax.set_zlim(min_vals[2], max_vals[2] + 5)
        self.ax.view_init(elev=40, azim=-90)

    def render_scene(self, points, gt_boxes=None, gt_labels=None, keypoints=None, ref_boxes=None, ref_labels=None):
        """Renders the scene with point clouds and bounding boxes."""
        self.clear_scene()
        self.plot_points(points, keypoints)

        if gt_boxes is not None:
            for i, box in enumerate(gt_boxes):
                color = 'blue'
                self.plot_bbox(box, color=color)
                if gt_labels is not None:
                    self.ax.text(box[0], box[1], box[2] + 1, classes.get(gt_labels[i], "Unknown"), color=color)

        if ref_boxes is not None:
            for i, box in enumerate(ref_boxes):
                color = 'green'
                self.plot_bbox(box, color=color)
                if ref_labels is not None:
                    self.ax.text(box[0], box[1], box[2] + 1, classes.get(ref_labels[i], "Unknown"), color=color)

        self.set_camera(points, gt_boxes if gt_boxes is not None else ref_boxes)

    def render_scene_tb(self, points, gt_boxes=None, gt_labels=None, keypoints=None):
        """Renders the scene and returns it as a NumPy image array."""
        self.render_scene(points, gt_boxes, gt_labels, keypoints)
        return self.fig
