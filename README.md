## Labeled to Pseudo-labeled contrastive learning for 3D LIDAR SSOD

This repository introduces a novel approach to address challenges in 3D object detection on point clouds within a semi-supervised learning context, using only 1% labeled data. Our method leverages embedding-level information to define the decision boundary, incorporating a feature bank with instance-specific feature guidance. We significantly improve the mean Average Precision (mAP) across the Car, Pedestrian, and Cyclist classes by facilitating information exchange between labelled and pseudo-labelled feature data, even under varying difficulty levels. Our baseline is 3DIoUMatch with a PV-RCNN backbone. The results, illustrated in the image below, demonstrate that our approach enhances the performance of the underrepresented Cyclist and Pedestrian classes prone to data imbalance by providing stronger feature-level guidance.


<p align="center">
  <img src="docs/LPCont_results.png" width="95%" height="320">
</p>



# About OpenPCDet Framework
`OpenPCDet` is a general PyTorch-based codebase for 3D object detection from point cloud.  As an open source project for LiDAR-based 3D scene perception that supports multiple LiDAR-based state-of-the-art perception models with highly refactored codes for both one-stage and two-stage 3D detection frameworks. Examples include Point-RCNN, PV-RCNN, PV-RCNN++, CenterPoint, SECOND, MPP-Net, and several others.


### `OpenPCDet` design pattern

* Data-Model separation with unified point cloud coordinate for easily extending to custom datasets:
<p align="center">
  <img src="docs/dataset_vs_model.png" width="95%" height="320">
</p>

* Unified 3D box definition: (x, y, z, dx, dy, dz, heading).

* Flexible and clear model structure to easily support various 3D detection models: 
<p align="center">
  <img src="docs/model_framework.png" width="95%">
</p>

* Support various models within one framework as: 
<p align="center">
  <img src="docs/multiple_models_demo.png" width="95%">
</p>


### Currently Supported Features

- [x] Support both one-stage and two-stage 3D object detection frameworks
- [x] Support distributed training & testing with multiple GPUs and multiple machines
- [x] Support multiple heads on different scales to detect different classes
- [x] Support stacked version set abstraction to encode various number of points in different scenes
- [x] Support Adaptive Training Sample Selection (ATSS) for target assignment
- [x] Support RoI-aware point cloud pooling & RoI-grid point cloud pooling
- [x] Support GPU version 3D IoU calculation and rotated NMS 



## Installation

Please refer to [INSTALL.md](docs/INSTALL.md) for the installation of `OpenPCDet`.

## Quick Demo
Please refer to [DEMO.md](docs/DEMO.md) for a quick demo to test with a pretrained model and 
visualize the predicted results on your custom data or the original KITTI data.

## Getting Started

Please refer to [GETTING_STARTED.md](docs/GETTING_STARTED.md) to learn more usage about this project.


## License

`OpenPCDet` is released under the [Apache 2.0 license](LICENSE).

## Acknowledgement
`OpenPCDet` is an open source project for LiDAR-based 3D scene perception that supports multiple
LiDAR-based perception models as shown above. Some parts of `PCDet` are learned from the official released codes of the above supported methods. 
We would like to thank for their proposed methods and the official implementation.   


## Citation 
If you find this project useful in your research, please consider cite:


```
@misc{openpcdet2020,
    title={OpenPCDet: An Open-source Toolbox for 3D Object Detection from Point Clouds},
    author={OpenPCDet Development Team},
    howpublished = {\url{https://github.com/open-mmlab/OpenPCDet}},
    year={2020}
}
```

