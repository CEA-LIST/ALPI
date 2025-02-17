# ALPI: Auto-Labeller with Proxy Injection for 3D Object Detection using 2D Labels Only
3D object detection plays a crucial role in various applications such as autonomous vehicles, robotics and augmented reality. However, training 3D detectors requires a costly precise annotation, which is a hindrance to scaling annotation to large datasets. To address this challenge, we propose a weakly supervised 3D annotator that relies solely on 2D bounding box annotations from images, along with size priors. One major problem is that supervising a 3D detection model using only 2D boxes is not reliable due to ambiguities between different 3D poses and their identical 2D projection. We introduce a simple yet effective and generic solution: we build 3D proxy objects with annotations by construction and add them to the training dataset. Our method requires only size priors to adapt to new classes. To better align 2D supervision with 3D detection, our method ensures depth invariance with a novel expression of the 2D losses. Finally, to detect more challenging instances, our annotator follows an offline pseudo-labelling scheme which gradually improves its 3D pseudo-labels. Extensive experiments on the KITTI dataset demonstrate that our method not only performs on-par or above previous works on the Car category, but also achieves performance close to fully supervised methods on more challenging classes. We further demonstrate the effectiveness and robustness of our method by being the first to experiment on the more challenging nuScenes dataset. We additionally propose a setting where weak labels are obtained from a 2D detector pre-trained on MS-COCO instead of human annotations.

# Reference
The preprint is on [arxiv](https://arxiv.org/abs/2407.17197) and the reference of the paper is:
```
@inproceedings{lahlali2025alpi,
  title={ALPI: Auto-Labeller with Proxy Injection for 3D Object Detection using 2D Labels Only},
  author={Lahlali, Saad and Granger, Nicolas and Borgne, Herv{\'e} Le and Pham, Quoc-Cuong},
  booktitle={IEEE/CVF Winter Conference on Applications of Computer Vision (WACV)},
  year={2025}
}
```
