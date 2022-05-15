## Introduction
基于 [MMDetection](https://github.com/open-mmlab/mmdetection) 和 [SlowFast](https://github.com/facebookresearch/SlowFast) 的动作时空定位，可以灵活选用 MMDetection 中任意检测模型进行动作时空定位。

<div align="center">
  <img src="https://github.com/yzfly/ActionDetection/raw/master/demo/run_city_short.gif" width="800px"/><br>
</div>

## Installation
先安装 MMDetection
* [mmdetection](https://github.com/yzfly/ActionDetection/raw/master/docs/en/get_started.md)

然后安装 python 包
```
pip install opencv-python decord pytorchvideo ipdb
```

## Getting Started

可以轻松的选用多种检测模型(参考 MMDetection 模型库） 作为检测器进行动作时空定位

使用 [Deformable DETR (ICLR'2021)](https://github.com/yzfly/ActionDetection/tree/master/configs/deformable_detr)

```
export CUDA_VISIBLE_DEVICES=0
python action_det.py --video demo/run_the_city.mp4  --imsize 224 \
--config configs/deformable_detr/deformable_detr_twostage_refine_r50_16x2_50e_coco.py \
--checkpoint weights/deformable_detr_twostage_refine_r50_16x2_50e_coco_20210419_220613-9d28ab72.pth \
--out demo/run_the_city_demo.mp4
```

使用 [YOLOX (CVPR'2021)](https://github.com/yzfly/ActionDetection/tree/master/configs/yolox)

```
export CUDA_VISIBLE_DEVICES=0
python action_det.py --video demo/run_the_city.mp4  --imsize 224 \
--config configs/yolox/yolox_x_8x8_300e_coco.py \
--checkpoint weights/yolox_x_8x8_300e_coco_20211126_140254-1ef88d67.pth \
--out demo/run_the_city_demo.mp4
```

and more ...


## More Results

<div align="center">
  <img src="https://github.com/yzfly/ActionDetection/raw/master/demo/meeting_demo.gif" width="800px"/><br>
</div>


## References

Thanks for these wonderful works:

[1] [mmdetection](https://github.com/open-mmlab/mmdetection) 

[2] [yolo_slowfast](https://github.com/wufan-tb/yolo_slowfast)

[3] [ZQPei/deepsort](https://github.com/ZQPei/deep_sort_pytorch) 

[4] [FAIR/PytorchVideo](https://github.com/facebookresearch/pytorchvideo)

[5] AVA: A Video Dataset of Spatio-temporally Localized Atomic Visual Actions. [paper](https://arxiv.org/pdf/1705.08421.pdf)

[6] SlowFast Networks for Video Recognition. [paper](https://arxiv.org/pdf/1812.03982.pdf)