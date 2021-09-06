# Voxel Transformer

This is a reproduced repo of Voxel Transformer for 3D object detection. 

The code is mainly based on [OpenPCDet](https://github.com/open-mmlab/OpenPCDet).

## Introduction
We provide code and training configurations of VoTr-SSD/TSD on the KITTI and Waymo Open dataset. Checkpoints will not be released.  

**Important Notes**: VoTr generally requires quite a long time (more than 60 epochs on Waymo) to converge, and a large GPU memory (32Gb) is needed for reproduction.
Please strictly follow the instructions and train with sufficient number of epochs.
If you don't have a 32G GPU, you can decrease the attention SIZE parameters in yaml files, but this may possibly harm the performance. 

## Requirements
The codes are tested in the following environment:
* Ubuntu 18.04
* Python 3.6
* PyTorch 1.5
* CUDA 10.1
* OpenPCDet v0.3.0
* spconv v1.2.1

## Installation
a. Clone this repository.
```shell
git clone https://github.com/PointsCoder/VOTR.git
```

b. Install the dependent libraries as follows:

* Install the dependent python libraries: 
```
pip install -r requirements.txt 
```

* Install the SparseConv library, we use the implementation from [`[spconv]`](https://github.com/traveller59/spconv). 
    * If you use PyTorch 1.1, then make sure you install the `spconv v1.0` with ([commit 8da6f96](https://github.com/traveller59/spconv/tree/8da6f967fb9a054d8870c3515b1b44eca2103634)) instead of the latest one.
    * If you use PyTorch 1.3+, then you need to install the `spconv v1.2`. As mentioned by the author of [`spconv`](https://github.com/traveller59/spconv), you need to use their docker if you use PyTorch 1.4+. 

c. Compile CUDA operators by running the following command:
```shell
python setup.py develop
```

## Training

All the models are trained with Tesla V100 GPUs (32G). 
The KITTI config of votr_ssd is for training with a **single** GPU.
Other configs are for training with 8 GPUs.
If you use different number of GPUs for training, it's necessary to change the respective training epochs to attain a decent performance.

The performance of VoTr is quite unstable on KITTI. If you cannnot reproduce the results, remember to run it multiple times.

* models
```shell script
# votr_ssd.yaml: single-stage votr backbone replacing the spconv backbone
# votr_tsd.yaml: two-stage votr with pv-head
```

* training votr_ssd on kitti
```shell script
CUDA_VISIBLE_DEVICES=0 python train.py --cfg_file cfgs/kitti_models/votr_ssd.yaml
```

* training other models
```shell script
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 sh scripts/dist_train.sh 8 --cfg_file cfgs/waymo_models/votr_tsd.yaml
```

* testing
```shell script
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 sh scripts/dist_test.sh 8 --cfg_file cfgs/waymo_models/votr_tsd.yaml --eval_all
```

## Citation 
If you find this project useful in your research, please consider cite:

```
@article{mao2021voxel,
  title={Voxel Transformer for 3D Object Detection},
  author={Mao, Jiageng and Xue, Yujing and Niu, Minzhe and others},
  journal={ICCV},
  year={2021}
}
```