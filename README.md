# MESFINet
 Pytorch implementation of "Multi-Stage Edge-Guided Stereo Feature Interaction Network for Stereoscopic Image Super-Resolution"
[paper](https://ieeexplore.ieee.org/document/10121360).

## Overview

<img src="/figs/arch.jpg" width="700px">

The architecture of our proposed Multi-Stage Edge-Guided Stereo Feature Interaction Network. 

Citation:

```latex
@ARTICLE{10121360,
  author={Wan, Jin and Yin, Hui and Liu, Zhihao and Liu, Yanting and Wang, Song},
  journal={IEEE Transactions on Broadcasting}, 
  title={Multi-Stage Edge-Guided Stereo Feature Interaction Network for Stereoscopic Image Super-Resolution}, 
  year={2023},
  volume={69},
  number={2},
  pages={357-368},
  doi={10.1109/TBC.2023.3264880}}
```

## Contents
1. [Train](#train)
2. [Test](#test)
3. [Results](#results)

## Train
### Begin to train

1. Run the following scripts to train models.
**You can use scripts in the file 'demo' to train models for our paper.**

    ```bash
    # BI, scale 2, 4
    # MESFINet in the paper (x2)
    # CUDA_VISIBLE_DEVICES=3,4 python train.py --model_name MESFINet --scale_factor 2 --checkpoints_dir ./log \
    # --edge_model_path ./pretrained_model/bdcn_pretrained_on_nyudv2_rgb.pth 
    
    # MESFINet in the paper (x4) - from MESFINet (x2)
    # CUDA_VISIBLE_DEVICES=3,4 python train.py --model_name MESFINet --scale_factor 4 --checkpoints_dir ./log --edge_model_path ./pretrained_model/bdcn_pretrained_on_nyudv2_rgb.pth \
    # --load_pretrain_modelx2 True --modelx2_path ./pretrained_model/MESFINet_2xSR_final.pth.tar

    ```


## Test

1. Clone this repository:

   ```shell
   git clone https://github.com/jinwan1994/MESFINet.git
   ```
2. All the models (BIX2/4) can be downloaded from [GoogleYun](https://drive.google.com/drive/folders/14v9jSUwdAu_-L-eilulYD3HQChYNFufb?usp=sharing), place the models to `./pretrained_model/`. 

3. Run the following scripts.

    **You can use scripts in the file 'demo' to produce results for our paper.**

    ```bash
    # BI, scale 2, 4
    # Standard benchmarks (Ex. MESFINetx2)
    # python test.py --model_name MESFINet_2xSR --scale_factor 2 --save_dir ./results_test --sr_model_path ./pretrained_model/MESFINet_2xSR_final.pth.tar \
    # 	--dataset_list KITTI2012+KITTI2015+Middlebury --edge_model_path ./pretrained_model/bdcn_pretrained_on_nyudv2_rgb.pth
    
    # Standard benchmarks (Ex. MESFINet_x4)
    python test.py --model_name MESFINet_4xSR --scale_factor 4 --save_dir ./results_test --sr_model_path ./pretrained_model/MESFINet_4xSR_final.pth.tar \
    	--dataset_list KITTI2012+KITTI2015+Middlebury --edge_model_path ./pretrained_model/bdcn_pretrained_on_nyudv2_rgb.pth
    ```
4. Finally, SR results and PSNR/SSIM values for test data are saved to `./results_test/*`. (PSNR/SSIM values in our paper are obtained using Matlab2021)

## Results

#### Quantitative Results

<img src="/figs/result_1.jpg" width="650px">

Benchmark SR results. Average PSNR/SSIM for scale factor x2 and x4 on four datasets.

#### Visual Results

<img src="/figs/result_2.jpg" width="650px">

Visual comparison for 2x SR on two datasets.

