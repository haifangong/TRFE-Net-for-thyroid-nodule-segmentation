# Multi-Task Learning for Thyroid Nodule Segmentation with Thyroid Region Prior [link](https://www.researchgate.net/publication/349074982_MULTI-TASK_LEARNING_FOR_THYROID_NODULE_SEGMENTATION_WITH_THYROID_REGION_PRIOR)

## Introduction
Thyroid nodule segmentation in ultrasound images is a valuable and challenging task, and it is of great significance for the diagnosis of thyroid cancer. Due to the lack of the prior knowledge of thyroid region perception, the inherent low contrast of ultrasound images and the complex appearance changes between different frames of ultrasound video, existing automatic segmentation algorithms for thyroid nodules that directly apply semantic segmentation techniques can easily mistake non-thyroid areas as nodules. In this work, we propose a thyroid region prior guided feature enhancement network (TRFE-Net) for thyroid nodule segmentation. In order to facilitate the development of thyroid nodule segmentation, we have contributed TN3k: an open-access dataset of thyroid nodule images with high-quality nodule masks labeling. Our proposed method is evaluated on TN3k and shows outstanding performance compared with existing state-of-the-art algorithms.

## Citing
If you find this work useful in your research, please consider citing:

```BibTex
@inproceedings{gong2021multi-task,  
author={Gong, Haifan and Chen, Guanqi and Wang, Ranran and Xie, Xiang and Mao, Mingzhi and Yu, Yizhou and Chen, Fei and Li, Guanbin},  
booktitle={2021 IEEE 18th International Symposium on Biomedical Imaging (ISBI)},   
title={Multi-Task Learning For Thyroid Nodule Segmentation With Thyroid Region Prior},   
year={2021}, 
pages={257-261},  
doi={10.1109/ISBI48211.2021.9434087}
}
```

## Architecture
![Overview](./picture/overview.png)
Overview of the proposed TRFE-Net for thyroid nodule segmentation. 

![pgfe](./picture/rpg.png)
Overview of the proposed RPG modules. 

### License
This code is released under the MIT License (refer to the LICENSE file for details).

## Instructions for Code:
### Requirements
python-3.7
pytorch-1.5.1
torchvision-0.6.1
cuda-10.1
gpu: TitanX 12GB

### Dataset and pretrained model
TN3K dataset: https://pan.baidu.com/s/1-_P_uETgkKs8CsCyVrKYCw with extract code: 49o8.

The pre-trained models can be downloaded from https://pan.baidu.com/s/1i2srV0hIu_a9NLgkm1mpbw with extract code: dol7.

### Training and Evaluation
Training script.
```bash
python train.py -fold 0 -model_name trfe -dataset TATN -gpu 0
``` 

Evaluation script.
```bash
python eval.py -model_name trfe -load_path './run/run_0/trfe_best.pth'
```

### Evaluation result
The evaluation results on 5 folds.

| Models        | fold 1 | fold 2 | fold 3 | fold 4 | fold 5 | Jaccard-avg | Jaccard-std |
|---------------|--------|--------|--------|--------|--------|-------------|-------------|
| FCN           | 65.50  | 65.59  | 64.30  | 65.74  | 65.85  |    65.40    |    0.56     |
| Segnet        | 62.39  | 62.17  | 62.22  | 61.84  | 61.19  |    61.96    |    0.43     |
| Deeplabv3+    | 53.99  | 54.45  | 53.40  | 54.23  | 53.36  |    53.89    |    0.44     |
| Unet          | 63.22  | 64.38  | 63.19  | 63.35  | 63.39  |    63.51    |    0.44     |
| Pretrain-Unet | 63.69  | 64.48  | 64.30  | 62.94  | 64.06  |    63.89    |    0.55     |
| MT-net        | 67.88  | 67.28  | 68.18  | 68.28  | 67.93  |    67.91    |    0.35     |
| TRFE-1        | 68.82  | 68.03  | 68.02  | 68.93  | 68.40  |    68.44    |    0.38     |
| TRFE-2        | 68.30  | 68.39  | 67.90  | 68.30  | 68.00  |    68.18    |    0.19     |
| TRFE-3        | 67.81  | 68.10  | 68.18  | 67.55  | 67.26  |    67.78    |    0.34     |

note: the deeplabv3+ is trained without ImageNet pretrain-weight.
