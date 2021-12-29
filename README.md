# CSR-FDN

This is an official implementation of Accurate Lightweight Super-Resolution by Color Separation and Feature Decomposition 


## Main Results

### Results about Setting 1
| Method | Scale |   Set5  |   Set5   | Set14   |  Set14   | B100   |  B100    | Urban100 | Urban100  | Mangan109 | Manga109  |
|:------:|:-----:|:-------:|:--------:|:-------:|---------|:-------:|----------|:-------:|-----------|:-------:|:-------:|
|        |       | PSNR    | SSIM     | PSNR    | SSIM     | PSNR    | SSIM     | PSNR     | SSIM     | PSNR      | SSIM     |
| IKC    | x2    |  37.19  |  0.9526  |  32.94  |  0.9024  |  31.51  |  0.8790  |  29.85   |  0.8928  |  36.93    |  0.9667  |
| DANv1  | x2    |  37.34  |  0.9526  |  33.08  |  0.9041  |  31.76  |  0.8858  |  30.60   |  0.9060  |  37.23    |  0.9710  |
| DANv2  | x2    |  37.60  |  0.9544  |  33.44  |  0.9094  |  32.00  |  0.8904  |  31.43   |  0.9174  |  38.07    |  0.9734  |
| IKC    | x3    |  33.06  |  0.9146  |  29.38  |  0.8233  |  28.53  |  0.7899  |  27.43   |  0.8302  |  32.43    |  0.9316  |
| DANv1  | x3    |  34.04  |  0.9199  |  30.09  |  0.8287  |  28.94  |  0.7919  |  27.65   |  0.8352  |  33.16    |  0.9382  |
| DANv2  | x3    |  34.19  |  0.9209  |  30.20  |  0.8309  |  29.03  |  0.7948  |  27.83   |  0.8395  |  33.28    |  0.9400  |
| IKC    | x4    |  31.67  |  0.8829  |  28.31  |  0.7643  |  27.37  |  0.7192  |  25.33   |  0.7504  |  28.91    |  0.8782  |
| DANv1  | x4    |  31.89  |  0.8864  |  28.42  |  0.7687  |  27.51  |  0.7248  |  25.86   |  0.7721  |  30.50    |  0.9037  |
| DANv2  | x4    |  32.00  |  0.8885  |  28.50  |  0.7715  |  27.56  |  0.7277  |  25.94   |  0.7748  |  30.45    |  0.9037  |

### Results about Setting 2 (DIV2KRK)

|Method          |  x2   |  x2    |  x4   |  x4    
|:--------------:|:-----:|:------:|:-----:|:------:|
|                | PSNR  | SSIM   | PSNR  |   SSIM |
|KernelGAN + ZSSR| 30.36 | 0.8669 | 26.81 | 0.7316 |
|DANv1           | 32.56 | 0.8997 | 27.55 | 0.7582 |
|DANv2           | 32.58 | 0.9048 | 28.74 | 0.7893 |


## Dependenices

* python3
* pytorch >= 1.6
* NVIDIA GPU + CUDA
* Python packages: pip3 install numpy opencv-python tqdm scikit-image Pillow

## Pretrained Weights


## Dataset Preparation
We use [DIV2K](https://data.vision.ee.ethz.ch/cvl/DIV2K/) as our training datasets. 

For evaluation, we use four datasets, i.e., [Set5](https://uofi.box.com/shared/static/kfahv87nfe8ax910l85dksyl2q212voc.zip), [Set14](https://uofi.box.com/shared/static/igsnfieh4lz68l926l8xbklwsnnk8we9.zip), [Urban100](https://uofi.box.com/shared/static/65upg43jjd0a4cwsiqgl6o6ixube6klm.zip), [BSD100](https://uofi.box.com/shared/static/qgctsplb8txrksm9to9x01zfa4m61ngq.zip).


## Train

For single GPU:
```bash
cd codes/config/DANv1
python3 train.py -opt=options/setting1/train_setting1_x4.yml
```

For distributed training
```bash
cd codes/config/DANv1
python3 -m torch.distributed.launch --nproc_per_node=8 --master_poer=4321 train.py -opt=options/setting1/train_setting1_x4.yml --launcher pytorch
```


## Test on Synthetic Images
```bash
cd codes/config/DANv1
python3 test.py -opt=options/setting1/test_setting1_x4.yml
```

## Test on Real Images
```bash
cd codes/config/DANv1
python3 inference.py -input_dir=/path/to/real/images/ -output_dir=/path/to/save/sr/results/
```
