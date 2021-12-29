# CSR-FDN

This is an official implementation of Accurate Lightweight Super-Resolution by Color Separation and Feature Decomposition.
Paper : https://drive.google.com/file/d/1Fu0VJDdj-OrCMYPfR1ucGrJTT9sDhZaY/view?usp=sharing


## Abstract

In lightweight super-resolution (SR) task, it is important to utilize a network parameters efficiently because lightweight SR aims enhancing reconstruction quality of super-resolved images with small number of parameters. This thesis proposes the feature decomposition method to efficiently use a network parameters. The feature decomposition module classifies features into two parts, one is hard to be reconstructed and the other is to be reconstructed, using attention mechanism. Then, we assign more parameters to compute hard features than easy features. This enables a network to reduce number of parameters about half without performance degradation. We also propose the color separated restoration method for lightweight SR to enhance restoration quality. We assume that it is too difficult to restore R, G, and B color channels at once from color aggregated feature map for lightweight networks because of its limited number of parameters. Proposed color separated restoration method converts the SR task from three to three color mapping to one to one mapping by separating each color channels. However, if there is no connection between colors, a SR network cannot utilize whole information in an image. Thus, the color separated restoration method partially fuses separated color features through color feature fusion layer to leverage information from other colors. Extensive experimental results show the novelty of our methods over other state-of-the-art lightweight SR methods. Especially, the feature decomposition module and the color separated restoration applied network, namely CSR-FDN, achieves superior performances on three out of four benchmark datasets with scale factor of 4.

<img src="https://github.com/POSTECH-IMLAB/CSR-FDN/tree/main/fig/fdn.jpg" width="800" height="300" align="middle"/>
<img src="https://github.com/POSTECH-IMLAB/CSR-FDN/tree/main/fig/csr-fdn.jpg" width="800" height="300" align="middle"/>

## Dependenices

* python3
* pytorch >= 1.6
* NVIDIA GPU + CUDA
* Python packages: pip3 install numpy opencv-python tqdm scikit-image Pillow

## Pretrained Weights
Pretrained weights are saved in code/experiment/CSR_FDN/

## Dataset Preparation
We use [DIV2K](https://data.vision.ee.ethz.ch/cvl/DIV2K/) as our training datasets. 

For evaluation, we use four datasets, i.e., [Set5](https://uofi.box.com/shared/static/kfahv87nfe8ax910l85dksyl2q212voc.zip), [Set14](https://uofi.box.com/shared/static/igsnfieh4lz68l926l8xbklwsnnk8we9.zip), [Urban100](https://uofi.box.com/shared/static/65upg43jjd0a4cwsiqgl6o6ixube6klm.zip), [BSD100](https://uofi.box.com/shared/static/qgctsplb8txrksm9to9x01zfa4m61ngq.zip).

## Demo
```bash
python3 code/demo.py
```

## Train
For single GPU:
```bash
python3 code/main_train.py
```

## Test
```bash
python3 code/main_test.py
```
