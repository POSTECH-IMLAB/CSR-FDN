# CSR-FDN

This is an official implementation of Accurate Lightweight Super-Resolution by Color Separation and Feature Decomposition.

Paper URL : https://drive.google.com/file/d/1Fu0VJDdj-OrCMYPfR1ucGrJTT9sDhZaY/view?usp=sharing


## Abstract

In lightweight super-resolution (SR) task, it is important to utilize a network parameters efficiently because lightweight SR aims enhancing reconstruction quality of super-resolved images with small number of parameters. This thesis proposes the feature decomposition method to efficiently use a network parameters. The feature decomposition module classifies features into two parts, one is hard to be reconstructed and the other is to be reconstructed, using attention mechanism. Then, we assign more parameters to compute hard features than easy features. This enables a network to reduce number of parameters about half without performance degradation. We also propose the color separated restoration method for lightweight SR to enhance restoration quality. We assume that it is too difficult to restore R, G, and B color channels at once from color aggregated feature map for lightweight networks because of its limited number of parameters. Proposed color separated restoration method converts the SR task from three to three color mapping to one to one mapping by separating each color channels. However, if there is no connection between colors, a SR network cannot utilize whole information in an image. Thus, the color separated restoration method partially fuses separated color features through color feature fusion layer to leverage information from other colors. Extensive experimental results show the novelty of our methods over other state-of-the-art lightweight SR methods. Especially, the feature decomposition module and the color separated restoration applied network, namely CSR-FDN, achieves superior performances on three out of four benchmark datasets with scale factor of 4.

## Network architecture
<img src="https://github.com/POSTECH-IMLAB/CSR-FDN/blob/main/fig/fdn.JPG" width="400" height="400" align="middle"/> <img src="https://github.com/POSTECH-IMLAB/CSR-FDN/blob/main/fig/csr-fdn.JPG" width="450" height="400" align="middle"/>

## Quantitative results
<img src="https://github.com/POSTECH-IMLAB/CSR-FDN/blob/main/fig/quantitative.PNG" width="400" height="400" align="middle"/>

## Qualitative results
<img src="https://github.com/POSTECH-IMLAB/CSR-FDN/blob/main/fig/qualitative.PNG" width="400" height="200" align="middle"/>

## Dependenices
* python3
* pytorch >= 1.6 (conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch)
* CUDA = 10.2
* Python packages: pip3 install numpy opencv-python tqdm scikit-image Pillow matplotlib scipy imageio ptflops

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
See option.py for default hyperparameter configuration
```bash
python3 code/main_train.py
```

## Test
See option.py for default hyperparameter configuration
```bash
python3 code/main_test.py
```

## Argument in option.py
```
optional arguments:
  --debug		                Enables debug mode
  --template		TEMPLATE   	You can set various templates in option.py
  --n_threads 		N_THREADS	number of threads for data loading
  --cpu			CPU		use cpu only
  --n_GPUs		N_GPUS		number of GPUs
  --seed		SEED		random seed
  --dir_data		DIR_DATA   	dataset directory
  --dir_demo 		DIR_DEMO   	demo image directory
  --data_train 		DATA_TRAIN	train dataset name
  --data_test 		DATA_TEST	test dataset name
  --n_train 		N_TRAIN     	number of training set
  --n_val		N_VAL		number of validation set
  --offset_val 		OFFSET_VAL	validation index offest
  --ext			EXT		dataset file extension
  --scale 		SCALE           super resolution scale
  --patch_size 		PATCH_SIZE	input patch size
  --rgb_range 		RGB_RANGE	maximum value of RGB
  --n_colors 		N_COLORS   	number of color channels to use
  --model 		MODEL      	model name
  --pre_train 		PRE_TRAIN	pre-trained model path
  --precision 		{single,half}	FP precision for test (single | half)
  --reset        	 		reset the training
  --test_every 		TEST_EVERY	do test per every N batches
  --epochs 		EPOCHS          number of epochs to train
  --batch_size 		BATCH_SIZE	input batch size for training
  --split_batch 	SPLIT_BATCH	split the batch into smaller chunks
  --self_ensemble       		use self-ensemble method for test
  --test_only 		TEST_ONLY	set this option to test the model
  --lr 		        LR              learning rate
  --lr_decay 		LR_DECAY  	learning rate decay per N epochs
  --decay_type 		DECAY_TYPE	learning rate decay type
  --gamma 		GAMMA           learning rate decay factor for step decay
  --weight_decay 	WEIGHT_DECAY	weight decay
  --loss 		LOSS           	loss function configuration
  --save 		SAVE           	file name to save
  --load 		LOAD           	file name to load
  --print_model         		print model
  --save_models        			save all intermediate models
  --print_every 	PRINT_EVERY	how many batches to wait before logging training status
  --save_results 	SAVE_RESULTS	save output results
  --testpath 		TESTPATH   	dataset directory for testing
  --testset 		TESTSET     	dataset name for testing
  --start_epoch 	START_EPOCH	resume from the snapshot, and the start_epoch
  --no_augment 		NO_AUGMENT	do not use data augmentation
```
