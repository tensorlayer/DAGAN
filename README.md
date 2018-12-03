# DAGAN

This is the official implementation code for [DAGAN: Deep De-Aliasing Generative Adversarial Networks for Fast Compressed Sensing MRI Reconstruction](https://ieeexplore.ieee.org/document/8233175/) published in IEEE Transactions on Medical Imaging (2018).  
[Guang Yang](https://www.imperial.ac.uk/people/g.yang)\*, [Simiao Yu](https://nebulav.github.io/)\*, et al.  
(* equal contributions) 

If you use this code for your research, please cite our paper.

```
@article{yang2018_dagan,
	author = {Yang, Guang and Yu, Simiao and Dong, Hao and Slabaugh, Gregory G. and Dragotti, Pier Luigi and Ye, Xujiong and Liu, Fangde and Arridge, Simon R. and Keegan, Jennifer and Guo, Yike and Firmin, David N.},
	journal = {IEEE Trans. Med. Imaging},
	number = 6,
	pages = {1310--1321},
	title = {{DAGAN: deep de-aliasing generative adversarial networks for fast compressed sensing MRI reconstruction}},
	volume = 37,
	year = 2018
}
```

If you have any questions about this code, please feel free to contact Simiao Yu (simiao.yu13@imperial.ac.uk).

# Prerequisites

The original code is in python 3.5 under the following dependencies:
1. tensorflow (v1.1.0)
2. tensorlayer (v1.7.2)
3. easydict (v1.6)
4. nibabel (v2.1.0)
5. scikit-image (v0.12.3)

Code tested in Ubuntu 16.04 with Nvidia GPU + CUDA CuDNN (whose version is compatible to tensorflow v1.1.0).

# How to use

1. Prepare data

    1) Data used in this work are publicly available from the MICCAI 2013 grand challenge ([link](https://my.vanderbilt.edu/masi/workshops/)). We refer users to register with the grand challenge organisers to be able to download the data.
    2) Download training and test data respectively into data/MICCAI13_SegChallenge/Training_100 and data/MICCAI13_SegChallenge/Testing_100 (We randomly included 100 T1-weighted MRI datasets for training and 50 datasets for testing)
    3) run 'python data_loader.py'
    4) after running the code, training/validation/testing data should be saved to 'data/MICCAI13_SegChallenge/' in pickle format.

2. Download pretrained VGG16 model

    1) Download 'vgg16_weights.npz' from [this link](http://www.cs.toronto.edu/~frossard/post/vgg16/)
    2) Save 'vgg16_weights.npz' into 'trained_model/VGG16'
    
3. Train model
    1) run 'CUDA_VISIBLE_DEVICES=0 python train.py --model MODEL --mask MASK --maskperc MASKPERC' where you should specify MODEL, MASK, MASKPERC respectively:
    - MODEL: choose from 'unet' or 'unet_refine'
    - MASK: choose from 'gaussian1d', 'gaussian2d', 'poisson2d'
    - MASKPERC: choose from '10', '20', '30', '40', '50' (percentage of mask)
 
4. Test trained model
    
    1) run 'CUDA_VISIBLE_DEVICES=0 python test.py --model MODEL --mask MASK --maskperc MASKPERC' where you should specify MODEL, MASK, MASKPERC respectively (as above).

# Results

Please refer to the paper for the detailed results.
