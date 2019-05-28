Basics from
UNET: `https://github.com/zhixuhao/unet.git`
SEGNET: `https://github.com/ykamikawa/SegNet.git`
SEG-UNET: `https://github.com/ykamikawa/Seg-UNet.git`
FastNet: `https://github.com/johnolafenwa/FastNet.git`

# Update

1. Fine tunining for ease in use
2. Multiple GPUs usage
3. Test (separate from train)

# SegNet

SegNet is a model of semantic segmentation based on Fully Comvolutional Network.

This repository contains the implementation of learning and testing in keras and tensorflow.
Also included is a custom layer implementation of index pooling, a new property of segnet.

# Seg-UNet(SegNet + UNet)
SegUNet is a model of semantic segmentation based on SegNet and UNet(these model are based on Fully Convolutional Network).
Architecture dedicated to restoring pixel position information.
This architecture is good at fine edge restoration etc.

This repository contains the implementation of learning and testing in keras and tensorflow.

This architecture is encoder-decoder model(29 conv2D layers).
- Skip connection(UNet) and indeces pooling(SegNet) are incorporated to propagate the spatial information of the image.

## Architecture
- encoder decoder architecture
- fully convolutional network
- indices pooling

    ![indicespooling](https://user-images.githubusercontent.com/27678705/33704612-81053eec-db70-11e7-9822-01dd48d68314.png)

## Description
This repository is SegNet architecture for Semantic Segmentation.
The repository of other people's segmentation, pooling with indices not implemented.But In this repository we implemented  pooling layer and unpooling layer with indices at MyLayers.py.

Segnet architecture is early Semantic Segmentation model, so acccuracy is low but fast.
In the future, we plan to implement models with high accuracy.(UNet,PSPNet,Pix2Pix etc.)

## Usage

### train

`
python3 train.py \
--model "segunet" \
--gpus 2 \
--batch_size 16 \
--verbose 1 \
--n_epochs 10 \
--epoch_steps 400 \
--val_steps 200 \
--lr 0.1 \
--decay 0.01 \
--initial_weights "../dataset/LIP/trained_models/initial_weights.hdf5" \
--save_dir "../dataset/LIP/trained_models/" \
--train_list "../dataset/LIP/Original_images/train_id.txt" \
--trainimg_dir "../dataset/LIP/Original_images/train_images/" \
--trainmsk_dir "../dataset/LIP/Ground_truth_for_original/train_segmentations/" \
--val_list "../dataset/LIP/Original_images/val_id.txt" \
--valimg_dir "../dataset/LIP/Original_images/val_images/" \
--valmsk_dir "../dataset/LIP/Ground_truth_for_original/val_segmentations/"
`

### test

`
python3 test.py \
--model "segunet" \
--gpus 2 \
--batch_size 16 \
--verbose 1 \
--epoch_steps 200 \
--save_path "../dataset/LIP/trained_models/model_segunet_batch_size_25_epoch_steps_400_n_epochs_3_lr_0.01_decay_0.01/test_results/" \
--model_weights "../dataset/LIP/trained_models/model_segunet_batch_size_25_epoch_steps_400_n_epochs_3_lr_0.01_decay_0.01/weights_at_epoch_010.hdf5" \
--test_list "../dataset/LIP/Original_images/test_id.txt" \
--testimg_dir "../dataset/LIP/Original_images/test_images/" \
--testmsk_dir "../dataset/LIP/Ground_truth_for_original/test_segmentations/"
`

### examine
`
tensorboard --logdir=../dataset/LIP/trained_models/
`

