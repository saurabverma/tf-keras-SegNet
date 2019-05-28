#!/bin/bash


# Train

# 30,462 train samples
# 5,000 val samples

# Previous plan 1:
# Examine best 'model', 'lr' and 'decay' values over 10K train and all val samples over 10 epochs
# Learning:
# It is useless to run multiple epochs for examining optimal values because variation over the epochs is minimal
# Forgot that there are 30K+ train samples instead of just 10K
# *** Contrary to general understanding, increasing batch_size can actually hinder the computational performance if the program starts running out of memory

# Previous plan 2:
# Examine best 'model', 'lr' and 'decay' values over all train and val samples for only 2 epochs
# Batch size reduced from 25 or 32 down to 16 -> this reduces computational time by half (from ~30 min to ~15 min per epoch)
# Learning:
# Initial weights might be different for all the runs which can potentially give one code better advantage over the other
# *** Initial weights are very important for faster tuning, hence must explore different initial weights before begining the actual training
# *** 'decay' (and most likely, 'model') affects the system in long run, cannot determine its value in just one epoch 

# Current plan:
# Examine best 'model' and 'lr' values over all train and val samples for 3 epochs, but with same initial weights for all (for each model)
# Also, for each model find the best initial weights from a random bunch

# # # # # Running a series of different variations - good for initial suitable param exploration
# # for model in segnet
# for model in segunet
# # for model in segnet segunet
# do
#     # for lr in 0.01
#     # for lr in 1.01 1.02 1.03 1.04 1.05 1.06 1.07 1.08 1.09 1.011 1.012 1.013 1.014 1.015 1.016 1.017 1.018 1.019 1.021 1.022 1.023 1.024 1.025 1.026 1.027 # This line is useful for creating a lot of initial weights, 25 of them, MAKE SURE # epochs and period = 1 AND initial_weights not set
#     for lr in 1.0 0.1 0.01 0.001 0.0001 0.00001 0.000001 # TODO: uncomment this line, set epoch = 3, period = 2, initial_weights and, run again
#     do
#         for decay in 0.0
#         # for decay in 0.33 0.1 0.033 0.01
#         do
#             echo
#             echo "################################################################################"
#             echo
#             echo "Model: "$model", Learning rate: "$lr", Decay rate: "$decay
#             echo

#             python3 train.py \
#             --n_epochs 3 \
#             --period 2 \
#             --initial_weights "../dataset/LIP/trained_models/initial_weights_segunet.hdf5" \
#             --model $model \
#             --lr $lr \
#             --decay $decay \
#             --gpus 2 \
#             --verbose 1 \
#             --batch_size 16 \
#             --epoch_steps 1904 \
#             --val_steps 313 \
#             --save_dir "../dataset/LIP/trained_models/" \
#             --train_list "../dataset/LIP/Original_images/train_id.txt" \
#             --trainimg_dir "../dataset/LIP/Original_images/train_images/" \
#             --trainmsk_dir "../dataset/LIP/Ground_truth_for_original/train_segmentations/" \
#             --val_list "../dataset/LIP/Original_images/val_id.txt" \
#             --valimg_dir "../dataset/LIP/Original_images/val_images/" \
#             --valmsk_dir "../dataset/LIP/Ground_truth_for_original/val_segmentations/"

#         done

#     done

# done





# # # # # Running a lengthy training based on optimization learned from initial exploration
python3 train.py \
--n_epochs 100 \
--period 5 \
--initial_weights "../dataset/LIP/trained_models/model_segnet_batch_size_16_epoch_steps_1220_n_epochs_3_lr_0.01_decay_0.0/weights_at_epoch_003.hdf5" \
--model segnet \
--lr 0.0001 \
--decay 0.001 \
--gpus 2 \
--verbose 1 \
--batch_size 32 \
--save_dir "../dataset/LIP/trained_models/" \
--train_list "../dataset/LIP/Original_images/train_id.txt" \
--trainimg_dir "../dataset/LIP/Original_images/train_images/" \
--trainmsk_dir "../dataset/LIP/Ground_truth_for_original/train_segmentations/" \
--val_list "../dataset/LIP/Original_images/val_id.txt" \
--valimg_dir "../dataset/LIP/Original_images/val_images/" \
--valmsk_dir "../dataset/LIP/Ground_truth_for_original/val_segmentations/"

# # # # # Running with only 2 labels
python3 train.py \
--n_epochs 100 \
--period 5 \
--model fastnet \
--lr 0.00001 \
--decay 0.0 \
--gpus 2 \
--verbose 1 \
--batch_size 32 \
--n_labels 2 \
--save_dir "../dataset/LIP/trained_models/" \
--train_list "../dataset/LIP/Original_images/train_id.txt" \
--trainimg_dir "../dataset/LIP/Original_images/train_images/" \
--trainmsk_dir "../dataset/LIP/Ground_truth_for_original/train_segmentations_2_labels/" \
--val_list "../dataset/LIP/Original_images/val_id.txt" \
--valimg_dir "../dataset/LIP/Original_images/val_images/" \
--valmsk_dir "../dataset/LIP/Ground_truth_for_original/val_segmentations_2_labels/"

# --epoch_steps 1220 \
# --val_steps 313 \

# 30462
# 36, 0938 > ~ 24 min for 30K-*36 => Unable to run, out of memory

# 32, 0938 > ~ 24 min for 30K-95*32 => 26960
# 28, 1072 > ~ 24 min for 30K-*28 => 
# 24, 1270 > ~ 24 min for 30K-108*24 => 27408
# 20, 1523 > ~ 24 min for 30K-108*20 => 27840
# 16, 1904 > ~ 24 min for 30K-150*16 => 27600
# 12, 2538 > ~ 24 min for 30K-*24 => 

python3 train.py \
--batch_size 32 \
--epoch_steps 0938 \
--n_epochs 100 \
--period 5 \
--initial_weights "../dataset/LIP/trained_models/model_segnet_batch_size_16_epoch_steps_1220_n_epochs_3_lr_0.01_decay_0.0/weights_at_epoch_003.hdf5" \
--model segnet \
--lr 0.01 \
--decay 0.001 \
--gpus 2 \
--verbose 1 \
--val_steps 200 \
--save_dir "../dataset/LIP/trained_models/" \
--train_list "../dataset/LIP/Original_images/train_id.txt" \
--trainimg_dir "../dataset/LIP/Original_images/train_images/" \
--trainmsk_dir "../dataset/LIP/Ground_truth_for_original/train_segmentations/" \
--val_list "../dataset/LIP/Original_images/val_id.txt" \
--valimg_dir "../dataset/LIP/Original_images/val_images/" \
--valmsk_dir "../dataset/LIP/Ground_truth_for_original/val_segmentations/"








# Test
# 5,000 test samples
