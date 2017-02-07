#!/bin/bash

# Working directory
WORKING_DIR=$HOME/projects

# Where the training (fine-tuned) checkpoint and logs will be saved to.
TRAIN_DIR=$HOME/dcgan.tensorflow/exp1

batch=$1

CUDA_VISIBLE_DEVICES=2 \
python generate.py \
    --checkpoint_path=${TRAIN_DIR} \
    --checkpoint_step=0 \
    --batch_size=$batch \
    --dataset_dir=${DATASET_DIR} \
    --seed=12345 \
    --make_gif=True \
    --save_step=2000 \

