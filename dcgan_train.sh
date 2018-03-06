#!/bin/bash

# Working directory
WORKING_DIR=$HOME/projects

# Where the training (fine-tuned) checkpoint and logs will be saved to.
TRAIN_DIR=$WORKING_DIR/dcgan.tensorflow/train/exp1

# Where the dataset is saved to.
DATASET_DIR=$WORKING_DIR/datasets/celebA/tfrecords

CUDA_VISIBLE_DEVICES=0 \
python train.py \
    --train_dir=${TRAIN_DIR} \
    --dataset_dir=${DATASET_DIR} \
    --initial_learning_rate=0.0002 \
    --num_epochs_per_decay=100 \
    --learning_rate_decay_factor=0.9 \
    --batch_size=128 \
    --num_examples=202599 \
    --max_epochs=30 \
    --save_steps=2000 \
    --adam_beta1=0.5 \

    # celebA
    #--num_examples=202599 \
    #--max_epochs=30 \
    # about 47484 steps

