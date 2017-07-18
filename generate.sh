#!/bin/bash

# Working directory
WORKING_DIR=$HOME/projects

# Where the training (fine-tuned) checkpoint and logs will be saved to.
TRAIN_DIR=$WORKING_DIR/dcgan.tensorflow/exp1

BATCH_SIZE=$1

#CUDA_VISIBLE_DEVICES=2 \
python generate.py \
    --checkpoint_dir=${TRAIN_DIR} \
    --checkpoint_step=-1 \
    --initial_learning_rate=0.0002 \
    --adam_beta1=0.5 \
    --batch_size=$BATCH_SIZE \
    --seed=12345 \
    --make_gif=True \

#convert -delay 30 -loop 0 *.jpg generated_images.gif
