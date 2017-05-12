#!/bin/bash

# Working directory
WORKING_DIR=$HOME/projects

# Where the training (fine-tuned) checkpoint and logs will be saved to.
TRAIN_DIR=$WORKING_DIR/dcgan.tensorflow/exp1

batch=$1

#CUDA_VISIBLE_DEVICES=2 \
python generate.py \
    --checkpoint_dir=${TRAIN_DIR} \
    --checkpoint_step=-1 \
    --batch_size=$batch \
    --seed=12345 \
    --make_gif=False \
    --save_step=2000 \

#convert -delay 30 -loop 0 *.jpg generated_images.gif
