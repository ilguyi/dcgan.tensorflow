#!/bin/bash

# Working directory
WORKING_DIR=$HOME/projects

# Where the training (fine-tuned) checkpoint and logs will be saved to.
TRAIN_DIR=$WORKING_DIR/dcgan.tensorflow/exp1

batch=$1

#CUDA_VISIBLE_DEVICES=2 \
python generate.py \
    --checkpoint_path=${TRAIN_DIR} \
    --checkpoint_step=-1 \
    --batch_size=$batch \
    --seed=12345 \
    --make_gif=True \
    --save_step=2000 \

convert -delay 20 -loop *.jpg generated_images.gif
