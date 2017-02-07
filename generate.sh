#!/bin/bash


# Where the training (fine-tuned) checkpoint and logs will be saved to.
TRAIN_DIR=$HOME/projects/tensorflow/gan/dcgan/exp1

# Where the dataset is saved to.
DATASET_DIR=$HOME/projects/tensorflow/gan/dcgan/datasets/celebA/tfrecords


step=0
batch=$1
#step=20000
#batch=16

CUDA_VISIBLE_DEVICES=2 \
python generate.py \
    --checkpoint_path=${TRAIN_DIR} \
    --checkpoint_step=$step \
    --batch_size=$batch \
    --dataset_dir=${DATASET_DIR} \
    --seed=12345 \
    --make_gif=True \
    --save_step=2000 \

