#!/bin/bash


# Where the training (fine-tuned) checkpoint and logs will be saved to.
TRAIN_DIR=$HOME/projects/tensorflow/gan/dcgan/exp1

# Where the dataset is saved to.
#DATASET_DIR=$HOME/projects/tensorflow/gan/dcgan/datasets/math_data
DATASET_DIR=$HOME/projects/tensorflow/gan/dcgan/datasets/celebA/tfrecords


#step=$1
#batch=$2
step=20000
batch=16
column_index=$1 \

CUDA_VISIBLE_DEVICES=2 \
python generate.py \
    --checkpoint_path=${TRAIN_DIR} \
    --checkpoint_step=$step \
    --batch_size=$batch \
    --dataset_dir=${DATASET_DIR} \
    --seed=12345 \
    --make_gif=False \
    --save_step=2000 \
    --column_index=$column_index \

