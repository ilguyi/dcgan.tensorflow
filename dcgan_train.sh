#!/bin/bash


# Where the training (fine-tuned) checkpoint and logs will be saved to.
TRAIN_DIR=$HOME/projects/tensorflow/gan/dcgan/exp5

# Where the dataset is saved to.
#DATASET_DIR=$HOME/projects/tensorflow/gan/dcgan/datasets/math_data
DATASET_DIR=$HOME/projects/tensorflow/gan/dcgan/datasets/celebA/tfrecords
#DATASET_DIR=$HOME/projects/datasets/imagenet-data/TFRecords

# Where the dataset is saved to.
#LABELS_FILE_PATH=$HOME/projects/datasets/homeware_data/homeware_labels_123.txt


CUDA_VISIBLE_DEVICES=0 \
python train.py \
    --train_dir=${TRAIN_DIR} \
    --dataset_dir=${DATASET_DIR} \
    --initial_learning_rate=0.0002 \
    --num_epochs_per_decay=5 \
    --learning_rate_decay_factor=0.9 \
    --batch_size=128 \
    --num_examples=202599 \
    --max_steps=50000 \
    --save_steps=2000 \
    --adam_beta1=0.5 \
    --image_size=64 \

    # ImageNet
    #--num_examples=1281167 \
    # about 24.98 epochs

    # celebA
    #--num_examples=202599 \
    #--max_steps=50000 \
    # about 31.59 epochs

