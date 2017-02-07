# Deep Convolutional Generative Adversarial Networks
##  implementation based on http://arxiv.org/abs/1511.06434
  "Unsupervised Representation Learning with
  Deep Convolutional Generative Adversarial Networks"
  Alec Radford, Luke Metz and Soumith Chintala

# Author
  Il Gu Yi

# TensorFlow rc1.0-alpha

# Usage
## Training
### dataset download (celebA)

### editing dcgan_train.sh
* Set the "WORKING_DIR path" to path you want
* Set the dataset path or move to where dcgan/datasets/celebA/tfrecords
* Set the hyper-parameters

### run ./dcgan_train.sh
* You can use tensorboard for monitoring loss and generated images
```shell
$ tensorboard --logdir=exp1
```

## Generating images
### 

