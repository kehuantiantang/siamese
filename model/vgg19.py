# coding=utf-8

"""VGG19 model for Keras.

# Reference

- [Very Deep Convolutional Networks for Large-Scale Image Recognition](
    https://arxiv.org/abs/1409.1556) (ICLR 2015)

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from model.layer_initialization import Conv2D_Initialize, Dense_Initialize

WEIGHTS_PATH = ('https://github.com/fchollet/deep-learning-models/'
                'releases/download/v0.1/'
                'vgg19_weights_tf_dim_ordering_tf_kernels.h5')
WEIGHTS_PATH_NO_TOP = ('https://github.com/fchollet/deep-learning-models/'
                       'releases/download/v0.1/'
                       'vgg19_weights_tf_dim_ordering_tf_kernels_notop.h5')

import keras.backend as backend
from keras import layers, models, Input, Model
from keras import utils as keras_utils

def VGG19(input_shape, include_top=True,
          weights='imagenet',
          pooling=None,
          classes=1000,
          final_activation = 'sigmoid',
          **kwargs):

    input = Input(input_shape)
    # Block 1
    x = Conv2D_Initialize(64, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block1_conv1', bias_initializer='zero')(input)
    x = Conv2D_Initialize(64, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block1_conv2')(x)
    x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)

    # Block 2
    x = Conv2D_Initialize(128, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block2_conv1')(x)
    x = Conv2D_Initialize(128, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block2_conv2')(x)
    x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)

    # Block 3
    x = Conv2D_Initialize(256, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block3_conv1')(x)
    x = Conv2D_Initialize(256, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block3_conv2')(x)
    x = Conv2D_Initialize(256, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block3_conv3')(x)
    x = Conv2D_Initialize(256, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block3_conv4')(x)
    x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)

    # Block 4
    x = Conv2D_Initialize(512, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block4_conv1')(x)
    x = Conv2D_Initialize(512, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block4_conv2')(x)
    x = Conv2D_Initialize(512, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block4_conv3')(x)
    x = Conv2D_Initialize(512, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block4_conv4')(x)
    x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)

    # Block 5
    x = Conv2D_Initialize(512, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block5_conv1')(x)
    x = Conv2D_Initialize(512, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block5_conv2')(x)
    x = Conv2D_Initialize(512, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block5_conv3')(x)
    x = Conv2D_Initialize(512, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block5_conv4')(x)
    x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)

    if include_top:
        # Classification block
        x = layers.Flatten(name='flatten')(x)
        x = Dense_Initialize(4096, activation='relu', name='fc1')(x)
        x = Dense_Initialize(4096, activation='relu', name='fc2')(x)
        x = Dense_Initialize(classes, activation=final_activation, name='predictions')(x)
    else:
        if pooling == 'avg':
            x = layers.GlobalAveragePooling2D()(x)
        elif pooling == 'max':
            x = layers.GlobalMaxPooling2D()(x)


    # Load weights.
    weights_path = None
    if weights == 'imagenet':
        if include_top:
            weights_path = keras_utils.get_file(
                'vgg19_weights_tf_dim_ordering_tf_kernels.h5',
                WEIGHTS_PATH,
                cache_subdir='models',
                file_hash='cbe5617147190e668d6c5d5026f83318')
        else:
            weights_path = keras_utils.get_file(
                'vgg19_weights_tf_dim_ordering_tf_kernels_notop.h5',
                WEIGHTS_PATH_NO_TOP,
                cache_subdir='models',
                file_hash='253f8cb515780f3b799900260a226db6')


    model = Model(input, x, name = 'vgg19')
    if weights_path and weights:
        model.load_weights(weights_path, by_name=True, skip_mismatch=True)
    return model
