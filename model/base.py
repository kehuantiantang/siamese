import warnings
import os
from keras.layers import Conv2D, ZeroPadding2D, Activation, Input, concatenate
from keras.models import Model
from model.base_siamese import base_middle_siamese
from model.layer_initialization import top_layer


def get_model(input_shape, pre_train=False, backbone=None):
    left_input = Input(input_shape)
    right_input = Input(input_shape)

    if not backbone:
        backbone_fun = base_middle_siamese
        backbone = 'base_siamese'
    elif backbone == 'vgg16':
        from model.vgg16 import VGG16
        backbone_fun = VGG16
    elif backbone == 'vgg19':
        from model.vgg19 import VGG19
        backbone_fun = VGG19
    elif backbone == 'resnet50':
        from model.resnet import ResNet50
        backbone_fun = ResNet50
    elif backbone == 'resnet101':
        from model.resnet import ResNet101
        backbone_fun = ResNet101
    elif backbone == 'inceptionv2':
        from model.inception_resnet_v2 import InceptionResNetV2
        backbone_fun = InceptionResNetV2
    else:
        raise ValueError('{} is invalid'.format(backbone))

    if pre_train:
        weights = 'imagenet'
    else:
        weights = None
    siamese = backbone_fun(input_shape=input_shape, weights=weights)
    siamese.summary()

    left_output = siamese(left_input)
    right_output = siamese(right_input)

    x = top_layer([left_output, right_output])

    model = Model(inputs=[left_input, right_input], outputs=x,
                  name=siamese.name)

    return model


if __name__ == '__main__':
    from keras.utils import plot_model

    model = get_model((105, 105, 1), backbone=None, pre_train=True)
    plot_model(model)
