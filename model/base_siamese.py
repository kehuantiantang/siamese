# coding=utf-8
from keras import Model, Input
from keras.layers import MaxPooling2D, Flatten

from model.layer_initialization import Conv2D_Initialize, Dense_Initialize


def base_middle_siamese(input_shape, final_activation='sigmoid', **kwargs):
    input = Input(shape=input_shape)
    x = Conv2D_Initialize(64, (10, 10),
                          activation='relu', bias_initializer='zero')(input)
    x = MaxPooling2D()(x)
    x = Conv2D_Initialize(128, (7, 7), activation='relu')(x)
    x = MaxPooling2D()(x)
    x = Conv2D_Initialize(128, (4, 4), activation='relu')(x)
    x = MaxPooling2D()(x)
    x = Conv2D_Initialize(256, (4, 4), activation='relu')(x)
    x = Flatten()(x)
    out = Dense_Initialize(4096, activation=final_activation)(x)

    model = Model(input, out, name='siamese')
    return model


# def get_siamese_model(input_shape):
#     """
#         Model architecture based on the one provided in: http://www.cs.utoronto.ca/~gkoch/files/msc-thesis.pdf
#     """
#
#     # Define the tensors for the two input images
#     left_input = Input(input_shape)
#     right_input = Input(input_shape)
#
#     # Convolutional Neural Network
#     model = Sequential()
#     model.add(Conv2D(64, (10, 10), activation='relu', input_shape=input_shape,
#                      kernel_initializer=initialize_weights,
#                      kernel_regularizer=l2(2e-4)))
#     model.add(MaxPooling2D())
#     model.add(Conv2D(128, (7, 7), activation='relu',
#                      kernel_initializer=initialize_weights,
#                      bias_initializer=initialize_bias,
#                      kernel_regularizer=l2(2e-4)))
#     model.add(MaxPooling2D())
#     model.add(Conv2D(128, (4, 4), activation='relu',
#                      kernel_initializer=initialize_weights,
#                      bias_initializer=initialize_bias,
#                      kernel_regularizer=l2(2e-4)))
#     model.add(MaxPooling2D())
#     model.add(Conv2D(256, (4, 4), activation='relu',
#                      kernel_initializer=initialize_weights,
#                      bias_initializer=initialize_bias,
#                      kernel_regularizer=l2(2e-4)))
#     model.add(Flatten())
#     model.add(Dense(4096, activation='sigmoid', kernel_regularizer=l2(1e-3),
#                     kernel_initializer=initialize_weights,
#                     bias_initializer=initialize_bias))
#
#     # Generate the encodings (feature vectors) for the two images
#     encoded_l = model(left_input)
#     encoded_r = model(right_input)
#
#     # Add a customized layer to compute the absolute difference between the encodings
#     L1_layer = Lambda(lambda tensors: K.abs(tensors[0] - tensors[1]))
#     L1_distance = L1_layer([encoded_l, encoded_r])
#     print(L1_distance.shape)
#     # Add a dense layer with a sigmoid unit to generate the similarity score
#     prediction = Dense(1, activation='sigmoid',
#                        bias_initializer=initialize_bias)(L1_distance)
#
#     # Connect the inputs with the outputs
#     siamese_net = Model(inputs=[left_input, right_input], outputs=prediction)
#
#     # return the model
#     return siamese_net