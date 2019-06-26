# coding=utf-8
from keras.layers import Conv2D, np, Dense, Lambda
from keras.regularizers import l2
import keras.backend as K

def initialize_weights(shape, name=None):
    """
        The paper, http://www.cs.utoronto.ca/~gkoch/files/msc-thesis.pdf
        suggests to initialize CNN layer weights with mean as 0.0 and standard deviation of 0.01
    """
    return np.random.normal(loc=0.0, scale=1e-2, size=shape)



def initialize_bias(shape, name=None):
    """
        The paper, http://www.cs.utoronto.ca/~gkoch/files/msc-thesis.pdf
        suggests to initialize CNN layer bias with mean as 0.5 and standard deviation of 0.01
    """
    return np.random.normal(loc=0.5, scale=1e-2, size=shape)

def initialize_dense_weights(shape, dtype=None):
    return np.random.normal(loc=0, scale=2e-2, size=shape)

def Conv2D_Initialize(filters, kernel_size, **kwargs):


    kernel_initializer = kwargs.pop('kernel_initializer', initialize_weights)
    bias_initializer = kwargs.pop('bias_initializer', initialize_bias)
    kernel_regularizer = kwargs.pop('kernel_regularizer', l2(2e-4))

    # kwargs.pop('kernel_initializer')
    # kwargs.pop('bias_initializer')
    # kwargs.pop('kernel_regularizer')

    def layer(x):
        return Conv2D(filters, kernel_size,
                      kernel_initializer=kernel_initializer,
                      bias_initializer=bias_initializer,
                      kernel_regularizer=kernel_regularizer, **kwargs)(x)

    return layer


def Dense_Initialize(units, **kwargs):
    bias_initializer = kwargs.pop('bias_initializer', initialize_bias)
    kernel_initializer = kwargs.pop('kernel_initializer', initialize_dense_weights)
    def layer(x):
        # wider normal distribution, normalization distribution: 0, 0.2
        return Dense(units, kernel_initializer = kernel_initializer,
        bias_initializer=bias_initializer, **kwargs)(x)

    return layer


def top_layer(x_list, final_activation='sigmoid'):
    L1_layer = Lambda(lambda tensors: K.abs(tensors[0] - tensors[1]))
    L1_distance = L1_layer(x_list)
    print(L1_distance.shape)
    # Add a dense layer with a sigmoid unit to generate the similarity score
    prediction = Dense_Initialize(1, activation=final_activation,
                       name='siamese_prediction')(L1_distance)

    return prediction
