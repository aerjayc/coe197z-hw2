from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from keras.layers import Dense, Input
from keras.layers import Conv2D, Flatten
from keras.layers import Reshape, Conv2DTranspose
from keras.models import Model
from keras import backend as K


def make_autoencoder(input_shape, layer_filters, kernel_size, latent_dim,
                     strides, padding, activation='relu',
                     loss='mse', optimizer='adam'):
    # encoder
    inputs = Input(shape=input_shape, name='encoder_input')
    x = inputs
    for filters in layer_filters:
        x = Conv2D(filters=filters, kernel_size=kernel_size,
                strides=strides, padding=padding,
                activation=activation)(x)

    # latent vector
    shape = K.int_shape(x)

    x = Flatten()(x)
    latent = Dense(latent_dim, name='latent_vector')(x)
    encoder = Model(inputs, latent, name='encoder')

    # decoder
    latent_inputs = Input(shape=(latent_dim,), name='decoder_input')
    x = Dense(shape[1] * shape[2] * shape[3])(latent_inputs)
    x = Reshape((shape[1], shape[2], shape[3]))(x)
    for filters in layer_filters[::-1]:
        x = Conv2DTranspose(filters=filters, kernel_size=kernel_size,
                            strides=strides, padding=padding,
                            activation=activation)(x)
    
    outputs = Conv2DTranspose(filters=1, kernel_size=kernel_size,
                              padding='same', activation='sigmoid',
                              name='decoder_output')(x)

    decoder = Model(latent_inputs, outputs, name='decoder')


    autoencoder = Model(inputs, decoder(encoder(inputs)), name='autoencoder')
    autoencoder.compile(loss=loss, optimizer=optimizer)

    return autoencoder