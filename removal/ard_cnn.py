import numpy as np
import tensorflow as tf
from tensorflow import keras


class ARDCNN:
    def __init__(self, inputs, is_train=True):
        self.is_train = is_train
        self.sigmoid = tf.keras.activations.sigmoid
        self.outputs = self.net(inputs)

    def net(self, inputs):
        with tf.name_scope("ARDCNN"):

            x = keras.layers.Conv2D(32, 3, padding='same', use_bias=False)(inputs)
            x = keras.layers.BatchNormalization()(x)
            x = keras.layers.LeakyReLU()(x)

            #x = keras.layers.Conv2D(64, 3, padding='same', dilation_rate=2, use_bias=False)(x)
            x = keras.layers.Conv2D(64, 3, 2, padding='same', use_bias=False)(x)
            x = keras.layers.BatchNormalization()(x)
            x = keras.layers.LeakyReLU()(x)

            x = keras.layers.Conv2D(256, 1, padding='same', use_bias=False)(x)

            for i in range(6):
                x = self._residual(x)

            x = keras.layers.Conv2D(64, 1, padding='same', use_bias=False)(x)
            x = keras.layers.BatchNormalization()(x)
            x = keras.layers.LeakyReLU()(x)

            #x = keras.layers.Conv2D(32, 3, padding='same', use_bias=False)(x)
            x = keras.layers.Conv2DTranspose(32, 3, 2, padding='same', use_bias=False)(x)
            x = keras.layers.BatchNormalization()(x)
            x = keras.layers.LeakyReLU()(x)

            x = keras.layers.Conv2D(1, 3, padding='same', use_bias=False,
                                    activation=self.sigmoid)(x)

        return x


    def _residual(self, inputs):
        x = keras.layers.BatchNormalization()(inputs)
        x = keras.layers.LeakyReLU()(x)
        x = keras.layers.Conv2D(64, 1, padding='same', use_bias=False)(x)

        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.LeakyReLU()(x)
        x = keras.layers.Conv2D(64, 3, padding='same', use_bias=False)(x)

        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.LeakyReLU()(x)
        x = keras.layers.Conv2D(256, 1, padding='same', use_bias=False)(x)

        x = keras.layers.add([inputs, x])

        return x
