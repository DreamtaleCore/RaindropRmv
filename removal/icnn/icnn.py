import numpy as np
import tensorflow as tf
from tensorflow import keras


class ICNN:
    def __init__(self, inputs, is_train=True):
        self.is_train = is_train
        self.relu = keras.activations.relu
        self.outputs = self.net(inputs)

    def net(self, inputs):
        with tf.name_scope("ICNN"):
            x = keras.layers.concatenate(inputs)

            x = keras.layers.Conv2D(32, 3, padding='same', use_bias=False)(x)
            x = keras.layers.BatchNormalization()(x)
            x = keras.layers.LeakyReLU()(x)

            x = keras.layers.Conv2D(64, 3, 2, padding='same', use_bias=False)(x)
            x = keras.layers.BatchNormalization()(x)
            x = keras.layers.LeakyReLU()(x)

            x = keras.layers.Conv2D(256, 1, padding='same', use_bias=False)(x)

            for i in range(13):
                x = self._residual(x)

            x = keras.layers.Conv2D(64, 1, padding='same', use_bias=False)(x)
            x = keras.layers.BatchNormalization()(x)
            x = keras.layers.LeakyReLU()(x)

            x = keras.layers.Conv2DTranspose(32, 3, 2, padding='same', use_bias=False)(x)
            x = keras.layers.BatchNormalization()(x)
            x = keras.layers.LeakyReLU()(x)

            x = keras.layers.Conv2D(3, 3, padding='same', use_bias=False,
                                    activation=self.relu)(x)

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
