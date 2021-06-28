import numpy as np
import tensorflow as tf
from tensorflow import keras
from IPython import embed


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

            for i in range(6):
                x = self._residual(x, i)
            x = keras.layers.BatchNormalization()(x)
            x = keras.layers.LeakyReLU()(x)

            self.quarter = keras.layers.Conv2D(64, 1, padding='same', use_bias=False)(self.quarter)
            self.quarter = keras.layers.LeakyReLU()(self.quarter)
            self.quarter = keras.layers.BatchNormalization()(self.quarter)

            x = keras.layers.Conv2D(64, 1, padding='same', use_bias=False)(x)
            x = keras.layers.LeakyReLU()(x)
            x = keras.layers.BatchNormalization()(x)

            x = keras.layers.concatenate([x, self.quarter])
            x = keras.layers.Conv2D(64, 1, padding='same', use_bias=False)(x)
            x = keras.layers.LeakyReLU()(x)
            x = keras.layers.BatchNormalization()(x)

            x = keras.layers.Conv2DTranspose(32, 3, 2, padding='same', use_bias=False)(x)
            x = keras.layers.LeakyReLU()(x)
            x = keras.layers.BatchNormalization()(x)

            self.half = keras.layers.Conv2D(32, 1, padding='same', use_bias=False)(self.half)
            self.half = keras.layers.LeakyReLU()(self.half)
            self.half = keras.layers.BatchNormalization()(self.half)
            x = keras.layers.concatenate([x, self.half])

            x = keras.layers.Conv2D(32, 1, padding='same', use_bias=False)(x)
            x = keras.layers.LeakyReLU()(x)
            x = keras.layers.BatchNormalization()(x)

            x = keras.layers.Conv2DTranspose(16, 3, 2, padding='same', use_bias=False)(x)
            x = keras.layers.LeakyReLU()(x)
            x = keras.layers.BatchNormalization()(x)

            x = keras.layers.Conv2D(3, 3, padding='same', activation=self.relu)(x)

        return x


    def _residual(self, inputs, level):
        if level == 2:
            stride = 2
            self.half = inputs
        else:
            stride = 1

        if level > 2:
            dilated = 2 * (level - 2)
        else:
            dilated = 1
        x = keras.layers.BatchNormalization()(inputs)
        x = keras.layers.LeakyReLU()(x)
        x = keras.layers.Conv2D(64, 1, padding='same', use_bias=False)(x)

        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.LeakyReLU()(x)
        x = keras.layers.Conv2D(64, 3, stride, padding='same', dilation_rate=dilated, use_bias=False)(x)

        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.LeakyReLU()(x)
        x = keras.layers.Conv2D(256, 1, padding='same', use_bias=False)(x)

        if level == 2:
            inputs = keras.layers.Conv2D(256, 1, 2, padding='same', use_bias=False)(inputs)
        x = keras.layers.add([inputs, x])
        if level == 2:
            self.quarter = x
        return x
