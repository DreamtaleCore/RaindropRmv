import tensorflow as tf
import cv2
from tensorflow import keras
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Lambda

def cast_mask(x):
    x = K.greater(x, 0.3)
    x = K.cast(x, tf.float32)
    return x

def subtract(x):
    return 1.0 - x

def dilate(x):
    kernel = cv2.getStructuringElement(2, (3, 3))
    kernel = kernel.reshape((3,3,1))
    stride = [1, 1, 1, 1]

    x = tf.nn.dilation2d(x, kernel, stride, stride, 'SAME')
    x = x - 1.0
    return x


class COMBINE:

    def __init__(self, inputs, is_training=False, mode='inception', dilated=False):
        self.is_training = is_training
        self.relu = keras.activations.relu
        self.net_mode = mode
        self.dilated = dilated

        self.outputs = self.net(inputs)

    def conv2D(self, filters, kernel, x):
        x = keras.layers.Conv2D(filters, kernel, padding='same', use_bias=False)(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.LeakyReLU()(x)
        return x


    def net(self, inputs):
        with tf.variable_scope("COMBINE"):
            clear, mask, rain = inputs
            mask = Lambda(cast_mask)(mask)
            if self.dilated:
                print("-----Use dilated mask!-----")
                mask = Lambda(dilate)(mask)
            smooth = keras.layers.multiply([clear, mask])
            original = keras.layers.multiply([rain, Lambda(subtract)(mask)])
            x = keras.layers.add([smooth, original])

            if self.net_mode == 'inception':
                x = self.__inception(x)
            elif self.net_mode == 'resnet':
                x = self.__resnet(x)
            else:
                raise Exception("unsupport net mode.")

        return x

    def __inception(self, inputs):
        x = self.conv2D(32, 5, inputs)

        ##Inception V4 Block A
        average = keras.layers.AveragePooling2D(strides=1, padding='same')(x)
        average = self.conv2D(16, 1, average)

        oneConv = self.conv2D(16, 1, x)

        first3Conv = self.conv2D(8, 1, x)
        first3Conv = self.conv2D(16,3, first3Conv)

        second3Conv = self.conv2D(8, 1, x)
        second3Conv = self.conv2D(16, 3, second3Conv)
        second3Conv = self.conv2D(16, 3, second3Conv)

        x = keras.layers.concatenate([average, oneConv, first3Conv, second3Conv])
        x = keras.layers.Conv2D(3, 3, padding='same', activation=self.relu)(x)

        return x

    def __resnet(self, inputs):
        x = self.conv2D(32, 1, inputs)
        for i in range(2):
            x = self._residual(x)

        x = keras.layers.Conv2D(3, 1, padding='same', activation=self.relu)(x)
        return x

    def _residual(self, inputs):
        x = keras.layers.BatchNormalization()(inputs)
        x = keras.layers.LeakyReLU()(x)
        x = keras.layers.Conv2D(32, 1, padding='same', use_bias=False)(x)

        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.LeakyReLU()(x)
        x = keras.layers.Conv2D(16, 3, padding='same', use_bias=False)(x)

        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.LeakyReLU()(x)
        x = keras.layers.Conv2D(32, 1, padding='same', use_bias=False)(x)

        x = keras.layers.add([inputs, x])

        return x

    #def net(self, inputs):
    #    clear, mask, rain = inputs
    #    mask = Lambda(cast_mask)(mask)
    #    smooth = keras.layers.multiply([clear, mask])
    #    original = keras.layers.multiply([rain, Lambda(subtract)(mask)])
    #    return keras.layers.Add()([smooth, original])
