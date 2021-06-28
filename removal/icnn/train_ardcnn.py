import tensorflow as tf
from tensorflow import keras
import tensorflow.keras.backend as K
import numpy as np

from ard_cnn import ARDCNN
from data import DataSet

from IPython import embed

BATCH = 32
STEPS = 50000 // BATCH
EPS = 1e-8


def weighted_cross_entropy(mask):
    def loss(mask, pred):
        weight = mask * 19 + 1
        cross_entropy = (mask * K.log(pred + EPS) + \
            (1 - mask) * K.log(1 - pred + EPS)) * weight
        return -K.mean(cross_entropy)
    return loss


if __name__ == '__main__':
    data_it = DataSet('ard-cnn', BATCH, True)()
    rain, labels = data_it.get_next()
    rain_input = keras.Input(shape=(100, 100, 3), name='rain')

    ard_cnn = ARDCNN(rain_input)

    model = keras.Model(rain_input, ard_cnn.outputs)

    optimizer = tf.keras.optimizers.Adam(0.0001)
    #loss = 'binary_crossentropy'
    loss = weighted_cross_entropy(labels)
    model.compile(optimizer=optimizer, loss=loss, metrics=["acc"])

    callbacks = [keras.callbacks.TensorBoard(log_dir='../logs'),
                 keras.callbacks.ModelCheckpoint('../model/w_ard.{epoch:02d}_{loss:.5f}.hdf5',
                                                 'acc',
                                                 save_best_only=False,
                                                 mode='max')]
                                                 #save_weights_only=True)]
    model.fit(rain, labels, epochs=40, shuffle=False, callbacks=callbacks,
              steps_per_epoch=STEPS)
              #steps_per_epoch=10)
