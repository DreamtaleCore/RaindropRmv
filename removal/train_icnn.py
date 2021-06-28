import tensorflow as tf
from tensorflow import keras
import tensorflow.keras.backend as K
import numpy as np
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '2'

from icnn import ICNN
from data_ros import DataSet

from IPython import embed

BATCH = 16
STEPS = 50000 // BATCH
ALPHA = 0.1

def my_loss(weights):
    def loss(y_true, y_pred):
        weight = weights * 19 + 1
        l1 = K.mean(K.abs(y_true - y_pred) * weight, axis=-1)
        return l1
    return loss

def learningRateDecay(epoch):
    if epoch < 2:
        return 0.001
    #elif epoch < 40:
    #    return 0.001 - (0.001 - 0.0001) * ((epoch - 20) / 20)
    return 0.00001

def mix_loss(y_true, y_pred):

    y_pred = tf.minimum(y_pred, 1.0)
    ssim = tf.reduce_mean(tf.image.ssim(y_true, y_pred, max_val=1.0))
    l1 = tf.losses.absolute_difference(y_true, y_pred)

    return ALPHA * (1 - ssim) + (1 - ALPHA) * l1

if __name__ == '__main__':

    x_path = 'repo/dataset/rain_train_bezier/'
    y_path = 'repo/dataset/rain_train_bezier/'
    data_it = DataSet(x_path, y_path, batch_size=BATCH, mode='icnn')()
    data_in = data_it.get_next()
    rain, edge, _, labels = data_in

    rain_input = keras.Input(shape=(None, None, 3), name='rain')
    edge_input = keras.Input(shape=(None, None, 1), name='edge')

    icnn = ICNN([rain_input, edge_input])

    model = keras.Model([rain_input, edge_input], icnn.outputs)
    optimizer = tf.keras.optimizers.Adam(0.0001)
    #optimizer = tf.keras.optimizers.SGD(0.01)
    #optimizer = tf.keras.optimizers.RMSprop(0.00001)
    #loss = 'mean_absolute_error'
    #loss = my_loss(mask)
    loss = mix_loss
    model.compile(optimizer=optimizer, loss=loss, metrics=["mae"])
    callbacks = [keras.callbacks.TensorBoard(log_dir='../log/'),
                 keras.callbacks.ModelCheckpoint('../model/icnn_weights.{epoch:02d}_{loss:.5f}.hdf5',
                                                 'loss',
                                                 save_best_only=False,
                                                 mode='min')]
                 #keras.callbacks.LearningRateScheduler(learningRateDecay)]

    model.fit({'rain': rain, 'edge': edge}, labels, epochs=50,
              shuffle=False, callbacks=callbacks,
              steps_per_epoch=STEPS)
