import tensorflow as tf
from tensorflow import keras
import tensorflow.keras.backend as K
import numpy as np

from icnn import ICNN
from data import DataSet
from data_qian import DataSet as DataSet_qian

from IPython import embed

BATCH = 32
STEPS = 50000 // BATCH

def my_loss(weights):
    def loss(y_true, y_pred):
        weight = weights * 19 + 1
        l1 = K.mean(K.abs(y_true - y_pred) * weight, axis=-1)
        return l1
    return loss

def learningRateDecay(epoch):
    if epoch < 20:
        return 0.001
    elif epoch < 60:
        return 0.001 - (0.001 - 0.0001) * ((epoch - 20) / 40)
    return 0.0001

def mix_loss(y_true, y_pred):
    alpha = 0.3

    y_pred = tf.minimum(y_pred, 1.0)
    ssim = tf.reduce_mean(tf.image.ssim(y_true, y_pred, max_val=1.0))
    l1 = tf.losses.absolute_difference(y_true, y_pred)

    return alpha * (1 - ssim) + (1 - alpha) * l1

if __name__ == '__main__':
    #data_it = DataSet('icnn', BATCH, True)()
    #rain, edge, mask, labels = data_it.get_next()

    x_path = '/home/zx/repo/dataset/qian_cvpr/train/data/'
    y_path = '/home/zx/repo/dataset/qian_cvpr/train/gt/'
    data_it = DataSet_qian(x_path, y_path, batch_size=BATCH)()
    rain, edge, labels = data_it.get_next()

    rain_input = keras.Input(shape=(100, 100, 3), name='rain')
    edge_input = keras.Input(shape=(100, 100, 1), name='edge')

    icnn = ICNN([rain_input, edge_input])

    model = keras.Model([rain_input, edge_input], icnn.outputs)
    optimizer = tf.keras.optimizers.Adam(0.0001)
    #loss = 'mean_absolute_error'
    #loss = my_loss(mask)
    loss = mix_loss
    model.compile(optimizer=optimizer, loss=loss, metrics=["mae"])
    callbacks = [keras.callbacks.TensorBoard(log_dir='../logs'),
                 keras.callbacks.ModelCheckpoint('../model_qian/icnn_weights.{epoch:02d}_{loss:.5f}.hdf5',
                                                 'loss',
                                                 save_best_only=True,
                                                 mode='min')]
                 #keras.callbacks.LearningRateScheduler(learningRateDecay)]

    model.fit({'rain': rain, 'edge': edge}, labels, epochs=20,
              shuffle=False, callbacks=callbacks,
              steps_per_epoch=STEPS)
              #steps_per_epoch=10)
