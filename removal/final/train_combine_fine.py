import tensorflow as tf
from tensorflow import keras
import tensorflow.keras.backend as K
import numpy as np

from icnn import ICNN
from ard_cnn import ARDCNN
from combine_all_trainable import COMBINE
from data_ros import DataSet

from IPython import embed
from scipy import misc
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '5'

BATCH = 8
STEPS = 10000 // BATCH


def get_model(model_name, input_shape, dilated=True):
    assert model_name in ['icnn', 'whole']
    image_input = keras.Input(shape=input_shape+[3], name='rain')
    edge_input = keras.Input(shape=input_shape+[1], name='edge')

    icnn = ICNN([image_input, edge_input], False)
    if model_name == 'icnn':
        return keras.Model([image_input, edge_input], icnn.outputs)
    ard_cnn = ARDCNN(image_input, False)

    combine = COMBINE([icnn.outputs, ard_cnn.outputs, image_input], True, 'resnet', dilated)
    model = keras.Model([image_input, edge_input], combine.outputs)

    return model


def mix_loss(y_true, y_pred):
    alpha = 0.3

    y_pred = tf.minimum(y_pred, 1.0)
    ssim = tf.reduce_mean(tf.image.ssim(y_true, y_pred, max_val=1.0))
    l1 = tf.losses.absolute_difference(y_true, y_pred)

    return alpha * (1 - ssim) + (1 - alpha) * l1

def learningRateDecay(epoch):
    if epoch < 20:
        return 0.001
    elif epoch < 40:
        return 0.001 - (0.001 - 0.0001) * ((epoch - 20) / 20)
    return 0.0001


def freeze(model):
    for i in model.layers:
        i.trainable = False


if __name__ == '__main__':

    x_path = '/home/ros/ws/ijcv/repo/dataset/rain_test_bezier/'
    y_path = '/home/ros/ws/ijcv/repo/dataset/rain_test_bezier/'
    data_it = DataSet(x_path, y_path, batch_size=BATCH, mode='whole')()
    rain, edge, clear = data_it.get_next()

    rain_input = keras.Input((None, None, 3), name='rain')
    edge_input = keras.Input((None, None, 1), name='edge')

    icnn = ICNN([rain_input, edge_input], False)
    icnn_model = keras.Model([rain_input, edge_input], icnn.outputs)
    ardcnn = ARDCNN(rain_input, False)
    ardcnn_model = keras.Model(rain_input, ardcnn.outputs)
    icnn_model.load_weights('../model/bezier/icnn_weights.50_0.00580.hdf5')
    ardcnn_model.load_weights('../model/bezier/ard.40_0.00555.hdf5')
    freeze(icnn_model)
    freeze(ardcnn_model)

    combine = COMBINE([icnn.outputs, ardcnn.outputs, rain_input], True, 'resnet', True)
    model = keras.Model(inputs=[rain_input, edge_input], outputs=combine.outputs)
    model.load_weights('../model/bezier/full.59_0.00728.hdf5')

    #optimizer = keras.optimizers.Adam(0.001)
    #optimizer = keras.optimizers.Nadam(0.001)
    optimizer = keras.optimizers.SGD(0.001)

    #loss = 'mean_absolute_error'
    loss = mix_loss
    #loss = 'mean_squared_error'
    model.compile(optimizer=optimizer, loss=loss, metrics=['mae'])
    callbacks = [keras.callbacks.ModelCheckpoint('../model/bezier/fine.{epoch:02d}_{loss:.5f}.hdf5',
                                                'loss',
                                                save_best_only=False,
                                                mode='min'),
                 keras.callbacks.LearningRateScheduler(learningRateDecay)]
    model.fit({'rain': rain, 'edge': edge}, clear, epochs=60,
              shuffle=False, callbacks=callbacks,
              steps_per_epoch=STEPS)
