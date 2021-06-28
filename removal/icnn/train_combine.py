import tensorflow as tf
from tensorflow import keras
import tensorflow.keras.backend as K
import numpy as np

from icnn import ICNN
from ard_cnn import ARDCNN
from combine import COMBINE
from data import DataSet

from IPython import embed
from scipy import misc

BATCH = 32
STEPS = 50000 // BATCH


def test(icnn, ardcnn, model):
    #icnn.load_weights('../model/icnn_weights.25_0.01892.hdf5')
    #ardcnn.load_weights('../model/ard_weights.36_0.00303.hdf5')

    rain_img = misc.imread('/home/zx/repo/dataset/rain_test/cityscapes_small/406_0_B.png')
    edge_img = misc.imread('/home/zx/repo/dataset/rain_test/cityscapes_small/406_0_E.png')

    rain_img = rain_img.astype(np.float32) / 255.0
    edge_img = edge_img.astype(np.float32) / 255.0
    rain_img = rain_img.reshape((1, 256, 512, 3))
    edge_img = edge_img.reshape((1, 256, 512, 1))

    keras.utils.plot_model(model, to_file='../model.png')
    embed()
    #clear = icnn.predict([rain_img, edge_img])[0]
    #mask = ardcnn.predict(rain_img)[0]

    #misc.imsave('../clear.png', clear)
    #misc.imsave('../mask.png', mask[:,:,0])
def freeze(model):
    for i in model.layers:
        i.trainable = False

def mix_loss(y_true, y_pred):
    alpha = 0.3

    y_pred = tf.minimum(y_pred, 1.0)
    ssim = tf.reduce_mean(tf.image.ssim(y_true, y_pred, max_val=1.0))
    l1 = tf.losses.absolute_difference(y_true, y_pred)

    return alpha * (1 - ssim) + (1 - alpha) * l1


def learningRateDecay(epoch):
    if epoch < 20:
        return 0.001
    else:
        return 0.001 - (0.001 - 0.0001) * ((epoch - 20) / 20)


if __name__ == '__main__':

    trainData = DataSet('whole', BATCH, True)()
    rain, edge, clear = trainData.get_next()
    input_h = 100
    input_w = 100
    rain_input = keras.Input((input_h, input_w, 3), name='rain')
    edge_input = keras.Input((input_h, input_w, 1), name='edge')

    icnn = ICNN([rain_input, edge_input], False)
    icnn_model = keras.Model([rain_input, edge_input], icnn.outputs)
    ardcnn = ARDCNN(rain_input, False)
    ardcnn_model = keras.Model(rain_input, ardcnn.outputs)
    icnn_model.load_weights('../model/icnn_weights.25_0.01892.hdf5')
    ardcnn_model.load_weights('../model/ard_weights.36_0.00303.hdf5')
    freeze(icnn_model)
    freeze(ardcnn_model)

    combine = COMBINE([icnn.outputs, ardcnn.outputs, rain_input], True, 'resnet', True)
    model = keras.Model(inputs=[rain_input, edge_input], outputs=combine.outputs)
    optimizer = keras.optimizers.Adam(0.0001)
    #loss = 'mean_absolute_error'
    loss = mix_loss
    model.compile(optimizer=optimizer, loss=loss, metrics=['mae'])
    callbacks = [keras.callbacks.ModelCheckpoint(#'../model/dilated.{epoch:02d}_{loss:.5f}.hdf5',
                                                '../tmp/dilated.{epoch:02d}_{loss:.5f}.hdf5',
                                                'loss',
                                                save_best_only=True,
                                                mode='min')]
                 #keras.callbacks.LearningRateScheduler(learningRateDecay)]

    #test(icnn_model, ardcnn_model, model)
    model.fit({'rain': rain, 'edge': edge}, clear, epochs=60,
              shuffle=False, callbacks=callbacks,
              steps_per_epoch=STEPS)
