import datetime

import tensorflow as tf
from tensorflow import keras
import tensorflow.keras.backend as K
import numpy as np
from scipy import misc
from icnn import ICNN
from ard_cnn import ARDCNN
from combine import COMBINE

from IPython import embed

def get_model(model_name, input_shape, dilate=False):
    assert model_name in ['icnn', 'whole']
    image_input = keras.Input(shape=input_shape+[3], name='rain')
    edge_input = keras.Input(shape=input_shape+[1], name='edge')

    icnn = ICNN([image_input, edge_input], False)
    if model_name == 'icnn':
        return keras.Model([image_input, edge_input], icnn.outputs)
    ard_cnn = ARDCNN(image_input, False)

    combine = COMBINE([icnn.outputs, ard_cnn.outputs, image_input], False, 'resnet', dilate)
    model = keras.Model([image_input, edge_input], combine.outputs)

    return model

def get_one():
    image = misc.imread("/home/zx/repo/dataset/rain_test/youtube/geko/01_B.png")
    edge = misc.imread("/home/zx/repo/dataset/rain_test/youtube/geko/01_E.png")
    image = image / 255.0
    edge = edge / 255.0
    image = image.reshape((1,) + image.shape)
    edge = edge.reshape((1,) + edge.shape + (1,))

    model = get_model('whole', [None, None], dilate=True)
    weight = '../model/dilated.01_0.10362.hdf5'
    model.load_weights(weight)

    # warm up
    for i in range(10):
        model.predict({'rain': image, 'edge': edge})

    sumtime = 0.0
    for i in range(20):
        start = datetime.datetime.now()
        model.predict({'rain': image, 'edge': edge})
        end = datetime.datetime.now()
        dur = end - start
        sumtime += dur.microseconds

    print(sumtime / 20)

if __name__ == '__main__':
    get_one()
