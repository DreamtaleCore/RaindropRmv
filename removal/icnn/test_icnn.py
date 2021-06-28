import math
import glob
import os

import tensorflow as tf
from tensorflow import keras
import tensorflow.keras.backend as K
import numpy as np
from scipy import misc
from icnn import ICNN
from ard_cnn import ARDCNN
from combine import COMBINE
import cv2

from IPython import embed

LEN = [-1]
#LEN = 1799
#LEN = 2232 # video
#LEN = 1525 # ours testset
#LEN = 58 #qian test_a
BATCH_SIZE = 2
#STEPS = math.ceil(LEN / BATCH_SIZE)

HEIGHT = 256
WIDTH = 512

#image = misc.imread("/home/zx/repo/dataset/rain_test/cityscapes_small/0_0_B.png")
#edge = misc.imread("/home/zx/repo/dataset/rain_test/cityscapes_small/0_0_E.png")



def get_data(images, edges=None):
    key = None
    if 'youtube' not in images:
        key = lambda x: int(os.path.basename(x).split('_')[0])
    if edges is None:
        file_path = images
        images = glob.glob(file_path + '*_B.png')
        #images = glob.glob(file_path + '*rain.png')
        images.sort(key=key)
        edges = glob.glob(file_path + '*_E.png')
        edges.sort(key=key)
    LEN[0] = len(images)

    images = np.array(images)
    edges = np.array(edges)

    data = tf.data.Dataset.from_tensor_slices((images, edges))
    data = data.apply(tf.contrib.data.map_and_batch(
        map_func, BATCH_SIZE, num_parallel_calls=12))
    data = data.prefetch(tf.contrib.data.AUTOTUNE)

    return data

def map_func(image, edge):
    image = tf.read_file(image)
    edge = tf.read_file(edge)
    image = tf.image.decode_png(image)
    edge = tf.image.decode_png(edge)

    #image.set_shape((HEIGHT, WIDTH, 3))
    #edge = tf.reshape(edge, (HEIGHT, WIDTH, 1))

    image = tf.image.convert_image_dtype(image, dtype=tf.float32)
    edge = tf.image.convert_image_dtype(edge, dtype=tf.float32)

    return image, edge

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

def get_all():
    images = '/home/zx/repo/dataset/rain_test/youtube/geko_all/'
    #images = '/home/zx/repo/dataset/rain_val/'
    #save_path = '/home/zx/repo/dataset/rain_result/ours/val/%03d.png'
    #save_path = '/tmp/tmp/%03d.png'
    save_path = '/home/zx/repo/dataset/rain_result/ours/youtube/geko/%04d.png'

    iterator = get_data(images).make_one_shot_iterator()
    image, edge = iterator.get_next()

    model = get_model('whole', [None, None], dilate=True)
    weights = '../model/dilated.41_0.00909.hdf5'

    #model = get_model('icnn', [None, None])
    #weights = '../model_qian/icnn_weights.11_0.09563.hdf5'
    #weights = '../model/icnn_weights.25_0.01892.hdf5'

    model.load_weights(weights)

    STEPS = math.ceil(LEN[0] / BATCH_SIZE)
    out = model.predict({'rain': image, 'edge': edge}, steps=STEPS)

    for i, img in enumerate(out):
        img = np.minimum(1.0, img)
        img = img * 255.0
        img = img.astype(np.uint8)
        cv2.imwrite(save_path % i, img[:, :, ::-1])

def get_one():
    #image = misc.imread("/home/zx/repo/dataset/rain_test/youtube/geko/01_B.png")
    #edge = misc.imread("/home/zx/repo/dataset/rain_test/youtube/geko/01_E.png")
    image = misc.imread("/home/zx/source/c++/rainEdge/build/Release/tmp/1_B.png")
    edge = misc.imread("/home/zx/source/c++/rainEdge/build/Release/tmp/1_E.png")
    image = image / 255.0
    edge = edge / 255.0
    image = image.reshape((1,) + image.shape)
    edge = edge.reshape((1,) + edge.shape + (1,))

    #model = get_model('icnn', [None, None])
    model = get_model('whole', [None, None], dilate=True)
    weights = glob.glob('../model/dilated*')

    weights.sort()

    for i, j in enumerate(weights):
        model.load_weights(j)
        out = model.predict({'rain': image, 'edge': edge})
        out = out.reshape((HEIGHT, WIDTH, 3))
        misc.imsave('../result/icnn/%02d.png' % i, out)

if __name__ == '__main__':
    get_all()
