import math
import glob
import os

import tensorflow as tf
from tensorflow import keras
import tensorflow.keras.backend as K
import numpy as np
from scipy import misc
from icnn import ICNN
import cv2

from ard_cnn import ARDCNN
from combine import COMBINE

from IPython import embed
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '4'
import tqdm

#LEN = 2232 # video
#LEN = 1525 # ours testset
LEN = 500 # ours testset
# LEN = 58 #qian test_a
BATCH_SIZE = 16
STEPS = math.ceil(LEN / BATCH_SIZE)

HEIGHT = 480
WIDTH = 720

#image = misc.imread("/home/zx/repo/dataset/rain_test/cityscapes_small/0_0_B.png")
#edge = misc.imread("/home/zx/repo/dataset/rain_test/cityscapes_small/0_0_E.png")



def get_data(images, edges=None):
    key = None
    if 'youtube' not in images:
        key = lambda x: int(os.path.basename(x).split('_')[0])
    if edges is None:
        file_path = images
        images = glob.glob(file_path + '*_B.png')
        # images = glob.glob(file_path + '*rain.png')
        images.sort()
        edges = glob.glob(file_path + '*_E.png')
        edges.sort()

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

def get_model(model_name, input_shape, dilated=True):
    assert model_name in ['icnn', 'whole']
    image_input = keras.Input(shape=input_shape+[3], name='rain')
    edge_input = keras.Input(shape=input_shape+[1], name='edge')

    icnn = ICNN([image_input, edge_input], False)
    if model_name == 'icnn':
        return keras.Model([image_input, edge_input], icnn.outputs)
    ard_cnn = ARDCNN(image_input, False)

    combine = COMBINE([icnn.outputs, ard_cnn.outputs, image_input], False, 'resnet', dilated)
    model = keras.Model([image_input, edge_input], combine.outputs)

    return model

def get_all():
    #images = '/home/zx/repo/dataset/rain_test/youtube/geko_all/'
    images = '/home/lyf/ws/ijcv/repo/dataset/rain_val_with_sem/'
    image_names = glob.glob(images + '*_B.png')
    image_names.sort()
    #save_path = '/home/zx/repo/dataset/rain_result/qian/ours/%04d.png'
    save_path = '../result/rain_val_with_sem/{}'

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    iterator = get_data(images).make_one_shot_iterator()
    image, edge = iterator.get_next()

    # model = get_model('whole', [None, None])
    # weights = '../model/model_big/dilated.39_0.09956.hdf5'

    model = get_model('icnn', [None, None])
    weights = '../model/model_big/icnn_weights.40_0.00621.hdf5'

    model.load_weights(weights)
    out = model.predict({'rain': image, 'edge': edge}, steps=STEPS)

    for i, img in enumerate(tqdm.tqdm(out)):
        image_base_name = os.path.basename(image_names[i]).replace('_0_B.png', '')
        img = np.minimum(1.0, img)
        img = img * 255.0
        img = img.astype(np.uint8)

        h, w = img.shape[:2]

        img_in = cv2.imread(os.path.join(images, '{}_0_B.png'.format(image_base_name)))
        cv2.imwrite(save_path.format('{}-input.png'.format(image_base_name)), cv2.resize(img_in, (w, h)))
        img_gt = cv2.imread(os.path.join(images, '{}_I.png'.format(image_base_name)))
        cv2.imwrite(save_path.format('{}-label.png'.format(image_base_name)), cv2.resize(img_gt, (w, h)))
        cv2.imwrite(save_path.format('{}-predict.png'.format(image_base_name)), img[:, :, ::-1])

    print('\nDone.')

def get_one():
    #image = misc.imread("/home/zx/repo/dataset/rain_test/youtube/geko/01_B.png")
    #edge = misc.imread("/home/zx/repo/dataset/rain_test/youtube/geko/01_E.png")
    image = misc.imread("/home/zx/repo/dataset/qian_cvpr/test_a/data/9_rain.png")
    edge = misc.imread("/home/zx/repo/dataset/qian_cvpr/test_a/data/9_E.png")
    image = image / 255.0
    edge = edge / 255.0
    image = image.reshape((1,) + image.shape)
    edge = edge.reshape((1,) + edge.shape + (1,))

    #model = get_model('icnn', [None, None])
    model = get_model('whole', [None, None])
    weights = glob.glob('../model/dilated*')

    weights.sort()

    for i, j in enumerate(weights):
        model.load_weights(j)
        out = model.predict({'rain': image, 'edge': edge})
        out = out.reshape((HEIGHT, WIDTH, 3))
        misc.imsave('../result/one/%02d.png' % i, out)

if __name__ == '__main__':
    get_all()
