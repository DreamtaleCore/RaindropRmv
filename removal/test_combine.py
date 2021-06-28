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
#from combine import COMBINE
from combine_all_trainable import COMBINE
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '4'
import tqdm


from IPython import embed

#LEN = 2232 # video
#LEN = 1525 # ours testset
# LEN = 58 #qian test_a
LEN = 500 # ours testset
BATCH_SIZE = 16
STEPS = math.ceil(LEN / BATCH_SIZE)

HEIGHT = 480
WIDTH = 720

#image = misc.imread("repo/dataset/rain_test/cityscapes_small/0_0_B.png")
#edge = misc.imread("repo/dataset/rain_test/cityscapes_small/0_0_E.png")


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


def freeze(model):
    for i in model.layers:
        i.trainable = False


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
    images = 'repo/dataset/rain_val_with_sem/'
    image_names = glob.glob(images + '*_B.png')
    image_names.sort()

    save_path = '../result/rain_val_with_sem/{}'

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    iterator = get_data(images).make_one_shot_iterator()
    image, edge = iterator.get_next()

    print(image.shape)
    print(image.shape)

    #model = get_model('icnn', [None, None])
    #weights = '../model/i
    rain_input = keras.Input((None, None, 3), name='rain')
    edge_input = keras.Input((None, None, 1), name='edge')

    # icnn = ICNN([rain_input, edge_input], False)
    # icnn_model = keras.Model([rain_input, edge_input], icnn.outputs)
    # ardcnn = ARDCNN(rain_input, False)
    # ardcnn_model = keras.Model(rain_input, ardcnn.outputs)
    # # freeze(icnn_model)
    # # freeze(ardcnn_model)

    # combine = COMBINE([icnn.outputs, ardcnn.outputs, rain_input], True, 'resnet', True)
    # model = keras.Model(inputs=[rain_input, edge_input], outputs=combine.outputs)

    # model.load_weights('../model/final_fine/full.60_0.06945.hdf5')
    # # ardcnn_model.load_weights('../model/bezier/ard.40_0.00555.hdf5')
    # # icnn_model.load_weights('../model/bezier/icnn_weights.50_0.00580.hdf5')

    model = get_model('whole', [None, None], False)
    weights = '../model/fine/full.42_0.07202.hdf5'

    #model = get_model('icnn', [None, None])
    #weights = '../model/i

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


if __name__ == '__main__':
    get_all()
    #get_all()

