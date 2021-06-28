import math
import glob
from os.path import basename

import tensorflow as tf
from tensorflow import keras
import numpy as np
from scipy import misc
from ard_cnn import ARDCNN
import cv2
from sklearn import metrics
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '4'


from IPython import embed

#LEN = 2232
LEN = 5
BATCH_SIZE = 64
STEPS = math.ceil(LEN / BATCH_SIZE)

STEPS_EVAL = math.ceil(500 / BATCH_SIZE)


def get_data(path):
    key = None
    if 'youtube' not in path:
        key = lambda x: int(basename(x).split('_')[0])
    images = glob.glob(path + '*_B.png')
    images.sort(key=key)
    mask = glob.glob(path + '*_M.png')
    mask.sort(key=key)

    images = np.array(images)
    mask = np.array(mask)

    if len(mask) != 0:
        data = tf.data.Dataset.from_tensor_slices((images, mask))
    else:
        data = tf.data.Dataset.from_tensor_slices((images))
    data = data.apply(tf.contrib.data.map_and_batch(
        map_func, BATCH_SIZE, num_parallel_calls=1))
    data = data.prefetch(tf.contrib.data.AUTOTUNE)

    return data

def map_func(image, mask=None):
    image = tf.read_file(image)
    image = tf.image.decode_png(image)
    image.set_shape((256, 512, 3))
    image = tf.image.convert_image_dtype(image, dtype=tf.float32)

    if mask != None:
        mask = tf.read_file(mask)
        mask = tf.image.decode_png(mask)
        mask = tf.reshape(mask, (256, 512, 1))
        mask = tf.image.convert_image_dtype(mask, dtype=tf.float32)

        return image, mask

    return image



def get_all():
    #images = '/home/zx/repo/dataset/rain_test/youtube/geko_all/'
    images = '/home/lyf/ws/ijcv/repo/dataset/test_bezier_PR/'
    iterator = get_data(images).make_one_shot_iterator()
    image, _ = iterator.get_next()

    image_input = keras.Input(shape=(256, 512, 3), name='rain')
    ard_cnn = ARDCNN(image_input, False)
    model = keras.Model(image_input, ard_cnn.outputs)

    weights = '../model/bezier/ard.40_0.00555.hdf5'
    model.load_weights(weights)

    out = model.predict(image, steps=STEPS)

    path = glob.glob(images + '*B.png')
    path.sort()

    #np.save('/home/zx/repo/dataset/rain_result/ours/mask/mask', out)
    for i, img in enumerate(out):
        img = np.where(img < 0.5, 0.0, 1.0)
        img = img * 255.0
        img = img.astype(np.uint8)
        cv2.imwrite('/home/ros/ws/ijcv/repo/rain_result/ours/mask-PR/{}'.format(os.path.basename(path[i]).replace('B.png', 'M_pred.png')), img)
        #cv2.imwrite(path[i].replace('B.png', 'P.png'), img)

def eval_all():
    images = '/home/zx/repo/dataset/rain_test/cityscapes_small/'
    iterator = get_data(images).make_initializable_iterator()
    image, _ = iterator.get_next()

    image_input = keras.Input(shape=(256, 512, 3), name='rain')
    ard_cnn = ARDCNN(image_input, False)
    model = keras.Model(image_input, ard_cnn.outputs)

    weights = glob.glob('../model/w_ard*')
    #weights = glob.glob('../model_sky/ard_weights.26*')
    weights.sort()

    gt = get_gt(images)
    confusion_matrixs = []

    for i, weight in enumerate(weights):
        keras.backend.get_session().run(iterator.initializer)
        model.load_weights(weight)
        res = model.predict(image, steps=STEPS_EVAL)
        res = np.where(res > 0.5, 1.0, 0.0)
        confusion_matrixs.append(metrics.confusion_matrix(gt, res.flatten()))

    show_score(confusion_matrixs)


def get_gt(path):
    gt = np.ndarray((500, 256, 512))
    path = glob.glob(path + '*M.png')
    path.sort()
    for i, img in enumerate(path):
        gt[i] = cv2.imread(img, 0)

    return gt.flatten() / 255

def show_score(confusion_matrixs):
    scores = []
    for i in range(len(confusion_matrixs)):
        TP = confusion_matrixs[i][1, 1]
        FP = confusion_matrixs[i][0, 1]
        TN = confusion_matrixs[i][0, 0]
        FN = confusion_matrixs[i][1, 0]

        precision = TP / (TP + FP)
        recall = TP / (TP + FN)
        f1 = 2 * precision * recall / (precision + recall)

        scores.append((i, precision, recall, f1))

    scores.sort(key=lambda x: x[-1])
    for i, p, r, f1 in scores:
        print('Index: %d, precision: %f, recall: %f, f1: %f' % (i, p, r, f1))

def get_one():
    #image = misc.imread("/home/zx/repo/dataset/rain_test/youtube/geko/01_B.png")
    image = misc.imread("../result/fuse/00000.png")
    image = image / 255.0
    image = image.reshape((1, 256, 512, 3))
    image_input = keras.Input(shape=(256, 512, 3), name='rain')

    ard_cnn = ARDCNN(image_input, False)
    model = keras.Model(image_input, ard_cnn.outputs)

    weights = glob.glob('../model/ard*')
    weights.sort()

    for i, j in enumerate(weights):
        model.load_weights(j)
        out = model.predict(image)
        out = out.reshape((256, 512))
        out = np.where(out > 0.5, 1.0, 0.0)
        misc.imsave('../result/ardcnn/%02d.png' % i, out)

get_all()
