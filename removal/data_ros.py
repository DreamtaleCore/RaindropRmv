import glob
import numpy as np
import tensorflow as tf
from scipy import misc

from IPython import embed

class DataSet:

    def __init__(self, x_path, y_path, suffix='.png', batch_size=32, mode='icnn'):
        self.batch = batch_size
        self.edge = sorted(glob.glob(x_path + '*_E' + suffix))
        self.rain = sorted(glob.glob(x_path + '*_B' + suffix))
        self.mask = sorted(glob.glob(x_path + '*_M' + suffix))
        self.label = [x.replace('_0_B.', '_I.') for x in self.rain]

        self.edge = np.array(self.edge)
        self.rain = np.array(self.rain)
        self.label = np.array(self.label)

        if mode == 'icnn':
            self.inputs = [self.rain, self.edge, self.mask]
            self.labels = [self.label]
        elif mode == 'ard-cnn':
            self.inputs = [self.rain]
            self.labels = [self.mask]
        elif mode == 'sb-cnn' or mode == 'whole':
            self.inputs = [self.rain, self.edge]
            self.labels = [self.label]
        else:
            raise ValueError('mode must be in [icnn, ard-cnn, sb-cnn, whole], however, it is {}.'.format(mode))

        self.dataset = self._get_data()


    def _get_data(self):
        self.shape = []
        for i in self.inputs + self.labels:
            image = misc.imread(i[0])
            shape = image.shape
            if len(shape) == 2:
                shape = tuple([*shape, 1])
            self.shape.append(shape)

        # self.shape.append(self.shape[0][:2] + (1,))

        data = tf.data.Dataset.from_tensor_slices((*self.inputs, *self.labels))

        #data = data.apply(tf.contrib.data.shuffle_and_repeat(tf.contrib.data.AUTOTUNE))
        # data = data.apply(tf.contrib.data.shuffle_and_repeat(1000))
        # data = data.apply(tf.contrib.data.repeat(1000))
        data = data.apply(tf.contrib.data.map_and_batch(
            self._map, self.batch, num_parallel_calls=12))

        data = data.prefetch(tf.contrib.data.AUTOTUNE)
        self.iterator = data.make_one_shot_iterator()

        return data

    def _map(self, *args):

        assert len(args) == len(self.inputs) + len(self.labels)
        output = []

        for j, i in enumerate(args):
            image = tf.read_file(i)
            image = tf.image.decode_png(image)
            if self.shape[j][-1] == 1:
                image = tf.reshape(image, self.shape[j])
            else:
                image.set_shape(self.shape[j])

            image = tf.image.convert_image_dtype(image, dtype=tf.float32)

            output.append(image)

        stacked = tf.concat(output, axis=-1)
        stacked = tf.random_crop(stacked, (200, 200, sum([i[-1] for i in self.shape])))

        output = []
        start = 0
        for shape in self.shape:
            output.append(stacked[:, :, start:start+shape[-1]])
            start += shape[-1]

        return tuple(output)


    def __call__(self):
        return self.iterator
