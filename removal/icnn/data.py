import glob
import numpy as np
import tensorflow as tf
from scipy import misc

from IPython import embed

class DataSet:

    def __init__(self, mode, batch_size=32, small=False):
        assert isinstance(mode, str), "mode must be a string"
        assert mode.casefold() in ["icnn", "ard-cnn", "sb-cnn", "whole"], \
            "mode must be one of 'icnn' 'ard-cnn' 'sb-cnn' and 'whole'"

        self.mode = mode.casefold()
        self.batch = batch_size

        if self.mode == 'icnn':
            self.inputs = ['raindrop.npy', 'edge.npy', 'mask.npy']
            self.labels = ['image.npy']

        elif self.mode == 'ard-cnn':
            self.inputs = ['raindrop.npy']
            self.labels = ['mask.npy']

        elif self.mode == 'sb-cnn' or self.mode == 'whole':
            self.inputs = ['raindrop.npy', "edge.npy"]
            self.labels = ['image.npy']

        if small:
            for i in range(len(self.inputs)):
                self.inputs[i] = self.inputs[i][:-4] + '_sky.npy'
            for i in range(len(self.labels)):
                self.labels[i] = self.labels[i][:-4] + '_sky.npy'

        self.dataset = self._get_data()


    def _get_data(self):
        inputs, labels= [], []
        self.shape = []
        for i in self.inputs:
            inputs.append(np.load("../data/" + i))

        for i in self.labels:
            labels.append(np.load("../data/" + i))

        for i in inputs + labels:
            image = misc.imread(i[0])
            shape = image.shape
            if len(shape) == 2:
                shape = tuple([*shape, 1])
            self.shape.append(shape)

        data = tf.data.Dataset.from_tensor_slices((*inputs, *labels))

        #data = data.apply(tf.contrib.data.shuffle_and_repeat(tf.contrib.data.AUTOTUNE))
        data = data.apply(tf.contrib.data.shuffle_and_repeat(10000))
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
        stacked = tf.random_crop(stacked, (100, 100, sum([i[-1] for i in self.shape])))

        output = []
        start = 0
        for shape in self.shape:
            output.append(stacked[:, :, start:start+shape[-1]])
            start += shape[-1]

        return tuple(output)


    def __call__(self):
        return self.iterator
