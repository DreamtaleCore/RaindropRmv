import tensorflow as tf
from tensorflow import keras
import tensorflow.keras.backend as K

from test_icnn import get_model

from IPython import embed

EXPORT_PATH='/home/zx/source/python/theis/savedModel/rain/1'

K.set_learning_phase(0)
model = get_model('whole', [None, None], dilate=True)
model.load_weights('../model/dilated.41_0.00909.hdf5')

for layer in model.layers:
    if layer.name == 'lambda_1':
        dilated_mask = layer.output

rain, edge = model.input

with K.get_session() as sess:
    tf.saved_model.simple_save(
        sess,
        EXPORT_PATH,
        inputs={'img': rain, 'edge': edge},
        outputs={'clear': model.output, 'mask': dilated_mask})

    print('Done')
