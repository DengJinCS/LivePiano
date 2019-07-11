# %%
from matplotlib import pyplot as plt

from glob import glob

import numpy as np
import os

from tensorflow import Session
from tensorflow.contrib.framework import list_variables, load_variable

from keras import backend as K
from keras.layers import Activation, BatchNormalization, Bidirectional, concatenate, \
    Conv2D, Dense, Input, LSTM, MaxPool2D, Reshape
from keras.models import load_model, Model

from librosa import midi_to_hz
from sklearn.metrics import auc, roc_curve
from sklearn.preprocessing import binarize

from magenta.music.sequences_lib import pianoroll_to_note_sequence
from magenta.models.onsets_frames_transcription.infer_util import sequence_to_valued_intervals
from mir_eval.transcription import precision_recall_f1_overlap as Overlap
from mir_eval.transcription_velocity import precision_recall_f1_overlap as OverlapVolumes

# %%
dataFolder, melsMinMin, melsMinMax, melsMeanMin, melsMeanMax, melsMaxMin, melsMaxMax \
    = 'Maestro/5-20 seconds', -87, -86, -31, -30, 40, 41

melsVal = np.zeros((3088,626,229))
print(len(melsVal), 'validation samples,', end='\t')

cptDir, lstmWidth, inputs = '/Code/PyCharm/LivePiano/GoogleMagenta/content/maestro_checkpoint', 256, Input(shape=(626,229))

ConvBnRelu = lambda n: lambda x: Activation('relu')(BatchNormalization(scale=False)(
    Conv2D(n, 3, padding='same', use_bias=False)(x)))
outputs = MaxPool2D((1, 2))(ConvBnRelu(96)(MaxPool2D((1, 2))(ConvBnRelu(48)(
    ConvBnRelu(48)(Reshape((melsVal.shape[1], melsVal.shape[2], 1))(inputs))))))

model = Model(inputs, Dense(88, activation='sigmoid')(Bidirectional(LSTM(lstmWidth,
                                                                         # https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/ops/rnn_cell_impl.py#L782
                                                                         # Sigmoid activation for hidden units, also unroll must be True, otherwise weights will be different
                                                                         # Cannot use stateful mode because backward LSTM would require reversed batch:
                                                                         recurrent_activation='sigmoid',
                                                                         implementation=2, return_sequences=True,
                                                                         unroll=True))(Dense(768, activation='relu')(
    Reshape((K.int_shape(outputs)[1], K.int_shape(outputs)[2] * K.int_shape(outputs)[3]))(outputs)))))

for i in [2, 3, 5, 6, 9, 10, 14, 15, 16]: print(i,*model.layers[i].weights, sep='\n', end='\n\n')
with Session() as sess: print(*['{}\t{}'.format(name, shape) for name, shape in list_variables(cptDir)], sep='\n')
model.summary()


# %%
def MagentaToKeras(modelName):
    if os.path.exists(modelName+'.hdf5'):
        return 0
    numLayers, varName, varProbName = 16, modelName.lower(), None

    for i, j in zip(range(3), [2, 5, 9]):
        model.layers[j].set_weights([load_variable(cptDir, '{}/conv{}/weights'.format(varName, i))])
        model.layers[j + 1].set_weights([load_variable(cptDir, '{}/conv{}/BatchNorm/{}'.format(varName, i, name))
                                         for name in ['beta', 'moving_mean', 'moving_variance']])
    model.layers[14].set_weights([load_variable(cptDir, '{}/fc_end/{}'.format(varName, name))
                                  for name in ['weights', 'biases']])

    if modelName in ['Onsets', 'Offsets']:
        varProbName = varName[:-1] + '_probs'
    elif modelName == 'Velocity':
        numLayers, varProbName = 15, 'onset_velocities'
    else:
        assert modelName == 'Frame', 'Illegal model name'
        numLayers, varProbName = 20, 'frame_probs'
        model.layers[15].set_weights([load_variable(cptDir, 'frame/activation_probs/{}'.format(name))
                                      for name in ['weights', 'biases']])

    if modelName != 'Velocity':
        fW, fB, bW, bB = [load_variable(cptDir, '{}/cudnn_lstm/stack_bidirectional_rnn/cell_0/bidirectional_rnn/'
                                                '{}/cudnn_compatible_lstm_cell/{}'.format(varName, fb, kb)) for fb in
                          ['fw', 'bw'] for kb in ['kernel', 'bias']]
        # https://stackoverflow.com/questions/48212694/in-what-order-are-weights-saved-in-a-lstm-kernel-in-tensorflow
        # Tensorflow order is I-C-F-O:
        (fWi, fWc, fWf, fWo), (bWi, bWc, bWf, bWo) = map(lambda arr: np.split(arr, 4, 1), [fW, bW])
        (fBi, fBc, fBf, fBo), (bBi, bBc, bBf, bBo) = map(lambda arr: np.split(arr, 4), [fB, bB])
        # https://stackoverflow.com/questions/47661105/order-of-lstm-weights-in-keras
        # Keras order is I-F-C-O:
        fWk, bWk, fBk, bBk = map(np.hstack, [[fWi, fWf, fWc, fWo], [bWi, bWf, bWc, bWo],
                                             [fBi, fBf, fBc, fBo], [bBi, bBf, bBc, bBo]])
        # https://stats.stackexchange.com/questions/280995/accessing-lstm-weights-tensors-in-tensorflow
        # Input units first, then hidden (recurrent) nodes:
        model.layers[numLayers - 1].set_weights([fWk[:-lstmWidth], fWk[-lstmWidth:], fBk,
                                                 bWk[:-lstmWidth], bWk[-lstmWidth:], bBk])

    model.layers[numLayers].set_weights([load_variable(cptDir, '{}/{}/{}'.format(
        varName, varProbName, name)) for name in ['weights', 'biases']])
    model.save('{}.hdf5'.format(modelName), include_optimizer=False)
    print("Saved to {}.hdf5".format(modelName))

MagentaToKeras('Onsets')
MagentaToKeras('Offsets')
MagentaToKeras('Frame')
MagentaToKeras('Velocity')
