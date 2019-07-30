import librosa.display
import matplotlib.pyplot as plt
from src.Onset import *
from src.Note_Match import *

piano = '../piano/MAPS_ISOL_CH0.3_F_AkPnBcht.wav'
piano_midi = '../piano/MAPS_ISOL_CH0.3_F_AkPnBcht.mid'

chopin_wav = '/Code/PyCharm/LivePiano/piano/chopin_nocturne_b49.wav'
chopin_midi = '/Code/PyCharm/LivePiano/piano/chopin_nocturne_b49.mid'
y,sr = librosa.load(chopin_wav,None)
y = y[:10*sr]

block_dur = 0.2#s
block_len = int(block_dur * sr)
n_block = int(len(y) / block_len)

count_onset = 0
pre_onset_index = 0
pre_onset = 0
margin = 0.45

for i in range(n_block):
    block = y[i*block_len:(i+1)*block_len]
    index,onset = Onset(block)
    print(index,onset)


    if index == block_dur / 0.01 - 1:
        # the local maxmum at the end of a block
        # read a new block
        i += 1
        count_onset += 1
        # cache the current result
        pre_onset_index, pre_onset = index, onset
        block = y[i * block_len:(i + 1) * block_len]
        index, onset = Onset(block)
        if pre_onset == onset and pre_onset_index == index:
            index, onset = pre_onset_index, pre_onset
        else:
            pre_onset_index, pre_onset = index, onset
            continue

    elif index == 0 and pre_onset == onset:
        # the local maxmum at the begain of a block
        # compere with the pre block
        index, onset = pre_onset_index, pre_onset


    else:
        if onset > margin:
            count_onset += 1

    if onset > margin:
        pre_onset_index, pre_onset = index, onset
        onset_time = i * block_dur + (index) / 100
        # every 10ms as a frame to get a result
        print("{}-{}:{}-{}".format(count_onset, index, onset,onset_time))

CNNOnset = madmom.features.onsets.CNNOnsetProcessor()
RNNOnset = madmom.features.onsets.RNNOnsetProcessor()
cnn_onset = CNNOnset(y)
rnn_onset = RNNOnset(y)
onset = (cnn_onset + rnn_onset)/2
plt.plot(onset)
plt.show()


A2dB,T_MCQS = MCQS(y,sr)
plt.subplot(2,1,2)
librosa.display.specshow(T_MCQS,sr=sr,x_axis='time',y_axis='cqt_note')
plt.colorbar(format = '%+3.0f dB')
plt.title("Modified CQT Spectrum")
plt.tight_layout()

plt.subplot(2,1,1)
librosa.display.specshow(A2dB,sr=sr,x_axis='time',y_axis='cqt_note')
plt.colorbar(format = '%+3.0f dB')
plt.title("Constant-Q power spectrum")
plt.tight_layout()

#plt.show()


