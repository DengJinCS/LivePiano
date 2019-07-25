import librosa
import numpy as np
from src.Onset import Onset
piano = 'piano/MAPS_ISOL_CH0.3_F_AkPnBcht.wav'
music = '/Users/atticus/Music/网易云音乐/nocturne.mp3'
y,sr = librosa.load(piano,None)

block_dur = 0.2#s
block_len = int(block_dur * sr)
n_block = int(len(y) / sr / block_dur)

count_onset = 0
pre_onset_index = 0
pre_onset = 0
margin = 0.4

for i in range(n_block):
    block = y[i*block_len:(i+1)*block_len]
    index,onset = Onset(block)


    if index == block_dur / 0.01 - 1:
        # the local maxmum at the end of a block
        # read a new block
        i += 1
        count_onset += 1
        # cache the current result
        pre_onset_index, pre_onset = index, onset
        block = y[i * block_len:(i + 1) * block_len]
        index, onset = Onset(block)
        if pre_onset == onset:
            index, onset = pre_onset_index, pre_onset
        else:
            pre_onset_index, pre_onset = index, onset
            continue

    elif index == 0:
        # the local maxmum at the begain of a block
        # compere with the pre block
        if pre_onset == onset:
            index, onset = pre_onset_index, pre_onset
        else:
            continue

    else:
        if onset > margin:
            count_onset += 1

    if onset > margin:
        pre_onset_index, pre_onset = index, onset
        onset_time = i * block_dur + (index-1) / 100
        # every 10ms as a frame to get a result
        print("{}-{}:{}".format(count_onset, index, onset_time))
