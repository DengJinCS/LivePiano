import numpy as np
import librosa
import madmom
from mido import MidiFile
from scipy.signal import argrelextrema
from math import log10
from scipy.stats.stats import pearsonr

def Onset(y,margin):
    CNNOnset = madmom.features.onsets.CNNOnsetProcessor()
    RNNOnset = madmom.features.onsets.RNNOnsetProcessor()
    # superflux = madmom.features.onsets.superflux(spectrogram=spec)
    # complexflux = madmom.features.onsets.complex_flux(spectrogram=spec)
    # percussive = librosa.effects.percussive(y)

    cnn_onset = CNNOnset(y)
    index1 = int(len(cnn_onset)/3) -1
    index2 = int(len(cnn_onset)*2/3) - 1
    target = cnn_onset[index1:index2]#中间窗口
    max_index, max_onset = 0, 0
    for index in range(index1+1):
        if target[index] > max_onset:
            max_index = index
            max_onset = target[index]
    if max_index == 0 and max_onset < cnn_onset[index1-1]:
        max_onset = 0
        max_index = -1
    if max_index == index1 - 1 and max_onset < cnn_onset[index2+1]:
        max_onset = 0
        max_index = -1
    if max_onset < margin:
        max_index = -1
    return max_index

def MCQS(y,sr,from_note=21,to_note=108,max_n = 12):
    #caculate the CQT spectrum
    #retaining only the local maximum pitch
    #restrict the top max_n pitch
    #return the modified CQT Spectrum

    n_bins = to_note - from_note
    C = np.abs(librosa.cqt(y, sr=sr,hop_length=256, bins_per_octave=12,
                           window='hamm', fmin=librosa.midi_to_hz(from_note),
                           n_bins=n_bins))
    A2dB = librosa.amplitude_to_db(C, ref=np.max)
    frame_pitch = A2dB.transpose()
    MCQS = frame_pitch - frame_pitch + frame_pitch.min()

    # retaining only the local maximum pitch
    for frame in range(len(MCQS)):
        maxIndex = argrelextrema(frame_pitch[frame], np.greater)
        #print("max", maxIndex[0])
        ranged_max_Index = np.argsort(frame_pitch[frame][maxIndex])
        #print("range:", ranged_max_Index)
        if len(ranged_max_Index) > max_n:
            restricted_max_Index = ranged_max_Index[-max_n:]
        else:
            restricted_max_Index = ranged_max_Index
        #print("Current Local Max:{}".format(len(maxIndex[0])))
        for i in maxIndex[0][restricted_max_Index]:
            MCQS[frame][i] = frame_pitch[frame][i]
    #print("Normlized:{}".format(MCQS.shape))
    T_MCQS = MCQS.transpose()

    return T_MCQS

def SDVs(MCQS,type=1,onset=-1,onste_len = 30):
    # Spectrum Difference Vectors (SDVs) ψi
    # dAi is the difference between these two vectors
    T_MCQS = MCQS.transpose()
    bias = int(len(T_MCQS) / 3)
    index = int(onset * len(T_MCQS) / onste_len)
    if onset != -1:
        V_before_onset = T_MCQS[index:index+bias]
        V_after_onset = T_MCQS[index+bias:index+2*bias]

        Mean_before = np.zeros_like(T_MCQS[0])
        Mean_after = np.zeros_like(T_MCQS[0])

        dA = np.zeros_like(T_MCQS[0])
        dAmax = 0
        Psi = np.zeros_like(T_MCQS[0])
        for k in range(len(T_MCQS[0])):
            # taking the averages of the magnitude spectrum = dAi(k)
            # immediately before and after an onset i.
            sum_before = 0
            for i in range(len(V_before_onset)):
                sum_before += V_before_onset[i][k]
            Mean_before[k] = sum_before / len(V_before_onset)

            sum_after = 0
            for j in range(len(V_after_onset)):
                sum_after += V_after_onset[j][k]
            Mean_after[k] = sum_after / len(V_after_onset)

            dA[k] = Mean_after[k] - Mean_before[k]
            if dAmax > dA[k]:
                dAmax = dA[k]
        for k in range(len(T_MCQS[0])):
            if type == 1 and dA[k] > dAmax/20:
                Psi[k] = dA[k]
            elif type == 2 and dA[k] > dAmax/20:
                Psi[k] = 1
            elif type == 3 and dA[k] > dAmax/20:
                Psi[k] = log10(dA[k])
        return Psi

def SPVs(midi='../piano/chopin_nocturne_b49.mid'):
    mid = MidiFile(midi)
    note_on_count = 0
    time = 0 # onset time
    onset_count = 1 # onset count
    onset_index = -1 #
    min_note = 108 # minimum note midi
    max_note = 0 # maxmum note midi
    max_concurrence = 1 #max concurrence count in whole concurrence
    cur_concurrence = 1 #concurrence count in every concurrence
    for msg in mid:
        if msg.type != 'note_on' and msg.type != 'note_off':
            continue
        if msg.type == "note_on":
            if msg.note > max_note:
                max_note = msg.note
            if msg.note < min_note:
                min_note = msg.note
        pre_time = time
        if msg.time != 0:
            time += msg.time
        if cur_concurrence > max_concurrence:
            max_concurrence = cur_concurrence
        if msg.type == "note_on":
            if pre_time != time:
                onset_count += 1
                cur_concurrence = 1
            else:
                cur_concurrence += 1

    note_range = max_note - min_note + 1
    #print(min_note,max_note,note_range)
    Concurrence = np.zeros((onset_count, note_range), dtype=int)
    SPV1 = np.zeros((onset_count, note_range), dtype=int)
    SPV2 = np.zeros((onset_count, note_range), dtype=int)
    SPV3 = np.zeros((onset_count, note_range), dtype=int)

    time = 0
    for msg in mid:
        if msg.type != 'note_on' and msg.type != 'note_off':
            continue
        pre_time = time
        if msg.time != 0:
            time += msg.time
        if msg.type == "note_on":
            if note_on_count == 1:
                #print(onset_index,time,msg)
                note_on_count += 1
                continue
            if pre_time != time:
                onset_index += 1
            """
            [ 0] PitchItself
            [12] SecondOctave
            [19] PerfectFifth
            [24] Third Octave
            """

            Concurrence[onset_index][msg.note - min_note + 0] = 1
            for over_tune in [0,12]:
                index = msg.note - min_note + over_tune
                if index < note_range:
                    SPV1[onset_index][index] = 1
            for over_tune in [0,12,19]:
                index = msg.note - min_note + over_tune
                if index < note_range:
                    SPV2[onset_index][index] = 1
            for over_tune in [0, 12, 19, 24,28,34]:
                index = msg.note - min_note + over_tune
                if index < note_range:
                    SPV3[onset_index][index] = 1
    return min_note,max_note,max_concurrence,Concurrence,SPV1,SPV2,SPV3

def Similarity(sdv,concurrence,spv1,spv2,spv3,k):
    correlation1 = pearsonr(sdv, concurrence[k])[0]
    correlation2 = pearsonr(sdv, spv1[k])[0]
    correlation3 = pearsonr(sdv, spv2[k])[0]
    correlation4 = pearsonr(sdv, spv3[k])[0]
    return max(correlation1,correlation2,correlation3,correlation4)

def Ita(audio_onset2,audio_onset1,audio_onset,
        score_onset2,score_onset1,score_onset,
        j,j0,a = 0.1,):
    # ita means the local tempo coefficient
    # m1 & m2 are the slopes of the latest two segments
    # j is the index of the concurrence
    # j0 is the index of the previous matched concurrence
    # delta_j is a penalty to avoid injurious score jumps & onset insertions
    delta_j = j - j0 - 1
    m1 = (score_onset - score_onset1) / (audio_onset - audio_onset1)
    m2 = (score_onset1 - score_onset2) / (audio_onset1 - audio_onset2)
    if m1/m2 < 4:
        ita = a * (1 - abs(log10(m1/m2) / log10(2)) - delta_j)
    else:
        ita = -1
    return ita
#sdv = MCQS(y,sr,min_note,max_note,max_concurrence)
#S = Similarity(sdv,Concurrence,spv1,spv2,spv3)


