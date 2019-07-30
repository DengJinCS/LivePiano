import numpy as np
import librosa
from mido import MidiFile
from scipy.signal import argrelextrema
from math import log10


def MCQS(y,sr,from_note='A0',to_note='C8',max_n = 5):
    #caculate the CQT spectrum
    #retaining only the local maximum pitch
    #restrict the top max_n pitch
    #return the modified CQT Spectrum

    n_bins = librosa.note_to_midi(to_note) - librosa.note_to_midi(from_note)
    C = np.abs(librosa.cqt(y, sr=sr, bins_per_octave=12,
                           window='hamm', fmin=librosa.note_to_hz(from_note),
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
    print("Normlized:{}".format(MCQS.shape))
    T_MCQS = MCQS.transpose()

    return A2dB,T_MCQS

def SDVs(MCQS,type=3,onset=-1,threshold=3):
    # Spectrum Difference Vectors (SDVs) ψi
    # dAi is the difference between these two vectors
    T_MCQS = MCQS.transpose()
    if onset != -1:
        if onset-threshold > 0:
            V_before_onset = T_MCQS[onset-threshold:onset]
        else:
            V_before_onset = T_MCQS[0:onset]
        if onset+threshold < len(T_MCQS) - 1:
            V_after_onset = T_MCQS[onset:onset+threshold]
        else:
            V_after_onset = T_MCQS[onset:len(T_MCQS) - 1]

        Mean_before = T_MCQS[0]
        Mean_after = T_MCQS[len(T_MCQS) - 1]

        dA = T_MCQS[0] - T_MCQS[0]
        dAmax = 0
        Psi = T_MCQS[0] - T_MCQS[0]
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

def SPVs(midi):
    mid = MidiFile(midi)
    count = 0
    for msg in mid:
        if msg.type == "note_on":
            count += 1
        print(count,msg)



