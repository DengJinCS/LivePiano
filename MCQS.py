import librosa.display
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import argrelextrema
piano = 'piano/MAPS_ISOL_CH0.3_F_AkPnBcht.wav'
music = '/Users/atticus/Music/网易云音乐/nocturne.mp3'
y,sr = librosa.load(music,None)
y = y[:100*sr]
#y = y[int(12.5904*sr):int(sr*12.8904)]#MIDI:60
#y = y[int(sr*6.39019):int(sr*6.6902)]#MIDI:40
#y = y[int(sr*3.29009):int(sr*3.5901)]#MIDI:30
#y = y[int(sr*0.5):int(sr*0.8)]#MIDI:21

def MCQS(y,sr,from_note='C2',to_note='C7',max_n = 5):
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

plt.show()


