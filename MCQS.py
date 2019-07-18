import librosa.display
import numpy as np
import matplotlib.pyplot as plt
from operator import itemgetter
piano = 'piano/MAPS_ISOL_CH0.3_F_AkPnBcht.wav'
music = '/Users/atticus/Music/网易云音乐/nocturne.mp3'
y,sr = librosa.load(music,None)
y = y
C = np.abs(librosa.cqt(y,sr=sr,bins_per_octave=12,
                       window='hamm',fmin=librosa.note_to_hz('C1'),
                       n_bins=88))
A2dB = librosa.amplitude_to_db(C,ref=np.max)
print("Origin:{}".format(A2dB.shape))
frame_pitch = A2dB.transpose()
print("Transposed:{}".format(frame_pitch.shape))
MCQS = frame_pitch - frame_pitch + frame_pitch.min()


for frame in range(len(MCQS)):
    index = np.argsort(frame_pitch[frame])
    index = index[-4:]
    for i in index:
        MCQS[frame][i] = frame_pitch[frame][i]
print("Normlized:{}".format(MCQS.shape))

T_MCQS = MCQS.transpose()
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

