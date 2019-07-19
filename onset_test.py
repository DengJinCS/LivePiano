from pitch import YIN_Pitch
import madmom
import librosa
import matplotlib.pyplot as plt
import numpy as np
piano = 'piano/MAPS_ISOL_CH0.3_F_AkPnBcht.wav'
music = '/Users/atticus/Music/网易云音乐/nocturne.mp3'
y,sr = librosa.load(piano,None)
k = y
spec = madmom.audio.spectrogram.spec(librosa.stft(k))
CNNOnset = madmom.features.onsets.CNNOnsetProcessor()
RNNOnset = madmom.features.onsets.RNNOnsetProcessor()
#superflux = madmom.features.onsets.superflux(spectrogram=spec)
#complexflux = madmom.features.onsets.complex_flux(spectrogram=spec)
percussive = librosa.effects.percussive(k)

cnn_onset = CNNOnset(k)
rnn_onset = RNNOnset(k)
onset = (cnn_onset + rnn_onset)/2
plt.plot(onset)
plt.show()
print(len(onset))
count = 0
n = 0
pre = 0
for i in onset:
    if i > 0.3:
        if n - pre > 5:
            pre = n
            print("onset:",n/100)
            count += 1
    n += 1
new_onset = np.zeros(len(k))
block = len(k)/len(onset)
onset_count = 0
for i in range(len(onset)):
    for j in range(int(block)):
        new_onset[int(i*block+j)] = onset[onset_count]
    onset_count += 1
print("durition:{}\nonset:{}(100frames/s)".format(len(k)/sr,count))


plt.plot(k)
plt.plot(new_onset)
plt.plot(percussive)
plt.title("Onset of Piano")
plt.show()