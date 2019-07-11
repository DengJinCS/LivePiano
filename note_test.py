# Generate and plot a constant-Q power spectrum
import numpy as np
import librosa.display
import matplotlib.pyplot as plt
music = 'piano/MAPS_ISOL_CH0.3_F_AkPnBcht.wav'
y, sr = librosa.load(music)
C = np.abs(librosa.cqt(y, sr=sr, fmin=librosa.note_to_hz('C5'),
                n_bins=60 * 2, bins_per_octave=12 * 3))
librosa.display.specshow(librosa.amplitude_to_db(C, ref=np.max),
                         sr=sr, x_axis='time', y_axis='cqt_note')
plt.colorbar(format='%+2.0f dB')
plt.title('Nocturne Constant-Q power spectrum')
plt.tight_layout()
plt.show()

# Limit the frequency range
"""
C = np.abs(librosa.cqt(y, sr=sr, fmin=librosa.note_to_hz('C2'),
                n_bins=60))


# Using a higher frequency resolution

C = np.abs(librosa.cqt(y, sr=sr, fmin=librosa.note_to_hz('C2'),
                n_bins=60 * 2, bins_per_octave=12 * 2))

"""