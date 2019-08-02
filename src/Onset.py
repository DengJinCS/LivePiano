import madmom
def Onset(y):
    CNNOnset = madmom.features.onsets.CNNOnsetProcessor()
    RNNOnset = madmom.features.onsets.RNNOnsetProcessor()
    # superflux = madmom.features.onsets.superflux(spectrogram=spec)
    # complexflux = madmom.features.onsets.complex_flux(spectrogram=spec)
    # percussive = librosa.effects.percussive(y)

    cnn_onset = CNNOnset(y)
    rnn_onset = RNNOnset(y)
    #onset = (cnn_onset + rnn_onset) / 2
    onset = cnn_onset
    max_index, max_onset = 0, 0
    for index in range(len(onset)):
        if onset[index] > max_onset:
            max_index = index
            max_onset = onset[index]
    return max_index,max_onset