import librosa
for i in range(21,108):
    print(i," Hz:",librosa.midi_to_hz(i)," Note:",librosa.midi_to_note(i))