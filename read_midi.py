from mxm.midifile import MidiInFile,MidiToCode
midi = '/Code/PyCharm/LivePiano/piano/MAPS_ISOL_CH0.3_F_AkPnBcht.mid'

midiIn = MidiInFile(MidiToCode(),midi)
midiIn.read()