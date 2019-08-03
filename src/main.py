
from src.Note_Match import *

piano_wav = '../piano/MAPS_ISOL_CH0.3_F_AkPnBcht.wav'
piano_midi = '../piano/MAPS_ISOL_CH0.3_F_AkPnBcht.mid'

chopin_wav = '/Code/PyCharm/LivePiano/piano/chopin_nocturne_b49.wav'
chopin_midi = '/Code/PyCharm/LivePiano/piano/chopin_nocturne_b49.mid'

y,sr = librosa.load(chopin_wav,None)
min_note,max_note,max_concurrence,concurrence,spv1,spv2,spv3 = SPVs(midi=chopin_midi)
#y = y[:50*sr]
margin = 0.9
onset_count = -1

block = 0.1#s
n_block = int((len(y)/sr)/block)
print(n_block)
buff = np.zeros(int(3 * block * sr))

for i in range(n_block-2):
    if i == 0:
        buff[int(block * sr):int(block * sr * 2 - 1)] = y[:int(block * sr - 1)]
    elif i == 1:
        buff[:int(block * sr * 2 - 1)] = y[:int(block * sr * 2 - 1)]
    else:
        buff = y[int((i - 1) * block * sr):int((i + 2) * block * sr)]
    onset = Onset(buff,margin=margin)
    if onset == -1:
        continue
    else:
        onset_count +=1
        mcqs = MCQS(y=buff,sr=sr,
                    from_note=min_note,to_note=max_note+1,max_n=max_concurrence*6)
        sdv = SDVs(mcqs,onset=onset)
        similarity = Similarity(sdv,concurrence,spv1,spv2,spv3,onset_count)
        print("sdv:\n",sdv)
        print("spv3:\n",spv3[onset_count])
        print("onset:",onset_count,"similarity:",similarity)



