from src.Note_Match import *
import matplotlib.pyplot as plt

piano_wav = '../piano/MAPS_ISOL_CH0.3_F_AkPnBcht.wav'
piano_midi = '../piano/MAPS_ISOL_CH0.3_F_AkPnBcht.mid'

chopin_wav = '/Code/PyCharm/LivePiano/piano/chopin_nocturne_b49.wav'
chopin_midi = '/Code/PyCharm/LivePiano/piano/chopin_nocturne_b49.mid'

y,sr = librosa.load(piano_wav,None)
min_note,max_note,max_concurrence,concurrence,spv1,spv2,spv3,concurrence_time = SPVs(midi=piano_midi)
onset_time = np.zeros(len(concurrence) * 2)#onset时间

#y = y[:50*sr]
margin = 0.9
onset_count = 0
score_count = 0# current matched score count
j0 = 0# j0 is the index of the previous matched score concurrence.

block = 0.1#s
n_block = int((len(y)/sr)/block)
buff = np.zeros(int(3 * block * sr))
S = np.zeros((len(concurrence) + 1,len(concurrence) * 2))
DP = np.zeros((len(concurrence) + 1,len(concurrence) * 2))
matched = [[0,0]]#score_count, onset_count

for i in range(n_block-2):
    if i == 0:
        buff[int(block * sr):int(block * sr * 2)] = y[:int(block * sr)]
    elif i == 1:
        buff[:int(block * sr * 2)] = y[:int(block * sr * 2)]
    else:
        buff = y[int((i - 1) * block * sr):int((i + 2) * block * sr)]
    onset = Onset(buff,margin=margin)
    if onset == -1:
        continue
    else:
        onset_count +=1
        begin_block = i
        if onset_count == 1:
            onset_time[onset_count] = concurrence_time[0]
        else:
            onset_time[onset_count] = onset_time[0] + (i - begin_block) * block + onset * 0.01
        mcqs = MCQS(y=buff,sr=sr,
                    from_note=min_note,to_note=max_note,
                    max_n=max_concurrence*6)
        sdv = SDVs(mcqs,onset=onset)

        ja = matched[score_count][0]
        j0, j1 = Get_j(DP, i=onset_count, ja=ja, aligned=matched,
                       concurrence_time=concurrence_time[ja],
                       onset_time=onset_time[onset_count])
        tempo, ita = Ita(aligned=matched,concurrence_time=concurrence_time,
                         onset_time=onset_time,i=onset_count,j=j1,j0=j0,a=0.1)
        Update_Matrix(S=S, DP=DP, sdv=sdv, concurrence=concurrence,
                      spv1=spv1, spv2=spv2, spv3=spv3,
                      onset_index=onset_count, ita=ita, scope=10)
        if tempo > 4:
            # Insersion
            continue
        else:
            if j1 == j0 + 1:
                # Matched
                current_match = [onset_count,j1]
                score_count = j1
            else:
                # Looking-ahead note matching
                j2 = j0 + 1
                maxS = S[onset_count][j2]
                for j in range(j2,j1+1):
                    if S[onset_count][j] > maxS:
                        maxS = S[onset_count][j]
                        j2 = j
                current_match = [onset_count,j2]
                score_count = j2
            matched.append(current_match)
            print("current matched pair:",current_match)





