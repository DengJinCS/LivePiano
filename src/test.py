from src.Note_Match import *
import librosa
piano = '/Code/PyCharm/LivePiano/piano/MAPS_ISOL_CH0.3_F_AkPnBcht.wav'
midi  = '/Code/PyCharm/LivePiano/piano/MAPS_ISOL_CH0.3_F_AkPnBcht.mid'

chopin = '/Code/PyCharm/LivePiano/piano/chopin_nocturne_b49.wav'
chopin_midi = '/Code/PyCharm/LivePiano/piano/chopin_nocturne_b49.mid'

min_note,max_note,max_concurrence,concurrence,spv1,spv2,spv3,concurrence_time = SPVs(midi=midi)
y,sr = librosa.load(piano,None)
k = y[:int(0.3*sr)]
S = 0
for onset_index in range(len(concurrence_time)):
    onset_time = concurrence_time[onset_index]
    if onset_time == 0:
        #k = y[0.1 * sr]
        continue
    else:
        k = y[int((onset_time - 0.15) * sr):int((onset_time + 0.15) * sr)]

    M = MCQS(y=k, sr=sr,
                from_note=min_note, to_note=max_note,
                max_n=max_concurrence * 2)
    T_MCQS = M.transpose()
    V_before_onset = T_MCQS[:int(len(T_MCQS)/2)]
    V_after_onset = T_MCQS[int(len(T_MCQS)/2):]

    Mean_before = np.zeros_like(T_MCQS[0])
    Mean_after = np.zeros_like(T_MCQS[0])

    dA = np.zeros_like(T_MCQS[0])
    dAmax = 0
    Psi = np.zeros_like(T_MCQS[0])
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
        if dAmax < dA[k]:
            dAmax = dA[k]
    for k in range(len(T_MCQS[0])):
        if dA[k] > dAmax / 20:
            Psi[k] = dA[k]
    sdv = Psi

    correlation1 = pearsonr(sdv, concurrence[onset_index])[0]
    correlation2 = pearsonr(sdv, spv1[onset_index])[0]
    correlation3 = pearsonr(sdv, spv2[onset_index])[0]
    correlation4 = pearsonr(sdv, spv3[onset_index])[0]
    similarity = max(correlation1, correlation2, correlation3, correlation4)
    S += similarity

    print(onset_index,int(similarity*100))
print("avr_S:",S/(len(concurrence_time) - 1))
