import numpy as np
import librosa
import madmom
from mido import MidiFile
from scipy.signal import argrelextrema
from math import log10
import pretty_midi
from scipy.stats.stats import pearsonr

def Onset(y,margin):
    CNNOnset = madmom.features.onsets.CNNOnsetProcessor()
    RNNOnset = madmom.features.onsets.RNNOnsetProcessor()
    # superflux = madmom.features.onsets.superflux(spectrogram=spec)
    # complexflux = madmom.features.onsets.complex_flux(spectrogram=spec)
    # percussive = librosa.effects.percussive(y)

    cnn_onset = CNNOnset(y)
    index1 = int(len(cnn_onset)/3) -1
    index2 = int(len(cnn_onset)*2/3) - 1
    target = cnn_onset[index1:index2]#中间窗口
    max_index, max_onset = 0, 0
    for index in range(index1+1):
        if target[index] > max_onset:
            max_index = index
            max_onset = target[index]
    if max_index == 0 and max_onset < cnn_onset[index1-1]:
        max_onset = 0
        max_index = -1
    if max_index == index1 - 1 and max_onset < cnn_onset[index2+1]:
        max_onset = 0
        max_index = -1
    if max_onset < margin:
        max_index = -1
    return max_index

def MCQS(y,sr,from_note=21,to_note=108,max_n = 12):
    #caculate the CQT spectrum
    #retaining only the local maximum pitch
    #restrict the top max_n pitch
    #return the modified CQT Spectrum

    n_bins = to_note - from_note + 26
    C = np.abs(librosa.cqt(y, sr=sr,hop_length=512, bins_per_octave=12,
                           window='hamm', fmin=librosa.midi_to_hz(from_note),
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
    #print("Normlized:{}".format(MCQS.shape))
    T_MCQS = MCQS.transpose()

    return T_MCQS

def SDVs(MCQS,type=1,onset=-1,onset_len = 30):
    # Spectrum Difference Vectors (SDVs) ψi
    # dAi is the difference between these two vectors
    T_MCQS = MCQS.transpose()
    bias = int(len(T_MCQS) / 3)
    index = int(onset * len(T_MCQS) / onset_len)
    if onset != -1:
        V_before_onset = T_MCQS[index:index+bias]
        V_after_onset = T_MCQS[index+bias:index+2*bias]

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
            if type == 1 and dA[k] > dAmax/20:
                Psi[k] = dA[k]
            elif type == 2 and dA[k] > dAmax/20:
                Psi[k] = 1
            elif type == 3 and dA[k] > dAmax/20:
                Psi[k] = log10(dA[k])
        return Psi

def SPVs(midi='../piano/chopin_nocturne_b49.mid'):
    mid = MidiFile(midi)
    note_on_count = 0
    time = 0 # onset time
    onset_count = 1 # onset count, retaining 0 index
    onset_index = 0 # retaining 0 index
    min_note = 108 # minimum note midi
    max_note = 0 # maxmum note midi
    max_concurrence = 1 #max concurrence count in whole concurrence
    cur_concurrence = 1 #concurrence count in every concurrence
    for msg in mid:
        if msg.type != 'note_on' and msg.type != 'note_off':
            continue
        if msg.type == "note_on":
            if msg.note > max_note:
                max_note = msg.note
            if msg.note < min_note:
                min_note = msg.note
        pre_time = time
        if msg.time != 0:
            time += msg.time
        if cur_concurrence > max_concurrence:
            max_concurrence = cur_concurrence
        if msg.type == "note_on":
            note_on_count += 1
            if note_on_count == 1 and msg.time == 0:
                onset_count += 1
                cur_concurrence = 1
            if pre_time != time:
                onset_count += 1
                cur_concurrence = 1
            else:
                cur_concurrence += 1

    note_range = max_note - min_note + 26
    #print(min_note,max_note,note_range)
    Concurrence = np.zeros((onset_count, note_range), dtype=int)
    Concurrence_time = np.zeros(onset_count)
    SPV1 = np.zeros((onset_count, note_range), dtype=int)
    SPV2 = np.zeros((onset_count, note_range), dtype=int)
    SPV3 = np.zeros((onset_count, note_range), dtype=int)


    time = 0
    note_on_count = 0
    for msg in mid:
        if msg.type != 'note_on' and msg.type != 'note_off':
            continue
        pre_time = time
        if msg.time != 0:
            time += msg.time
        if msg.type == "note_on":
            note_on_count += 1
            if note_on_count == 1 and msg.time == 0:
                #print(onset_index,time,msg)
                onset_index += 1
            if pre_time != time:
                onset_index += 1
            Concurrence_time[onset_index] = time
            """
            [ 0] PitchItself
            [12] SecondOctave
            [19] PerfectFifth
            [24] Third Octave
            """

            Concurrence[onset_index][msg.note - min_note + 0] = 1
            for over_tune in [0,12]:
                index = msg.note - min_note + over_tune
                if index < note_range:
                    SPV1[onset_index][index] = 1
            for over_tune in [0,12,19]:
                index = msg.note - min_note + over_tune
                if index < note_range:
                    SPV2[onset_index][index] = 1
            for over_tune in [0, 12, 19, 24, 28, 31, 34]:
                index = msg.note - min_note + over_tune
                if index < note_range:
                    SPV3[onset_index][index] = 1
    return min_note,max_note,max_concurrence,Concurrence,SPV1,SPV2,SPV3,Concurrence_time

def ImprovedSPVs(midi='../piano/chopin_nocturne_b49.mid',step = 0.1):
    mid = MidiFile(midi)
    note_on_count = 0
    time = 0 # onset time
    onset_count = 1 # onset count, retaining 0 index
    onset_index = 0 # retaining 0 index
    min_note = 108 # minimum note midi
    max_note = 0 # maxmum note midi
    max_concurrence = 1 #max concurrence count in whole concurrence
    cur_concurrence = 1 #concurrence count in every concurrence
    for msg in mid:
        if msg.type != 'note_on' and msg.type != 'note_off':
            continue
        if msg.type == "note_on":
            if msg.note > max_note:
                max_note = msg.note
            if msg.note < min_note:
                min_note = msg.note
        pre_time = time
        if msg.time != 0:
            time += msg.time
        if cur_concurrence > max_concurrence:
            max_concurrence = cur_concurrence
        if msg.type == "note_on":
            note_on_count += 1
            if note_on_count == 1 and msg.time == 0:
                onset_count += 1
                cur_concurrence = 1
            if pre_time != time:
                onset_count += 1
                cur_concurrence = 1
            else:
                cur_concurrence += 1

    note_range = max_note - min_note + 26
    print("range:",max_note,min_note,note_range)
    Concurrence_time = np.zeros(onset_count)

    Concurrence = np.zeros((onset_count, note_range))
    SPV1 = np.zeros((onset_count, note_range))
    SPV2 = np.zeros((onset_count, note_range))
    SPV3 = np.zeros((onset_count, note_range))

    """
    synthesized from midi
    get CQT
    """
    midi_data = pretty_midi.PrettyMIDI(midi)
    y, sr = midi_data.synthesize(), 44100
    print("audio synthesized from", midi)
    n_bins = max_note - min_note + 26
    print("bins:",n_bins)
    C = np.abs(librosa.cqt(y, sr=sr, hop_length=512, bins_per_octave=12,
                           window='hamm', fmin=librosa.midi_to_hz(min_note),
                           n_bins=n_bins))
    CQT = C.transpose()
    A2dB = librosa.amplitude_to_db(CQT, ref=np.max)
    cqt_per_second = len(A2dB) / len(y) * sr
    """
    synthesized from midi
    get CQT
    """

    time = 0
    note_on_count = 0
    for msg in mid:
        if msg.type != 'note_on' and msg.type != 'note_off':
            continue
        pre_time = time
        if msg.time != 0:
            time += msg.time
        if msg.type == "note_on":
            note_on_count += 1
            if note_on_count == 1 and msg.time == 0:
                #print(onset_index,time,msg)
                onset_index += 1
            if pre_time != time:
                onset_index += 1
            Concurrence_time[onset_index] = time
            """
            [ 0] PitchItself
            [12] SecondOctave
            [19] PerfectFifth
            [24] Third Octave
            """
            index1, index2 = int(time * cqt_per_second), int((time + step) * cqt_per_second)
            print(index2-index1)
            Mean_CQT = np.mean(A2dB[index1:index2],axis=0)

            Concurrence[onset_index][msg.note - min_note + 0] = 80 + Mean_CQT[msg.note - min_note + 0]
            for over_tune in [0,12]:
                index = msg.note - min_note + over_tune
                if index < note_range:
                    SPV1[onset_index][index] = 80 + Mean_CQT[index]
            for over_tune in [0,12,19]:
                index = msg.note - min_note + over_tune
                if index < note_range:
                    SPV2[onset_index][index] = 80 + Mean_CQT[index]
            for over_tune in [0, 12, 19, 24, 28, 31, 34]:
                index = msg.note - min_note + over_tune
                if index < note_range:
                    SPV3[onset_index][index] = 80 + Mean_CQT[index]

    return min_note,max_note,max_concurrence,Concurrence,SPV1,SPV2,SPV3,Concurrence_time

def Update_Matrix(S,DP,sdv,concurrence,spv1,spv2,spv3,ita=0,onset_index = 0,scope = 10):
    if onset_index - scope < 0:
        low_index = 0
    else:
        low_index = onset_index - scope
    if onset_index + scope > len(concurrence):
        high_index = len(concurrence)
    else:
        high_index = onset_index + scope

    #update Similarity matrix
    for j in range(low_index,high_index):
        correlation1 = pearsonr(sdv, concurrence[j])[0]
        correlation2 = pearsonr(sdv, spv1[j])[0]
        correlation3 = pearsonr(sdv, spv2[j])[0]
        correlation4 = pearsonr(sdv, spv3[j])[0]
        similarity =  max(correlation1, correlation2, correlation3, correlation4)
        S[onset_index][j] = similarity

    # update DP matrix
    for j in range(low_index, high_index):
        dp1 = DP[onset_index - 1][j]
        dp2 = DP[onset_index][j - 1]
        dp3 = DP[onset_index - 1][j - 1] + S[onset_index][j] + ita
        dp = max(dp1, dp2, dp3)
        DP[onset_index][j] = dp

def Get_j(DP,i,ja,aligned,concurrence_time,onset_time,delta_j = 5):
    # DP (Dynamic programming ) is employed to determine the path with maximum overall similarity.
    # i is the ith onset
    # ja is the index of the previous matched concurrence and
    # Δj is a tolerance window.
    # concurrence_time from midi score
    # onset_time form real time onset
    maxD, minT = 0,1000000
    j0, j1 = 0, 0
    for j in range(ja - delta_j, ja + delta_j + 1):
        if DP[i-1][j] > maxD:
            maxD = DP[i-1][j]
            j0 = j
    """
        最小二乘法求回归直线
        预测下一个concurrence可能对应的onset_time
        loss = 1/N * ∑ (Yi - (mXi + b))^2
        ∂loss/∂m = -2/N * ∑(Xi(Yi - (mXi + b))) = 0
        ∂loss/∂b = -2/N * ∑(Yi - (mXi + b)) = 0
    """
    if ja == 0:
        for j in range(j0,j0 + delta_j + 1):
            if abs(concurrence_time[j] - onset_time[i]) < minT:
                j1 = j
    else:
        # back tracing 5(j0) steps from(i-1, j0)
        if ja > 0 and ja < 5:
            index = 0
            N = ja
        else:
            index = ja - 4
            N = 5
        # sum1 =  ∑XiYi Xi->onset_index
        # sum2 =  ∑XiXi Yi->concurrence_index
        # sum3 =  ∑Xi∑Xi
        sum1, sum2, sum3, sum4 = 0, 0, 0, 0
        for j in range(index, ja + 1):
            # sum1 =  ∑XiYi Xi->onset_index
            # sum2 =  ∑XiXi Yi->concurrence_index
            # sum3 =  ∑Xi∑Xi
            # sum4 =  ∑Xi∑Yi
            # aligned[j][0] -> concurrence,Y
            # aligned[j][1] -> onset,X
            sum1 += aligned[j][1] * aligned[j][0]
            sum2 += aligned[j][1] * aligned[j][1]
            sum3 += aligned[j][1]
            sum4 += aligned[j][0]
        m = (N*sum1 - sum3*sum4) / (N*sum2 - sum3*sum3)
        b = (sum4/N - m*sum3/N)
        for j in range(j0,j0 + delta_j + 1):
            predicted_time = (concurrence_time[j] - b) / m
            if abs(predicted_time - onset_time[i]) < minT:
                j1 = j
    return j0,j1

def Ita(aligned,concurrence_time,onset_time,i,j,j0,a = 0.1):
    # ita means the local tempo coefficient
    # m1 & m2 are the slopes of the latest two segments
    # i is the onset index
    # j is the index of the new matched concurrence
    # j0 is the index of the previous matched concurrence
    # delta_j is a penalty to avoid injurious score jumps & onset insertions
    if j0 == 0:
        tempo = 0
    elif j0 == 1:
        tempo = aligned[1][0] / aligned[1][1]
    else:
        m1 = (concurrence_time[j] - aligned[j0][0]) / (onset_time[i] - aligned[j0][1])
        m2 = (aligned[j0][0] - aligned[j0-1][0]) / (aligned[j0][1] - aligned[j0-1][1])
        tempo = m1/m2
    delta_j = j - j0 - 1
    if j0 != 0 and tempo < 4:
        ita = a * (1 - abs(log10(tempo) / log10(2)) - delta_j)
    else:
        ita = -1
    return tempo,ita
#sdv = MCQS(y,sr,min_note,max_note,max_concurrence)
#S = Similarity(sdv,Concurrence,spv1,spv2,spv3)
