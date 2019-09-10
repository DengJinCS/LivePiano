import SDV
import SPV
import numpy as np
from math import log
from scipy.stats.stats import pearsonr
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


def main(**params):

    spv = SPV.SPV(**params)
    min_note, max_note, max_concurrence, concurrence, spv1, spv2, spv3, concurrence_time = spv.get_spv()
    print("total onset：",len(concurrence_time))
    onset_true_time = np.zeros(concurrence.shape[0] * 2) # 每一个onset的时间
    sdv = SDV.SDV(**params)
    similarity_matrix = np.zeros((len(concurrence)+1, len(concurrence)*2))
    DP = np.zeros((len(concurrence)+1, len(concurrence)*2))
    matched_pair = [[0, 0]] # [score_time,onset_time]
    score_count = 0 # 当前匹配到的concurrence索引
    #audio = sdv.audio[:int(50*sdv.sr)]
    audio = sdv.audio
    block_length = 0.1
    buffer_block = np.zeros(int(3*block_length*sdv.sr))
    number_block = int((len(audio)/sdv.sr)//block_length)
    onset_count = 0  # 外部的onset计数器
    first_block_index = 0 # 第一个onset出现block区的索引
    """
    处理是否平滑，保存前面局部正常的瞬时速度
    """
    offset = 0 #比正常标准值的偏移量
    smooth = np.zeros(len(concurrence))
    tempo = 1 #初始化为默认跟随midi速度
    ita = 0.1
    for i in range(number_block-1):
        if i == 0:
            buffer_block[int(block_length*sdv.sr):int(2*block_length*sdv.sr)] \
                = audio[:int(block_length*sdv.sr)]
        elif i == 1:
            buffer_block[:int(2*block_length*sdv.sr)] = audio[:int(2*block_length*sdv.sr)]
        else:
            buffer_block = audio[int((i-1)*block_length*sdv.sr):int((i+2)*block_length*sdv.sr)]
        index_onset, _ = sdv.Onset_Detector(buffer_block)
        if index_onset == -1:
            continue
        else:
            print("===============================================================================")
            print(f"Onset at block-{i}/{number_block}")
            onset_count += 1
            if onset_count == 1:
                onset_true_time[onset_count] = concurrence_time[1]
                first_block_index = i
            else:
                time = onset_true_time[1] + (i-first_block_index)*block_length + index_onset * 0.01
                if abs(time - matched_pair[-1][1]) < 0.07: # 相隔小于0.08秒被认为是误检onset
                    onset_count -= 1
                    continue
                onset_true_time[onset_count] = onset_true_time[1] + (i-first_block_index)*block_length + index_onset * 0.01

        print(f"onset_time: {onset_true_time[onset_count]}, onset_count: {onset_count}")
        SDVs = sdv.get_SDV(buffer_block, index_onset, min_note, max_note)
        ja = score_count
        j0, j1, pre_time,new_offset = Get_J(DP, i=onset_count, ja=ja, aligned_path=matched_pair,concurrence_time=concurrence_time,
                       onset_time=onset_true_time,offset=offset,smooth=smooth)
        offset = new_offset
        print(f"j0:{j0}, j1:{j1}, offse:{offset}, pre_time:{pre_time}\n")
        if j0 == j1 and j0 != 0: #j0是不能等于j1的，j0一开始等于0也不需要考虑
            j0 -= 1
        tempo, ita = Ita(aligned_path=matched_pair, concurrence_time=concurrence_time, onset_time=onset_true_time
                         , i=onset_count, j0=j0, j=j1, a=0.1)

        Update_Similarity_Matrix(S=similarity_matrix, DP=DP,sdv=SDVs,concurrence=concurrence,spv1=spv1
                                 ,spv2=spv2,spv3=spv3,ita=ita, onset_count=onset_count, scope=10)
        print(f"tempo={tempo}\nIta={ita}")
        if tempo > 10:  # Insertion
            continue
        else:
            if j1 == j0 + 1:
                #正常速度匹配
                current_match = [concurrence_time[j1], onset_true_time[onset_count]]
                print(f"S={similarity_matrix[j1][onset_count]}")
                score_count = j1
            else:
                #带异常处理
                jumped_s_pre = similarity_matrix[j1 - 1][onset_count]
                jumped_s = similarity_matrix[j1][onset_count]
                jumped_s_next = similarity_matrix[j1 + 1][onset_count]

                next_s_pre = similarity_matrix[j0][onset_count]
                next_s = similarity_matrix[j0 + 1][onset_count]
                next_s_next = similarity_matrix[j0 + 2][onset_count]
                if jumped_s > next_s and \
                        jumped_s >= jumped_s_pre and jumped_s >= jumped_s_next:#区分跳跃和卡顿   比对的峰值
                    """ 
                    ##############################   处理漏检测或者/ScoreJump  ##############################
                    """
                    print("S with 【Jumped】 Next Score：", jumped_s)
                    for jumped in range(score_count,j1):
                        smooth[jumped] = 2#漏检/跳音标记为2

                    score_count = j1
                    current_match = [concurrence_time[j1], onset_true_time[onset_count]]
                elif next_s >= next_s_pre and next_s >= next_s_next:
                    """
                    ##############################     处理长时间卡顿/Jump     ##############################
                    """
                    # 可能是一时跟不上谱子，这个onset比标准谱时间晚，按照前面的平均速率，这个onset已经和后面的score时间对齐了
                    # 这时尝试直接跟下一个比对相似性。
                    # 需要更新瞬时速度，因为卡住之后的瞬时速度已经很慢了，属于异常值，不能用作下一个j1的预测
                    print("S with Next Score：", next_s)
                    offset = onset_true_time[onset_count] - pre_time
                    j1 = j0 + 1
                    smooth[j1] = 1#卡顿标记为1
                    current_match = [concurrence_time[j1], onset_true_time[onset_count]]
                    print(f"*********************************************")
                    print(f"YOU MISS THE BEATS!!!\nS={similarity_matrix[j1][onset_count]}")
                    print(f"*********************************************")
                    score_count = j1
                # 可能是用户跳过了一些音符
                # 从已经对齐的j0开始往后搜索到预测的score点，找最可能匹配上的score
                else:
                    j2 = j0 + 1
                    max_S = 0
                    for j in range(j2, j1+1):
                        if similarity_matrix[onset_count][j] > max_S:
                            max_S = similarity_matrix[onset_count][j]
                            j2 = j
                    current_match = [concurrence_time[j2], onset_true_time[onset_count]]
                    print(f"S={similarity_matrix[j2][onset_count]}")
                    score_count = j2
                    print(f"j2: {j2}")
                    # if abs(concurrence_time[j2] - onset_true_time[onset_count]) > 0.1:
                    #     score_count += 1
            matched_pair.append(current_match)
            print(f"current matched pair: {current_match}\n\n")
            yield matched_pair

def Update_Similarity_Matrix(S, DP, sdv, concurrence, spv1, spv2, spv3, ita, onset_count, scope=8):
    """

    :param S: similatity matrix
    :param sdv:
    :param concurrence:
    :param spv1:
    :param spv2:
    :param spv3:
    :param onset_count: the index of onset was detected
    :param scope: the number of notes should be considerated when onset was detected
    :return:
    """

    if onset_count - scope < 0:
        low_index = onset_count
    else:
        low_index = onset_count - scope
    if onset_count + scope > len(concurrence):
        high_index = len(concurrence)
    else:
        high_index = onset_count + scope
    for j in range(low_index, high_index):
        similarity1 = pearsonr(sdv, concurrence[j])[0]
        similarity2 = pearsonr(sdv, spv1[j])[0]
        similarity3 = pearsonr(sdv, spv2[j])[0]
        similarity4 = pearsonr(sdv, spv3[j])[0]

        similarity = max(similarity1, similarity2, similarity3, similarity4)
        S[onset_count][j] = similarity
        #print(f"S[{onset_count}][{j}] = {similarity}")
    #print("=================================")
    for j in range(low_index, high_index):
        dp1 = DP[onset_count-1][j]
        dp2 = DP[onset_count][j-1]
        dp3 = DP[onset_count-1][j-1] + S[onset_count][j] + ita
        dp = max(dp1, dp2, dp3)
        DP[onset_count][j] = dp
        #print(f"DP[{onset_count}][{j}] = {dp}")

def Get_J(DP, i, ja, aligned_path, concurrence_time, onset_time, smooth, offset=0, delta_j=3,line_cope = 10):
    aligned_path = np.array(aligned_path)
    maxD, minT1, minT2 = 0, 1000, 1000
    new_offset = offset
    j0, j1 = 0, 0
    print(f"ja: {ja}, smooth[{ja}]:{smooth[ja-1]},{smooth[ja]}")
    for j in range(ja-delta_j, ja+delta_j+1):
        if DP[i-1][j] > maxD:
            maxD = DP[i-1][j]
            j0 = j
    if ja != j0:
        j0 = ja

    if ja == 0:
        for j in range(j0, j0+delta_j+1):
            if abs(concurrence_time[j]- onset_time[i]) < minT1:
                minT1 = abs(concurrence_time[j]-onset_time[i])
                j1 = j
        pre_next_onset_time = concurrence_time[ja + 1]  + new_offset  # 根据score预测onset时间
    else:
        end_index  = ja + 1
        if ja > 0 and ja < line_cope:
            start_index = 0

        else:
            start_index,tmp_start_index = ja-line_cope+1, ja-line_cope+1# 处理不光滑点
            tmp_N = 0
            for k in range(tmp_start_index, end_index):
                if smooth[k] == 0:
                    break
                tmp_N += 1
            if tmp_N < line_cope/2:
                new_offset = 0 #回到正常值了
                start_index = tmp_start_index + tmp_N + 1
            else:
                new_offset = offset
                end_index = tmp_start_index + tmp_N
        print(f"start:{start_index}, end:{end_index}")

        N = end_index - start_index
        if aligned_path[start_index, 0] == 0 and aligned_path[start_index+1, 0] == 0: # 第一个onset出现在0s， 由于aligned_path 第一个元素初始化也为[0,0]
            N += 1
        #print(f"Aligned_path: {aligned_path[start_index:end_index, 0]}")
        sum1 = np.sum(aligned_path[start_index:end_index, 0]*aligned_path[start_index:end_index, 1])
        sum2 = np.sum(aligned_path[start_index:end_index, 1]**2)
        x_mean = np.mean(aligned_path[start_index:end_index, 1])
        y_mean = np.mean(aligned_path[start_index:end_index, 0])
        w = (sum1 - N*(x_mean*y_mean))/(sum2- N*(x_mean**2))
        b = y_mean - w * x_mean
        print(f"Regression_Line: y = {w}x + {b}")
        print(f"Correction_Line: y = {w}(x - {new_offset}) + {b}")
        predicted_time = (concurrence_time[j0:j0+delta_j+1]-b)/w + new_offset
        #print(f"Predicted_time: {predicted_time}")
        for j in range(j0, j0+delta_j+1):
            predicted_time = (concurrence_time[j]-b)/w + new_offset#根据score预测onset时间
            if abs(predicted_time - onset_time[i]) < minT2:
                minT2 = abs(predicted_time - onset_time[i])
                j1 = j
        pre_next_onset_time = (concurrence_time[j0 + 1]-b)/w + new_offset#根据score预测onset时间
    return  j0, j1, pre_next_onset_time,new_offset

def Ita(aligned_path, concurrence_time, onset_time, i, j0, j, a=0.1,smooth=True):
    if j0 == 0:
        tempo = 0
    elif j0 == 1:
        tempo = aligned_path[j0][0] / aligned_path[j0][1]
    else:
        m1 = (concurrence_time[j] - aligned_path[-2][0]) / (onset_time[i] - aligned_path[-2][1])
        m2 = (aligned_path[-2][0] - aligned_path[-3][0]) / (aligned_path[-2][1] - aligned_path[-3][1])
        tempo = m1 / m2
    if j0 != 0 and tempo < 4:
        delta_j = j - j0 - 1
        ita = a * (1 - abs(log(tempo, 2) - delta_j))
    elif j0 == 0:  # 如果直接返回ita为-1，Dp矩阵一开始就不会更新，因为ita(i, j)会大于S（i，j）
        ita = 0
    else:
        ita = -1
    return tempo, ita


    # spv = SPV(**params)
    # min_note, max_note, max_concurrence, concurrence, spv1, spv2, spv3, concurrence_time = spv.get_spv()
    # print(f"max_concurrence: {max_concurrence}")
    # print(concurrence_time)
    # print(f"spv1: {spv1}")
    # print(f"spv2: {spv2}")
    # print(f"spv3: {spv3}")
    # sdv = SDV(**params)
    # index, onset = sdv.Onset_Detector(sdv.audio)
    # print(index, onset)
    # sdv = sdv.get_SDV(index)
    # print(sdv)
    # sdv.Calculate_Onset_Recall()
    # sdv.Realtime_Capture_Onset()

def init():
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.set_xlabel("Audio: Onset time(Sec)")
    ax.set_ylabel("Score: Concurrence time(Sec)")
    return ln,

def update(frame):
    xdata.append(frame)
    ydata.append(frame)
    ln.set_data(xdata, ydata)
    return ln,

params = {
        'audio_path': '/Code/PA/LivePiano_new/piano/test_R.wav',
        'midi_path': '/Code/PA/LivePiano_new/piano/MAPS_MUS-grieg_wanderer_AkPnBsdf.mid'
    }
frame = main(**params)
spv = SPV.SPV(**params)
min_note, max_note, max_concurrence, concurrence, spv1, spv2, spv3, concurrence_time = spv.get_spv()
fig, ax = plt.subplots(figsize = (10,10))
xdata, ydata = [], []
plt.plot(concurrence_time, concurrence_time, 'r-')
ln, = plt.plot([], [], 'gv')

ani = FuncAnimation(fig, update, frames=frame,
                    init_func=init, blit=True)
plt.show()






