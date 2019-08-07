import madmom
import librosa
import matplotlib.pyplot as plt
import numpy as np
from math import log10
from mido import MidiFile
from scipy.signal import argrelextrema

class SDV():
    def __init__(self, **params):
        self.audio_path = params['audio_path']
        self.midi_path = params['midi_path']
        self.audio, self.sr = librosa.load(self.audio_path, None)
        self.onset_capture_window = 0.05  # the window used to capture onset in a block of audio
        self.block_length = int(self.onset_capture_window * self.sr)  # block length
        self.onset_bound = 0.7   # the bound of a possible onset
        self.frames_threshold = 5  # the frame numbers can't longer than a onset intervel or a threshold
        self.frame_length = 0.01  # onset detector samples 100 frames per second
        self.concurrence = 5  # concurrence
        self.sdv_type = 1 # the type of SDV vectors

    def Onset_Detector(self, y):
        CNNOnsetProcessor = madmom.features.onsets.CNNOnsetProcessor()
        RNNOnsetProcessor = madmom.features.onsets.RNNOnsetProcessor()

        cnn_results_onset = CNNOnsetProcessor(y)
        # rnn_results_onset = RNNOnsetProcessor(y)
        # onset = (cnn_results_onset + rnn_results_onset)/2
        onset = cnn_results_onset
        start_point_1 = int(onset.shape[0]/3)
        start_point_2 = int(2*onset.shape[0]/3)
        target = onset[start_point_1:start_point_2]
        max_onset, index_onset = 0, 0
        for index in range(start_point_1):
            if target[index] > max_onset:
                max_onset = target[index]
                index_onset = index
        if max_onset < self.onset_bound:
            index_onset = -1
        if index_onset == 0 and max_onset < onset[start_point_1-1]:
            index_onset = -1
            max_onset = 0
        if index_onset == start_point_1-1 and max_onset < onset[start_point_2]:
            index_onset = -1
            max_onset = 0
        return index_onset, max_onset


    def Realtime_Capture_Onset(self, plot=False):
        onset_x = []
        onset_y = []

        index_onsettime = []
        pre_index_onset, pre_onset = 0, 0
        counts_onset = 0
        # audio = self.audio[:10 * self.sr]
        audio = self.audio
        number_block = int(len(audio)/int(self.onset_capture_window*self.sr))
        for i in range(number_block):
            block = audio[i*self.block_length:(i+1)*self.block_length]
            index, onset = self.Onset_Detector(block)

            if index == self.block_length/self.frame_length -1:  # onset 出现在一个窗口的最后一帧
                i += 0.5
                pre_index_onset, pre_onset = index, onset
                new_block = audio[i*self.block_length:(i+1)*self.block_length]
                index, onset = self.Onset_Detector(new_block)
                if pre_onset == onset:  # onset 出现在两个窗口的中间
                    counts_onset += 1
                    index, onset = pre_index_onset, pre_onset
                else: # onset 出现在新的窗口中
                    continue
            elif index == 0 and pre_onset >= onset: # onset出现在两个窗口中，新的窗口不需要处理这种情况；onset出现在前面窗口也不需要处理这种情况
                continue
            else: # onset 出现在上述两种情况之外
                if onset > self.onset_bound:
                    counts_onset += 1
            if onset > self.onset_bound:
                onset_time = i * self.onset_capture_window + index / 100
                # print(f"{counts_onset}_{onset}_{onset_time}")
                index_onsettime.append([counts_onset, onset_time])
                onset_x.append(onset_time*100)
                onset_y.append(onset)
        if plot:
            CNN_onset = madmom.features.onsets.CNNOnsetProcessor()
            total_onset = CNN_onset(audio)
            plt.plot(total_onset, 'g-', linewidth=1)
            plt.plot(onset_x, onset_y, 'r.')
            plt.show()

        return index_onsettime

    def MCQS(self, y, start_note=21, end_note=108):
        overtone_scale = self.concurrence * 4
        n_bins = end_note - start_note
        Cqt = np.abs(librosa.cqt(y, sr=self.sr, bins_per_octave=12, window='hamm',
                                 fmin=librosa.midi_to_hz(start_note), n_bins=n_bins))
        amplide2dB = librosa.amplitude_to_db(Cqt, ref=np.max)
        frame_pitch = amplide2dB.transpose()
        MCQS = np.zeros_like(frame_pitch)
        MCQS += frame_pitch.min()
        for frame in range(len(MCQS)):
            max_component_Index = argrelextrema(frame_pitch[frame], np.greater)
            index = np.argsort(frame_pitch[frame][max_component_Index])
            if len(index) > overtone_scale:
                restricted_index = index[-overtone_scale:]
            else:
                restricted_index = index
            # for i in max_component_Index[0][restricted_index]:
            #     MCQS[frame][i] = frame_pitch[frame][i]
            MCQS[frame][max_component_Index[0][restricted_index]] = frame_pitch[frame][max_component_Index[0][restricted_index]]
        print(f"pruned: {MCQS.shape}") # 4915,87
        # print(MCQS[20])
        # MCQS = MCQS.T
        return MCQS

    def get_SDV(self, index_onset, onset_number=30 ):
        MCQS = self.MCQS(self.audio)
        bias = int(MCQS.shape[0]/3)
        index = int(index_onset*MCQS.shape[0]/onset_number)
        if index_onset != -1:
            vector_before_onset = MCQS[index:index+bias]
            vector_after_onset = MCQS[index+bias:index+2*bias]

            mean_before=np.mean(vector_before_onset, axis=0)
            mean_after = np.mean(vector_after_onset, axis=0)

            dif = mean_after-mean_before
            print(dif)
            d_max = np.max(dif)
            sdv = np.zeros_like(dif)
            if self.sdv_type == 1:
                for i in range(len(dif)):
                    if dif[i] > d_max/20:
                        sdv[i] = dif[i]
            if self.sdv_type == 2:
                for i in range(len(dif)):
                    if dif[i] > d_max/20:
                        sdv[i] = 1
            if self.sdv_type == 3:
                for i in range(len(dif)):
                    if dif[i] > d_max/20:
                        sdv[i] = log10(dif[i])
            return sdv

    def Calculate_Onset_Recall(self):
        midi = self.midi_path
        with open(midi, 'r') as f:
            midi_onset = f.readlines()
            ground_truth_onset = []
            for i in range(1, len(midi_onset)):
                ground_truth_onset.append([i, float(midi_onset[i].split('\t')[0])])
        predicted_onset = self.Realtime_Capture_Onset()
        # print(ground_truth_onset)
        # print(predicted_onset)
        recall_score = 0
        for i in range(len(ground_truth_onset)):
            for j in range(len(predicted_onset)):
                if abs(ground_truth_onset[i][1] - predicted_onset[j][1]) < 0.05: # tolrance window
                    recall_score += 1
                    # print([ground_truth_onset[i][1], predicted_onset[j][1]])
                    break
                else:
                    continue
        print(f"recall_score: {recall_score}, recall: {recall_score/len(ground_truth_onset)}")

class SPV():
    def __init__(self, **params):
        self.midi_path = params['midi_path']

    def get_spv(self):
        mid = MidiFile(self.midi_path)
        time = 0
        first_onset_flag = 0  # 进行初始化concurrence,spv1,spv2,spv3时需要判断第一个onset出现在0秒时
        same_first_onset_flag = 0 #第二个索引onset的时候需要判断onset出现在0秒
        onset_count = 0 # 可能的onset数目
        onset_index = -1 # onset索引的起始条件
        max_note=0 # 最大的音符
        min_note = 108 # 最小的音符
        max_concurrence =1 # 最多的concurrence
        cur_concurrence =1 # 当前的concurrence
        for msg in mid:
            print(msg)
            if msg.type != 'note_on' and msg.type != 'note_off':
                continue
            if msg.type == "note_on":
                first_onset_flag += 1
                if msg.note>max_note:
                    max_note= msg.note
                if msg.note < min_note:
                    min_note = msg.note
            pre_time = time
            if msg.time != 0:
                time += msg.time
            if cur_concurrence > max_concurrence:
                max_concurrence = cur_concurrence
            if msg.type == "note_on" and first_onset_flag==1 and msg.time==0:
                onset_count += 1
            elif msg.type == "note_on":
                if pre_time != time:
                    onset_count += 1
                    cur_concurrence = 1
                else:
                    cur_concurrence += 1
        note_range = max_note - min_note + 1
        concurrence = np.zeros((onset_count, note_range), dtype=int)
        concurrence_time = np.zeros(onset_count)
        spv1 = np.zeros((onset_count,note_range), dtype=int)
        spv2 = np.zeros((onset_count, note_range), dtype=int)
        spv3 = np.zeros((onset_count, note_range), dtype=int)
        print(concurrence.shape)
        time = 0  # 重新初始化
        for msg in mid:
            if msg.type != 'note_on' and msg.type != 'note_off':
                continue
            pre_time = time
            if msg.time != 0:
                time += msg.time
            if msg.type == 'note_on':
                same_first_onset_flag += 1
                if same_first_onset_flag==1 and msg.time == 0:
                    onset_index += 1
                if pre_time != time:
                    onset_index += 1
                concurrence_time[onset_index] = time
            """
            [0] pitchitself
            [12] secondOctave
            [19] perfetfifth
            [24] third Octave
            """
            print(onset_index)
            print(f"msg.note:{msg.note}")
            print(f"min_note:{min_note}")
            concurrence[onset_index][msg.note-min_note + 0] = 1
            print(concurrence[onset_index])
            for over_tune in [0, 12]:
                over_tune_index = msg.note-min_note+over_tune
                if over_tune_index < note_range:
                    spv1[onset_index][over_tune_index] = 1
            for over_tune in [0, 12, 19]:
                over_tune_index = msg.note-min_note +over_tune
                if over_tune_index < note_range:
                    spv2[onset_index][over_tune_index] = 1
            for over_tune in [0, 12, 19, 24]:
                over_tune_index = msg.note-min_note +over_tune
                if over_tune_index < note_range:
                    spv3[onset_index][over_tune_index] = 1
        return min_note, max_note, max_concurrence, concurrence, spv1, spv2, spv3, concurrence_time


if __name__ == "__main__":
    params = {
        'audio_path': '../piano/MAPS_ISOL_CH0.3_F_AkPnBcht.wav',
        'midi_path': '/Users/wanglei/intern_at_pingan/LivePiano/piano/MAPS_ISOL_CH0.3_F_AkPnBcht.mid'
    }
    spv = SPV(**params)
    min_note, max_note, max_concurrence, concurrence, spv1, spv2, spv3, concurrence_time = spv.get_spv()
    print(f"max_concurrence: {max_concurrence}")
    print(concurrence_time)
    print(f"spv1: {spv1}")
    print(f"spv2: {spv2}")
    print(f"spv3: {spv3}")
    # sdv = SDV(**params)
    # index, onset = sdv.Onset_Detector(sdv.audio)
    # print(index, onset)
    # sdv = sdv.get_SDV(index)
    # print(sdv)
    # sdv.Calculate_Onset_Recall()
    # sdv.Realtime_Capture_Onset()