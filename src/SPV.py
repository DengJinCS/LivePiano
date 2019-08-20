from mido import MidiFile
import numpy as np

class SPV():
    def __init__(self, **params):
        print(1)
        self.midi_path = params['midi_path']

    def get_spv(self):
        mid = MidiFile(self.midi_path)
        time = 0
        first_onset_flag = 0  # 进行初始化concurrence,spv1,spv2,spv3时需要判断第一个onset出现在0秒时
        same_first_onset_flag = 0 #第二个索引onset的时候需要判断onset出现在0秒
        onset_count = 0 # 可能的onset数目
        onset_index = 0 # onset索引的起始条件
        max_note=0 # 最大的音符
        min_note = 108 # 最小的音符
        max_concurrence =1 # 最多的concurrence
        cur_concurrence =1 # 当前的concurrence
        for msg in mid:
            # print(msg)
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
        note_range = max_note - min_note + 26
        concurrence = np.zeros((onset_count+1, note_range), dtype=int)
        concurrence_time = np.zeros(onset_count+1)
        spv1 = np.zeros((onset_count+1,note_range), dtype=int)
        spv2 = np.zeros((onset_count+1, note_range), dtype=int)
        spv3 = np.zeros((onset_count+1, note_range), dtype=int)
        # print(concurrence.shape)
        time = 0  # 重新初始化
        for msg in mid:
            if msg.type != 'note_on' and msg.type != 'note_off':
                continue
            pre_time = time
            if msg.time != 0:
                time += msg.time
            if msg.type == 'note_on':
                same_first_onset_flag += 1
                if same_first_onset_flag==1 and msg.time == 0: #第一个onset处有多个音符
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
            # print(onset_index)
            # print(f"msg.note:{msg.note}")
            # print(f"min_note:{min_note}")
            concurrence[onset_index][msg.note-min_note + 0] = 1
            for over_tune in [0, 12]:
                over_tune_index = msg.note-min_note+over_tune
                if over_tune_index < note_range:
                    spv1[onset_index][over_tune_index] = 1
            for over_tune in [0, 12, 19]:
                over_tune_index = msg.note-min_note +over_tune
                if over_tune_index < note_range:
                    spv2[onset_index][over_tune_index] = 1
            for over_tune in [0, 12, 19, 24, 28, 31, 34]:
                over_tune_index = msg.note-min_note +over_tune
                if over_tune_index < note_range:
                    spv3[onset_index][over_tune_index] = 1
        return min_note, max_note, max_concurrence, concurrence, spv1, spv2, spv3, concurrence_time

# params = {
#         'audio_path': '../piano/MAPS_ISOL_CH0.3_F_AkPnBcht.wav',
#         'midi_path': '/Users/wanglei/intern_at_pingan/LivePiano/piano/MAPS_ISOL_CH0.3_F_AkPnBcht.mid'
#     }
#
# spv = SPV(**params)
# min_note, max_note, max_concurrence, concurrence, spv1, spv2, spv3, concurrence_time = spv.get_spv()
# print(f"max_concurrence: {max_concurrence}")
# print(concurrence_time)
# print(f"spv1: {spv1}")
# print(f"spv2: {spv2}")
# print(f"spv3: {spv3}")