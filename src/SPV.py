from mido import MidiFile
import numpy as np

class SPV():
    def __init__(self, **params):
        self.midi_path = params['midi_path']

    def  get_spv(self):
        mid = MidiFile(self.midi_path)
        time = 0
        onset_count = 0 # 可能的onset数目
        onset_index = 0 # onset索引的起始条件
        max_note=0 # 最大的音符
        min_note = 108 # 最小的音符
        max_concurrence =1 # 最多的concurrence
        for msg in mid:
            # print(msg)
            if msg.type != 'note_on' and msg.type != 'note_off':
                continue
            if msg.type == "note_on":
                onset_count += 1
                if msg.note>max_note:
                    max_note= msg.note
                if msg.note < min_note:
                    min_note = msg.note
        note_range = max_note - min_note + 26
        concurrence_time = np.zeros(onset_count+1)
        # print(concurrence.shape)
        for msg in mid:
            # if msg.type != 'note_on' and msg.type != 'note_off': # 有些歌曲的control_change是有时间的
            #     continue
            if msg.time != 0:
                time += msg.time
            if msg.type == 'note_on':
                onset_index += 1
                concurrence_time[onset_index] = time
        # print(concurrence_time)
        temp = np.unique(concurrence_time[1:])
        concurrence_time = np.zeros(len(temp)+1)
        concurrence_time[1:] = temp

        concurrence = np.zeros((len(temp) + 1, note_range), dtype=int)
        spv1 = np.zeros((len(temp) + 1, note_range), dtype=int)
        spv2 = np.zeros((len(temp) + 1, note_range), dtype=int)
        spv3 = np.zeros((len(temp)+ 1, note_range), dtype=int)
        time = 0
        for msg in mid:
            if msg.time != 0:
                time += msg.time
            if msg.type == "note_on":
                index = np.argwhere(concurrence_time == time)
                # print(index[0][0])  #[[]]
                """
                [0] pitchitself
                [12] secondOctave
                [19] perfetfifth
                [24] third Octave
                """
                concurrence[index[0][0]][msg.note - min_note + 0] = 1
                for over_tune in [0, 12]:
                    over_tune_index = msg.note - min_note + over_tune
                    if over_tune_index < note_range:
                        spv1[index[0][0]][over_tune_index] = 1
                for over_tune in [0, 12, 19]:
                    over_tune_index = msg.note - min_note + over_tune
                    if over_tune_index < note_range:
                        spv2[index[0][0]][over_tune_index] = 1
                for over_tune in [0, 12, 19, 24, 28, 31, 34]:
                    over_tune_index = msg.note - min_note + over_tune
                    if over_tune_index < note_range:
                        spv3[index[0][0]][over_tune_index] = 1
                if concurrence_time[0] == 0:
                    concurrence_time[0] = 0.01
        return min_note, max_note, max_concurrence, concurrence, spv1, spv2, spv3, concurrence_time


        # same_element= concurrence_time[1]
        # count_set = []
        # count = 0
        # for i in range(1, onset_count+1):
        #     if concurrence_time[i] == same_element:
        #         count += 1
        #     else:
        #         count_set.append(count)
        #         same_element = concurrence_time[i]
        #         count = 1
        # if concurrence_time[-1] == concurrence_time[-2]:
        #     count_set[-1] += 1
        # else:
        #     count_set.append(1)
        # print(count_set)
        # for msg in mid:
        #     if
