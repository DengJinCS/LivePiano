import src.SDV as sdvs
import src.SPV as spvs
import numpy as np

def main(**params):
    min_note, max_note, max_concurrence, concurrence, spv1,spv2,spv3, concurrence_time = spvs.SPV(**params)
    onset_true_time = np.zeros(concurrence.shape[0] * 2)
    sdv = sdvs.SDV(**params)
    audio = sdv.audio
    block_length = 0.1
    buffer_block = np.zeros(int(3*block_length*sdv.sr))
    number_block = (len(audio)/sdv.sr)//block_length
    onset_count = -1  # 外部的onset计数器

    for i in range(number_block-2):
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
            onset_count += 1
            if onset_count == 0:
                onset_true_time[0] = concurrence_time[0]
            else:
                onset_true_time[onset_count] =
        SDVs = sdv.get_SDV(buffer_block, index_onset, min_note, max_note)



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
if __name__ ==" __main__ ":
    params = {
        'audio_path': '../piano/MAPS_ISOL_CH0.3_F_AkPnBcht.wav',
        'midi_path': '/Users/wanglei/intern_at_pingan/LivePiano/piano/MAPS_ISOL_CH0.3_F_AkPnBcht.mid'
    }






