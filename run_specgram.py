"""
run_specgram.py
Created By DENGJIN (dengjin1995@gmail.com)

Main Script for the Live Spectrogram project, a real time spectrogram
visualization tool

Dependencies: matplotlib, numpy and the mic_read.py module
"""
############### Import Libraries ###############
from matplotlib.mlab import window_hanning,specgram
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.colors import LogNorm
import numpy as np
import librosa.display
from pitch import *

############### Import Modules ###############
import mic_read

############### Constants ###############

SAMPLES_PER_FRAME = 100#Number of mic reads concatenated within a single window
nfft = 2048#NFFT value for spectrogram
overlap = 1024#overlap value for spectrogram
rate = mic_read.RATE #sampling rate
chunk = mic_read.CHUNK_SIZE

############### Functions ###############
"""
get_sample:
gets the audio data from the microphone
inputs: audio stream and PyAudio object
outputs: int16 array
"""
def get_sample(stream,pa):
    data = mic_read.get_data(stream,pa)
    return data
"""
get_specgram:
takes the FFT to create a spectrogram of the given audio signal
input: audio signal, sampling rate
output: 2D Spectrogram Array, Frequency Array, Bin Array
see matplotlib.mlab.specgram documentation for help
"""
def get_specgram(signal,rate):
    arr2D,freqs,bins = specgram(signal,window=window_hanning,
                                Fs = rate,NFFT=nfft,noverlap=overlap)
    return arr2D,freqs,bins

"""
update_fig:
updates the image, just adds on samples at the start until the maximum size is
reached, at which point it 'scrolls' horizontally by determining how much of the
data needs to stay, shifting it left, and appending the new data. 
inputs: iteration number
outputs: updated image
"""


def main():
    ############### Initialize Plot ###############
    fig = plt.figure()
    """
    Launch the stream and the original spectrogram
    """
    stream,pa = mic_read.open_mic()
    data = get_sample(stream,pa)

    arr2D,freqs,bins = get_specgram(data,rate)
    """
    Setup the plot paramters
    """


    """
    自定义映射颜色空间
    """
    import matplotlib.cm as cm
    import matplotlib.colors as col
    startcolor = '#000000'  # 黑色，读者可以自行修改
    mid = '#0277bd'
    endcolor = '#4FC3F7'  # 蓝色，读者可以自行修改
    cmap = col.LinearSegmentedColormap.from_list('J', [startcolor, mid,endcolor])
    # extra arguments are N=256, gamma=1.0
    cm.register_cmap(cmap=cmap)



    extent = (bins[0],bins[-1]*SAMPLES_PER_FRAME,freqs[-1],freqs[0])#坐标
    im = plt.imshow(arr2D,aspect='auto',extent = extent,interpolation="hanning",
                    cmap='J',norm=LogNorm(vmin=arr2D.min()*100000))
    #cmap = 'jet'
    #interpolation="quadric",
    #interpolation = "gaussian",
    plt.yscale("linear")

    plt.xlabel('Time (s)')
    plt.ylabel('Frequency (Hz)')
    plt.title('Real Time Spectogram')
    plt.gca().invert_yaxis()
    ##plt.colorbar() #enable if you want to display a color bar
    def update_fig(n):
        data = get_sample(stream, pa)
        arr2D, freqs, bins = get_specgram(data, rate)

        im_data = im.get_array()
        if n < SAMPLES_PER_FRAME:
            im_data = np.hstack((im_data, arr2D))
            im.set_array(im_data)
        else:
            keep_block = arr2D.shape[1] * (SAMPLES_PER_FRAME - 1)
            im_data = np.delete(im_data, np.s_[:-keep_block], 1)
            im_data = np.hstack((im_data, arr2D))
            im.set_array(im_data)
        return im,


    ############### Animate ###############
    anim = animation.FuncAnimation(fig,update_fig,blit = True,
                                interval=mic_read.CHUNK_SIZE/1000)

                                
    try:
        plt.show()
    except:
        print("Plot Closed")

    ############### Terminate ###############
    stream.stop_stream()
    stream.close()
    pa.terminate()
    print("Program Terminated")

if __name__ == "__main__":
    main()
