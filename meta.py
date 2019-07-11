import pyaudio

"""
meta for mic
"""
RATE = 16000 #sample rate
#RATE = 16000
FORMAT = pyaudio.paInt16 #conversion format for PyAudio stream
CHANNELS = 1 #microphone audio channels
CHUNK_SIZE = 2028 #number of samples to take per read
SAMPLE_LENGTH = int(CHUNK_SIZE*1000/RATE) #length of each sample in ms


"""
meta for spec
"""
SAMPLES_PER_FRAME = 50#Number of mic reads concatenated within a single window
nfft = 1024#NFFT value for spectrogram
overlap = 512#overlap value for spectrogram
rate = RATE #sampling rate
chunk = CHUNK_SIZE
show_rate = 1000000#显示频谱的规整

"""
meta for pitch
"""
PianoA0 = RATE / nfft * 2#hz