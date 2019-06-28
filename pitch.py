import numpy as np
from meta import RATE,nfft,overlap,PianoA0

sr = RATE
win = nfft/RATE
hop = overlap/RATE


class Librosa_Pitch(object):
    def __init__(self, inputSignal, fs=sr, windowSize=win, hopTime=hop, minFreq=PianoA0/2):

        self.fs = fs
        self.maxLag = int(np.ceil(fs / minFreq) + 1)
        self.windowSize = round(fs * windowSize)
        self.hopSize = round(fs * hopTime)
        self.frameCount = int(np.floor((len(inputSignal) - self.windowSize) / self.hopSize))
        self.inputSignal = np.append(inputSignal, np.zeros(self.windowSize - int(len(inputSignal) / self.hopSize)))
        #self.timeStamp = np.linspace(0, (self.frameCount - 1) * self.hopSize / self.fs, self.frameCount)
        self.ptr = 0


class YIN_Pitch(object):
    """
    Pitch Estimation Algorithm that finds the pitches (in Hz) of an audio file. So far, there is only one type
    of pitch estimation algorithm, which is the YIN method that works well with monophonic signals.
    Attributes:
    inputSignal: The audio file
    fs: Sampling rate (default = 44100 Hz)
    windowSize: Length of window size in seconds (default = 80ms)
    hopTime: Length of overlapping window in seconds (default = 10ms)
    threshold: The threshold for voice or unvoiced decision (default = 0.2)
    minFreq: The minimum frequency for the search range (default = 65 Hz, low C note)
    timeStamp: An array of the time stamps for each frame
    """

    def __init__(self, inputSignal, fs=sr, windowSize=win, hopTime=hop, minFreq=PianoA0/2, threshold=0.3):

        self.fs = fs
        self.maxLag = int(np.ceil(fs / minFreq) + 1)
        self.threshold = threshold
        self.windowSize = round(fs * windowSize)
        self.hopSize = round(fs * hopTime)
        self.frameCount = int(np.floor((len(inputSignal) - self.windowSize) / self.hopSize))
        self.inputSignal = np.append(inputSignal, np.zeros(self.windowSize - int(len(inputSignal) / self.hopSize)))
        #self.timeStamp = np.linspace(0, (self.frameCount - 1) * self.hopSize / self.fs, self.frameCount)
        self.ptr = 0

    def getYIN(self):

        d = np.zeros(self.maxLag - 1)

        #Step（1）2：The Difference function
        #autocorrelation function（ACF）运算可以用于寻找周期信号的周期。因为周期信号的自相关函数也是周期信号，而且周期一致。
        for j in range(len(d)):
            d[j] = np.sum(pow(self.inputSignal[self.ptr: self.ptr + self.windowSize] - self.inputSignal[
                                                                                       self.ptr + j + 1: self.ptr + self.windowSize + j + 1],
                              2))

        #Step 3: Cumulative mean normalized difference function
        #相比于step2，减少了τ比较小的时候出现零点的情况，减弱了小周期的产生。
        yin = d / np.cumsum(d) * np.arange(1, self.maxLag)

        return yin

    def getPitchOfFrame(self):

        yin = self.getYIN()

        #Step 4: Absolute threshold
        """
        step3，如果极小值出现的时间点比周期对应的时间点要长，会出现周期变长的情况。
        为了解决这类问题，可以设置一个绝对的门限（比如0.1），找到低于这个门限的每一段的极小值，
        这些极小值中的第一个极小值对应ττ作为周期（第三步是使用这些极小值的中的最小值对应的ττ作为周期）。
        如果没有低于门限的点，那么选择全局最小值作为周期。
        
        Step 5: Parabolic interpolation
        前面的步骤有一个假设，即信号的周期是采样周期的倍数。如果不满足的话，会引起错误。 
        为了解决这个问题，每个d′(τ)d′(τ)的局部最小值以及相邻的点，可以使用二次函数抛物线进行拟合。后续周期计算的时候使用每个局部最小值对应的拟合值。找到的最小值对应的时间即为周期。使用这种方法有可能存在bias，可以用d(τ)d(τ)来避免这个问题。

        Step 6: Best local estimate
        在找到的周期值附近，在时间窗口[t−Tmax/2,t+Tmax/2][t−Tmax/2,t+Tmax/2]寻找极小值对应的时间作为周期，TmaxTmax是最大期望周期值。 
        总体上，YIN算法使用d′(τ)d′(τ)进行周期估计，同时设置一个绝对门限，在找得到周期值附近重新进行搜索获得最优值。
        """
        idxBelowThresh = np.where(yin < self.threshold)[0]
        pitch = 0

        if (len(idxBelowThresh) != 0):
            #print("index len:", len(idxBelowThresh))
            stopAt = np.where(np.diff(idxBelowThresh) > 1)[0]
            #print("stop at:",len(stopAt),stopAt)
            if (len(stopAt) == 0):
                idxMin = np.argmin(yin[idxBelowThresh])
            else:
                searchRange = stopAt
                idxMin = np.argmin(yin[searchRange])

            idx = idxBelowThresh[idxMin]

            num = yin[idx - 1] - yin[idx + 1]
            den = yin[idx - 1] - 2 * yin[idx] + yin[idx + 1]

            if (den != 0):
                pitch = self.fs / (idx + num / den / 2)

        return pitch

    def process(self):
        # Process each frame of the audio signal
        pitches = np.zeros(self.frameCount)
        for i in range(self.frameCount):
            self.ptr = i * self.hopSize  # points to the first index of current frame
            pitches[i] = self.getPitchOfFrame()
        return pitches

    def getPitches(self):

        pitches = self.process()

        return pitches