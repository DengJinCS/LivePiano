import numpy as np
from scipy.signal import argrelextrema
a = np.array([1,2,3,4,5,4,3,2,1,2,3,2,1,2,3,4,5,6,5,4,3,2,1])

maxIndex = argrelextrema(a,np.greater)
RmaxIndex = np.argsort(a[maxIndex])
print("maxIndex:",maxIndex,len(maxIndex[0]))
print("RmaxIndex:",RmaxIndex,len(RmaxIndex))
print(a[maxIndex])
print(a[maxIndex[RmaxIndex]])