import numpy as np
a = np.zeros((5,5))
for i in range(len(a)):
    for j in range(len(a[0])):
        a[i][j] =i
print(a)

b = a.transpose()
print(b)
print(a.shape)