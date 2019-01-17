import numpy as np


data = np.array([[1,2,3,3], [2, 1, 3, 4], [3, 3, 10, 5]])
print(data)

print(data[np.arange(3), np.arange(3)])

print(data < 3)

data[data < 3] = 0
print(data)

print(data[np.arange(2)])