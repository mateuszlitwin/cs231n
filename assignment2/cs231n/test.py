import numpy as np

x = np.array([[1, 2], [3, 4]])
x_mean = np.mean(x, axis=0)
x_var = np.mean((x - x_mean) ** 2, axis=0)

print(x)
print(x_mean)
print((x - x_mean))
print(x_var)
