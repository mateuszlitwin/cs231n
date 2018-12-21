import numpy as np

# print(np.exp(-1000))
# print(np.exp(-100))
# print(np.exp(-10))
# print(np.exp(-1))
# print(np.exp(1))
# print(np.exp(10))
# print(np.exp(100))
# print(np.exp(1000))
#
#
# f = np.array([123, 456, 789])  # example with 3 classes and each having large scores
# p = np.exp(f) / np.sum(np.exp(f))  # Bad: Numeric problem, potential blowup
# print(p)
#
# # instead: first shift the values of f so that the highest number is 0:
# f -= np.max(f)  # f becomes [-666, -333, 0]
# p = np.exp(f) / np.sum(np.exp(f))  # safe to do, gives the correct answer
# print(p)


print(np.array([-1, 0, 1, 2, 3]))
print(np.array([-1, 0, 1, 2, 3]) == 3)
print(np.flatnonzero(
    np.array([-1, 0, 1, 2, 3])
))


import matplotlib.pyplot as plt

plt.subplot()


a = np.reshape(np.arange(30).reshape(3, 2, 5), (15, 2))
print(a)

np.hstack()