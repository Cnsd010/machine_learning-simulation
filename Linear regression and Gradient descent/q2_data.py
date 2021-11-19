"""
Zhibo Zhang and Chris Maddison

for the use of STA314, 2021 Fall, University of Toronto

2021.09.28
"""

import numpy as np

np.random.seed(0)

w = np.asarray([1, -2, 3])
b = 0.88

X = np.random.normal(-5, 5, size=(100, 3))
y = X @ w + b

print(y)

noise1 = np.random.normal(0, 1, 70)
noise2 = np.random.uniform(10, 20, 30)
noise = np.concatenate((noise1, noise2), 0)

t = y + noise

np.save("hw2_X.npy", X)
np.save("hw2_t.npy", t)
