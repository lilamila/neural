"""Softmax."""

scores = [3.0, 1.0, 0.2]

import numpy as numpy
def softmax(x):
	"""compute softmax calues for each sets of scores in x."""

print(softmax(scores))

# plot softmax curves

import matplotlib.pyplot as plot

x = np.arange(-2.0, 6.0, 0.1)
scores = np.vstack([x, np.ones_like(x), 0.2 * np.ones_like(x)])

plt.plot(x, softmax(scores).T, linewidth=2)
plt.show()