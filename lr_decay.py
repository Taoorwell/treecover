import math
import numpy as np
import matplotlib.pyplot as plt

initial_lr_rate = 0.01
epochs = 100


def lr_exponential_decay(epoch):
    # something happen
    decay_rate = 0.04
    return initial_lr_rate * math.pow(decay_rate, epoch / epochs)


def lr_cosine_decay(epoch):
    cosine_decay = 0.5 * (1 + math.cos(math.pi * epoch / epochs))
    return initial_lr_rate * cosine_decay


x = np.arange(1, epochs+1)
plt.plot(x, list(map(lr_exponential_decay, x)), c='red', label='exponential decay')
plt.plot(x, list(map(lr_cosine_decay, x)), c='green', label='cosine decay')
plt.xlabel('epochs')
plt.ylabel('learning rate')
plt.legend()
plt.show()
