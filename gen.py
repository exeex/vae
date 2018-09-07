import random
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal
import torch
from torch.utils.data import Dataset


def sine(peroid=50, length=98, phase=0.):
    return _sine(length=length, k=length / peroid, phase=phase)


def _sine(length=98, k=20., phase=0.):
    x = np.linspace(0 + phase, 2 * np.pi * k + phase, num=length)
    return np.sin(x)


def gaussian(length=98, std=20):
    return scipy.signal.gaussian(length, std=std)


def exp(length=98, tau=1):
    return np.exp(-np.linspace(0, 2, num=length))


def indentity(length=100, ):
    return np.zeros((98,)) + 1


def pad_and_shift(x, pad=99, shift=50):
    x = np.pad(x, (pad, pad), 'constant', constant_values=(0, 0))

    return np.roll(x, shift, axis=-1)


class DataGen:
    def __init__(self):

        self.periods = [10, 20, 30]
        self.envelopes = [gaussian, exp, indentity]
        self.length = 98

    def gen(self, p_idx, e_idx, shift=None, phase=None, gain=None):

        if shift is None:
            shift = random.randint(-100, 100)
        if phase is None:
            phase = random.uniform(-1, 1)
        if gain is None:
            gain = random.gauss(1, 0.3)

        return pad_and_shift(self.envelopes[e_idx]() * sine(self.periods[p_idx], phase=phase), shift=shift) * gain


def demo():
    g = DataGen()
    plt.plot(g.gen(1, 2))
    plt.show()


class SineData(Dataset):
    def __init__(self):
        super(SineData, self).__init__()
        self.g = DataGen()

    def __len__(self):
        return 10000

    def __getitem__(self, idx):
        data = self.g.gen(random.randint(0, 2), random.randint(0, 2))
        return data.astype('float32')



if __name__ == '__main__':

    g = SineData()
    print(g[0].shape)
