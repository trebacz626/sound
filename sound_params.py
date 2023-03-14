import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from scipy.signal import convolve

DEFAULT_WINDOW_SIZE = 10


def ste(sound, window_size=DEFAULT_WINDOW_SIZE):
    return (convolve(sound * sound, np.ones(window_size)) / window_size)[(window_size-1):-(window_size-1)]


def volume(sound, window_size=DEFAULT_WINDOW_SIZE):
    # print(ste(sound,window_size))
    return np.sqrt(ste(sound,window_size))


def zcr(sound, window_size=DEFAULT_WINDOW_SIZE):
    slided = sliding_window_view(sound, window_size-1)
    return 1/window_size*0.5*np.sum(np.abs(np.sign(slided[1:]) - np.sign(slided[:-1])), axis=1)

def rn(sound, l, window_size=DEFAULT_WINDOW_SIZE):
    pass
