import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from scipy.signal import convolve
from numba import jit


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



#efficient

@jit(nopython=True)
def efficient_ste(sound, window_size=DEFAULT_WINDOW_SIZE):
    l = len(sound)
    distance = window_size//2
    ste = []
    sound_squared = sound*sound
    for i in range(window_size, l, distance):
        ste.append(sound_squared[i-window_size:i].sum())
    return np.array(ste)

@jit(nopython=True)
def efficient_volume(sound, window_size=DEFAULT_WINDOW_SIZE):
    return np.sqrt(efficient_ste(sound, window_size))


@jit(nopython=True)
def efficient_zcr(sound, window_size=DEFAULT_WINDOW_SIZE):
    l = len(sound)
    distance = window_size // 2
    zcr = []
    for i in range(window_size, l, distance):
        c_sound = np.sign(sound[i-window_size:i])
        zcr.append(np.sum(np.abs(c_sound[1:] - c_sound[:-1])))
    return np.array(zcr)/2/window_size

@jit(nopython=True)
def efficient_sample_time(times, window_size=DEFAULT_WINDOW_SIZE):
    l = len(times)
    distance = int(window_size // 2)
    return np.array([times[i-distance] for i in range(window_size, l, distance)])



###CLIP LEVEL###

@jit(nopython=True)
def vdr(sound, window_size=DEFAULT_WINDOW_SIZE):
    vol = efficient_volume(sound, window_size)

    return (vol.max()-vol.min())/vol.min()
