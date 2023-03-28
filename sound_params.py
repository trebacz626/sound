import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from scipy.signal import convolve
from numba import jit


DEFAULT_WINDOW_SIZE = 10


def ste(sound, window_size=DEFAULT_WINDOW_SIZE):
    return (convolve(sound * sound, np.ones(window_size)) / window_size)[(window_size-1):-(window_size-1)]


def volume(sound, window_size=DEFAULT_WINDOW_SIZE):
    return np.sqrt(ste(sound, window_size))


def zcr(sound, window_size=DEFAULT_WINDOW_SIZE):
    slided = sliding_window_view(sound, window_size-1)
    return 1/window_size*0.5*np.sum(np.abs(np.sign(slided[1:]) - np.sign(slided[:-1])), axis=1)


def rn(sound, l, window_size=DEFAULT_WINDOW_SIZE):
    pass


def lster(sound, window_size=DEFAULT_WINDOW_SIZE, big_window_size=DEFAULT_WINDOW_SIZE*25):
    calculated_ste = efficient_ste(sound, window_size)
    num_windows = int(big_window_size // window_size)
    lster = []
    for i in range(num_windows, len(calculated_ste), num_windows//2):
        chunk = np.array(calculated_ste[i-num_windows:i])
        avg = chunk.mean()
        v = np.sign(chunk - 0.5*avg) + 1
        lster.append(v.sum() * 0.5 / num_windows)

    return np.array(lster)


# efficient

@jit(nopython=True)
def efficient_ste(sound, window_size=DEFAULT_WINDOW_SIZE):
    l = len(sound)
    distance = window_size//2
    ste = []
    sound_squared = np.power(sound, 2)
    for i in range(window_size, l, distance):
        ste.append(sound_squared[i-window_size:i].mean())
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
def efficient_rn(sound, l=10):
    if len(sound) <= l:
        return 0
    N = len(sound)
    total = 0
    for i in range(N-l):
        total += (sound[i]*sound[i+l])
    return total/(len(sound)-l)

@jit(nopython=True, parallel = True)
def efficient_amdf(sound, l=10):
    N = len(sound)
    total = 0
    for i in range(N-l):
        total += abs(sound[i]-sound[i+l])
    return total/(len(sound)-l)

# @jit(nopython=True, parallel = True)
def efficient_fundamental_frequency(sound, frame_rate,window_size=DEFAULT_WINDOW_SIZE):
    L = len(sound)
    distance = window_size // 2
    freqs = []
    min_freq=20
    max_freq=1000
    minperiod=frame_rate//max_freq
    maxperiod=frame_rate//min_freq
    for i in range(window_size, L, distance):
        corrs=[]
        step = 1
        for l in range(minperiod,maxperiod,step):
            corrs.append(efficient_rn(sound[i-window_size:i+l],l))
        f0 = frame_rate/(np.argmax(np.array(corrs))*step+minperiod)
        freqs.append(f0)
    return np.array(freqs)




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


def find_silences(zcr, volume, times_window):
    #chrzaszcz volume 4000
    silence_selector = ((zcr > 0.07) & (volume < 5000)).astype(int)
    # silence_selector = [silence_selector[0]] + (
    #             (silence_selector[0:-2] & silence_selector[2]) | silence_selector[1:-1]).tolist() + [
    #                        silence_selector[-1]]
    differences = silence_selector[1:] - silence_selector[:-1]
    start_idxes = np.where(differences > 0)[0] + 1
    if silence_selector[0] == 1:
        start_idxes = np.insert(start_idxes, 0, 0, axis=0)
    end_idxes = np.where(differences < 0)[0]
    if silence_selector[-1] == 1:
        end_idxes = np.append(end_idxes, len(silence_selector) - 1)
    print(np.array([[times_window[s],times_window[e]] for s,e in zip(start_idxes, end_idxes)]))
    return np.array([[times_window[s],times_window[e]] for s,e in zip(start_idxes, end_idxes)])
