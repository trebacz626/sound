import math

import numpy as np
from numba import jit
import librosa

DEFAULT_WINDOW_SIZE = 10

@jit(nopython=True, parallel=True)
def lster(sound, window_size=DEFAULT_WINDOW_SIZE, big_window_size=DEFAULT_WINDOW_SIZE*25):
    calculated_ste = efficient_ste(sound, window_size)
    num_windows = int(big_window_size // window_size)
    lster = []
    for i in range(num_windows, len(calculated_ste), num_windows//2):
        chunk = calculated_ste[i-num_windows:i]
        avg = chunk.mean()
        v = np.sign(0.5*avg-chunk) + 1
        lster.append(v.sum() * 0.5 / num_windows)

    return np.array(lster)


@jit(nopython=True, parallel=True)
def efficient_ste(sound, window_size=DEFAULT_WINDOW_SIZE):
    l = len(sound)
    distance = window_size//2
    ste = []
    sound_squared = np.power(sound, 2)
    for i in range(window_size, l, distance):
        ste.append(sound_squared[i-window_size:i].mean())
    return np.array(ste)


@jit(nopython=True, parallel=True)
def efficient_volume(sound, window_size=DEFAULT_WINDOW_SIZE):
    return np.sqrt(efficient_ste(sound, window_size))


@jit(nopython=True, parallel=True)
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


@jit(nopython=True, parallel=True)
def efficient_fundamental_frequency(sound, frame_rate,window_size=DEFAULT_WINDOW_SIZE):
    L = len(sound)
    distance = window_size // 2
    freqs = []
    min_freq = 20
    max_freq = 1000
    minperiod = frame_rate//max_freq
    maxperiod = frame_rate//min_freq
    for i in range(window_size, L, distance):
        corrs = []
        step = 1
        for l in range(minperiod, maxperiod, step):
            corrs.append(efficient_rn(sound[i-window_size:i+l], l))
        f0 = frame_rate/(np.argmax(np.array(corrs))*step+minperiod)
        freqs.append(f0)
    return np.array(freqs)


@jit(nopython=True)
def efficient_sample_time(times, window_size=DEFAULT_WINDOW_SIZE):
    l = len(times)
    distance = int(window_size // 2)
    return np.array([times[i-distance] for i in range(window_size, l, distance)])


###CLIP LEVEL###
@jit(nopython=True, parallel=True)
def vdr(sound, window_size=DEFAULT_WINDOW_SIZE):
    vol = efficient_volume(sound, window_size)
    return (vol.max()-vol.min())/vol.max()


@jit(forceobj=True)
def find_detections(zcr, volume, times_window):
    bezdzw_selector = ((zcr > 0.07) & (volume < 6000)).astype(int)
    silence_selector = ((zcr > 0.07) & (volume < 300)).astype(int)
    bezdzw_selector = bezdzw_selector - silence_selector
    if bezdzw_selector[0] == 1 and bezdzw_selector[1] == 0:
        bezdzw_selector[0] = 0
    result =  np.concatenate([
        find_continuous_segments(bezdzw_selector, times_window, 1),
        find_continuous_segments(silence_selector, times_window, 0)
    ], axis=0)
    result = result[result[:, 0].argsort()]
    return result


@jit(forceobj=True)
def find_continuous_segments(selector, times_window, type=0):
    differences = selector[1:] - selector[:-1]
    start_idxes = np.where(differences > 0)[0] + 1
    if selector[0] == 1:
        start_idxes = np.insert(start_idxes, 0, 0, axis=0)
    end_idxes = np.where(differences < 0)[0]
    if selector[-1] == 1:
        end_idxes = np.append(end_idxes, len(selector) - 1)
    if len(start_idxes) == 0 or len(end_idxes) == 0:
        return np.empty([0,3])
    return np.array([[times_window[s],times_window[e], type] for s,e in zip(start_idxes, end_idxes)])

def rectangular_window_discrete(N):
    return np.ones(N)

def trinangular_window_discrete(N):
    return np.array([1 - 2*np.abs(n - (N-1)/2)/(N-1) for n in range(N)])

def hanning_window_discrete(N):
    return np.array([0.5 *(1-np.cos(2*np.pi*n/(N-1))) for n in range(N)])

def hamming_window_discrete(N):
    return np.array([0.54 - 0.46*np.cos(2*np.pi*n/(N-1)) for n in range(N)])

def blackman_window_discrete(N):
    return np.array([0.42 - 0.5*np.cos(2*np.pi*n/(N-1)) + 0.08*np.cos(4*np.pi*n/(N-1)) for n in range(N)])


def get_window(window_type: str):
    if window_type == 'rectangular':
        return rectangular_window
    elif window_type == 'triangular':
        return trinangular_window
    elif window_type == 'hanning':
        return hanning_window
    elif window_type == 'hamming':
        return hamming_window
    elif window_type == 'blackman':
        return blackman_window
    else:
        raise ValueError('Unknown window type')


def rectangular_window(t: np.array, T: float):
    return np.abs(t) < T/2

def trinangular_window(t: np.array, T: float):
    return np.maximum(0, 1 - np.abs(t) / T)

def hanning_window(t: np.array, T: float):
    return (0.5 + 0.5*np.cos(np.pi*t/T))*rectangular_window(t, T)

def hamming_window(t: np.array, T: float):
    return 0.54 + 0.46 * np.cos(np.pi*t/T)*rectangular_window(t, T)

def blackman_window(t: np.array, T: float):
    return 0.42 + 0.5 * np.cos(2*np.pi*t/T) + 0.08 * np.cos(2*np.pi*t/T)*rectangular_window(t, T)

def rectangular_window_frequency_domain(f: np.array, T: float):
    return 2*np.sin(f*T)/f


def trinangular_window_frequency_domain(f: np.array, T: float):
    return T*(np.sin(f*T/2)/(f*T/2))**2


def hanning_window_frequency_domain(f: np.array, T: float):
    return (np.pi**2*np.sin(f*T))/(f*(np.pi**2 - T**2*f**2))


def hamming_window_frequency_domain(f: np.array, T: float):
    return (1.08*np.pi**2-0.16*T**2*f**2)/(f*(np.pi**2 - T**2*f**2))*np.sin(f*T)


def parzen_window_frequency_domain(f: np.array, T: float):
    return 3*T/4*(np.sin(f*T/4)/(f*T/4))**4


def fft(signal, window_size=2048, hop_size=512):
    Y = librosa.stft(signal, n_fft=window_size, hop_length=hop_size)#complex with magnitude and phase
    return Y, np.abs(Y)**2


def our_windowed_fft(singal:np.array, window_type:str, window_size=2048, hop_size=512):
    window = get_window(window_type)(window_size)
    for end in range(window_size-1, len(singal), hop_size):
        start = end - window_size
        yield np.fft.fft(singal[start:end]*window)


def volume_freq(f):
    return 1/f.shape[0]*np.power(f,2).sum()
def volume_freq_all_windows(f_windows):
    result = []
    for f in f_windows:
        result.append(1/f.shape[0]*np.power(f,2).sum())
    return np.array(result)

BANDS = np.array([
    [0,630],
    [630, 1720],
    [1720, 4400],
    [4400, 11025]
])

def spectral_flatness_measure(b:int, f_):
    pass
