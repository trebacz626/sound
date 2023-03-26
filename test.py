import numpy as np
from scipy.io import wavfile
from scipy.signal import find_peaks
import matplotlib.pyplot as plt

from sound_params import efficient_fundamental_frequency

framerate, sound = wavfile.read('./sounds/me.wav')
fram_size_ms = 1000
frame_size = framerate*fram_size_ms//1000
sound = sound


fundamental = efficient_fundamental_frequency(sound,framerate, frame_size)
print("plotting")
plt.plot(list(range(len(fundamental))), fundamental)
plt.show()



# print("gonna correlate")
# corr = np.correlate(sound, sound)
# print("correlated")
# # Find peaks in autocorrelation
# plt.plot(list(range(len(corr))), corr)
# plt.show()

# peaks, _ = find_peaks(corr)
#
# # Calculate fundamental frequency
# f1 = framerate / peaks[0]
#
# print(f"The fundamental frequency is {f1:.2f} Hz")

