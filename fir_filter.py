
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import freqz

fs = 1000  # Sampling frequency
fc = 100   # Cutoff frequency
N = 20    # Order of filter

n = np.arange(N+1)
h_ideal = np.sinc(2*fc/fs*(n-N/2))
rectangular_window = np.ones(N+1)
hamming_window = 0.54 - 0.46*np.cos(2*np.pi*n/N)

h_rectangular = h_ideal * rectangular_window
h_hamming = h_ideal * hamming_window

w, H_rectangular = freqz(h_rectangular, worN=8000)
w, H_hamming = freqz(h_hamming, worN=8000)

frequencies = w * fs / (2 * np.pi)

plt.figure(figsize=(14, 6))

plt.subplot(1, 2, 1)

plt.plot(frequencies, 20*np.log10(np.abs(H_rectangular)),
         label='Rectangular Window')
plt.plot(frequencies, 20*np.log10(np.abs(H_hamming)),
         label='Hamming Window')

plt.xlabel('Frequency (Hz)')
plt.ylabel('Magnitude (dB)')
plt.title('Frequency Response of Rectangular and Hamming Windows')
plt.legend()

plt.subplot(1, 2, 2)

plt.stem(n, h_rectangular, label='Rectangular Window')
plt.stem(n, h_hamming, label='Hamming Window', markerfmt='o')

plt.xlabel('Sample Index')
plt.ylabel('Amplitude')
plt.title('Impulse Response of Rectangular and Hamming Windows')
plt.legend()

plt.tight_layout()
plt.show()
plt.title('Frequency Response of FIR Low-Pass Filter')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Magnitude (dB)')
plt.grid(True)
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(frequencies, np.abs(H_rectangular), label='Rectangular Window')
plt.plot(frequencies, np.abs(H_hamming), label='Hamming Window')

plt.title('Linear Magnitude Frequency Response')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Magnitude')
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()