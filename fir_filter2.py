import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import freqz

# Function to design FIR filter
def design_rectangular(N, wc):
    n = np.arange(-N//2, N//2 + 1)
    h_ideal = np.sinc(wc/np.pi * n)
    rectangular_window = np.ones(N + 1)
    h = h_ideal * rectangular_window
    return h
def design_hamming_window(N,wc):
    n = np.arange(-N//2, N//2 + 1)
    h_ideal = np.sinc(wc/np.pi * n)
    hamming_window = 0.54 - 0.46*np.cos(2*np.pi*n/N)
    h = h_ideal * hamming_window
    return h
# Input parameters
N = 40
wc = 0.2 * np.pi

# Design the filter
# h_rectangular = design_rectangular(N, wc)
h_hamming = design_hamming_window(N,wc)

# Function to plot impulse and frequency response
def plot_responses(h, fs):
    w, H = freqz(h, worN=8000)
    frequencies = w * fs / (2 * np.pi)

    plt.figure(figsize=(14, 6))

    plt.subplot(1, 2, 1)
    plt.plot(frequencies, 20 * np.log10(np.abs(H)), label='Rectangular Window')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Magnitude (dB)')
    plt.title('Frequency Response of Rectangular Window')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.stem(np.arange(len(h)), h, label='Rectangular Window')
    plt.xlabel('Sample Index')
    plt.ylabel('Amplitude')
    plt.title('Impulse Response of Rectangular Window')
    plt.legend()

    plt.tight_layout()
    plt.show()

# Plot the responses
fs = 2 * np.pi  # Normalized frequency
# plot_responses(h_rectangular, fs)
plot_responses(h_hamming, fs)