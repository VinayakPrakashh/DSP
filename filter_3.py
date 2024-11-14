import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import freqz

# Function to design FIR filter using rectangular window
def design_rectangular(N, wc):
    n = np.arange(-N//2, N//2 + 1)
    print(n)
    h_ideal = np.sinc(wc/np.pi * n)
    rectangular_window = np.ones(N + 1)
    h = h_ideal * rectangular_window
    return h

# Function to design FIR filter using Hamming window
def design_hamming_window(N, wc):
    n = np.arange(-N//2, N//2 + 1)
    print(n)
    h_ideal = np.sinc(wc/np.pi * n)
    hamming_window = 0.54 - 0.46 * np.cos(2 * np.pi * n / N)
    h = h_ideal * hamming_window
    return h

# Design the filter using rectangular window
N = 40
wc = 0.2 * np.pi
fs = 1000  # Sampling frequency in Hz
h_rectangular = design_rectangular(N, wc)

# Calculate the frequency response
w, H = freqz(h_rectangular, worN=8000)

# Convert frequency to Hz
frequencies = w * fs / (2 * np.pi)

# Plot the frequency response
plt.figure(figsize=(14, 6))
plt.plot(frequencies, np.abs(H), label='Rectangular Window')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Magnitude (dB)')
plt.title('Frequency Response of FIR Filter with Rectangular Window')
plt.legend()
plt.grid()
plt.show()