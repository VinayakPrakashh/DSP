import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import freqz

# Filter design parameters
cutoff_frequency = 0.2  # Cutoff frequency as a fraction of the maximum frequency (0 to 1)
num_samples = 51        # Number of samples in the filter's impulse response

# Design the low-pass filter using a rectangular window
n = np.arange(num_samples) - (num_samples - 1) / 2  # Centered sample indices
print(n)
h = np.sinc(2 * cutoff_frequency * n)               # Ideal sinc filter
h =h/ np.sum(h)                                      # Normalize filter coefficients

# Frequency response
w, h_response = freqz(h)

# Plotting magnitude response (linear and dB scales)
plt.figure(figsize=(10, 6))

# Linear scale
plt.subplot(2, 1, 1)
plt.plot(w / np.pi, abs(h_response), 'r')
plt.title("Magnitude Response (Linear Scale)")
plt.xlabel("Normalized Frequency (xπ rad/sample)")
plt.ylabel("Magnitude")
plt.grid()

# dB scale
plt.subplot(2, 1, 2)
plt.plot(w / np.pi, 20 * np.log10(abs(h_response)), 'b')
plt.title("Magnitude Response (dB Scale)")
plt.xlabel("Normalized Frequency (xπ rad/sample)")
plt.ylabel("Magnitude (dB)")
plt.grid()

plt.tight_layout()
plt.show()