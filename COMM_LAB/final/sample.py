import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve

# SRRC Pulse Function
def srrc_pulse(Tsym, beta, L, Nsym):
    t = np.arange(-Nsym / 2, Nsym / 2, 1 / L)
    p = np.zeros_like(t)
    for i in range(len(t)):
        if t[i] == 0:
            p[i] = (1 - beta + 4 * beta / np.pi) / np.sqrt(Tsym)
        elif abs(t[i]) == Tsym / (4 * beta):
            p[i] = (beta / np.sqrt(2 * Tsym)) * ((1 + 2 / np.pi) * np.sin(np.pi / (4 * beta)) +
                                                 (1 - 2 / np.pi) * np.cos(np.pi / (4 * beta)))
        else:
            num = (np.sin(np.pi * t[i] * (1 - beta) / Tsym) +
                   4 * beta * t[i] / Tsym * np.cos(np.pi * t[i] * (1 + beta) / Tsym))
            denom = (np.pi * t[i] / Tsym * (1 - (4 * beta * t[i] / Tsym) ** 2))
            p[i] = num / denom
    return p / np.sqrt(np.sum(p ** 2)), t

# Parameters
Tsym = 1
L = 8  # Samples per symbol
beta = 0.35
Nsym = 6

# Input symbols
symbols = np.array([1, 0, -1, 1])

# 1. Upsample
upsampled = np.zeros(len(symbols) * L)
upsampled[::L] = symbols

# 2. SRRC pulse
srrc, t_srrc = srrc_pulse(Tsym, beta, L, Nsym)

# 3. Convolve
tx_signal = convolve(upsampled, srrc, mode='full')

# 4. Plotting

plt.figure(figsize=(15, 6))

# Plot 1: Upsampled signal
plt.subplot(3, 1, 1)
plt.stem(upsampled)
plt.title("1. Upsampled Signal")
plt.xlabel("Sample Index")
plt.ylabel("Amplitude")
plt.grid(True)

# Plot 2: SRRC pulse
plt.subplot(3, 1, 2)
plt.plot(t_srrc, srrc)
plt.title("2. SRRC Pulse")
plt.xlabel("Time")
plt.ylabel("Amplitude")
plt.grid(True)

# Plot 3: Convolution Result (Transmit Signal)
plt.subplot(3, 1, 3)
plt.plot(tx_signal)
plt.title("3. Transmit Signal (Upsampled ⊗ SRRC)")
plt.xlabel("Sample Index")
plt.ylabel("Amplitude")
plt.grid(True)

plt.tight_layout()
plt.show()
