import numpy as np
import cv2
from scipy.signal import convolve
import matplotlib.pyplot as plt

# Parameters
L = 4                   # Upsampling factor
T_sym = 1               # Symbol time
N_sym = 8               # Pulse shaping filter length in symbols
SNR = 10                # Signal-to-noise ratio (dB)
nTraces = 100           # Number of traces for eye diagram
nSamples = 3 * L        # Number of samples per trace

# 1️⃣ **SRRC Pulse Shaping Filter**
def srrc_pulse(T_sym, L, N_sym, beta=0.25):
    """Generate SRRC pulse shaping filter."""
    t = np.arange(-N_sym / 2, N_sym / 2, 1 / L)
    p = np.zeros_like(t)
    
    for i in range(len(t)):
        if t[i] == 0:
            p[i] = (1 - beta) + 4 * beta / np.pi
        elif np.abs(t[i]) == T_sym / (4 * beta):
            p[i] = (beta / np.sqrt(2)) * (
                (1 + 2 / np.pi) * np.sin(np.pi / (4 * beta)) +
                (1 - 2 / np.pi) * np.cos(np.pi / (4 * beta))
            )
        else:
            num = np.sin(np.pi * t[i] * (1 - beta) / T_sym) + \
                  4 * beta * t[i] / T_sym * np.cos(np.pi * t[i] * (1 + beta) / T_sym)
            denom = np.pi * t[i] / T_sym * (1 - (4 * beta * t[i] / T_sym)**2)
            p[i] = num / denom

    return p / np.sqrt(T_sym)

# 2️⃣ **Load and Convert Image to Bits**
# Use a sample image
image = cv2.imread("D:/cameraman.png", cv2.IMREAD_GRAYSCALE)
img_flat = image.flatten()
bits = np.unpackbits(img_flat)

# 3️⃣ **BPSK Modulation**
bpsk_symbols = 2 * bits - 1

# 4️⃣ **Upsample by inserting L-1 zeros**
upsampled = np.zeros(len(bpsk_symbols) * L)
upsampled[::L] = bpsk_symbols

# 5️⃣ **Pulse Shaping**
srrc_filter = srrc_pulse(T_sym, L, N_sym)
tx_signal = convolve(upsampled, srrc_filter, mode='same')

# 6️⃣ **Add Gaussian Noise**
noise_power = np.mean(tx_signal**2) / (10**(SNR / 10))
noise = np.sqrt(noise_power / 2) * (np.random.randn(len(tx_signal)) + 1j * np.random.randn(len(tx_signal)))
rx_signal = tx_signal + noise

# 7️⃣ **Matched Filtering**
matched_output = convolve(rx_signal, srrc_filter, mode='same')

# 8️⃣ **Downsampling**
delay = (N_sym * L - 1) // 2
downsampled = matched_output[delay::L].real

# 9️⃣ **Plot Eye Diagram**
fig, ax = plt.subplots(figsize=(12, 6))

# Generate eye diagram traces
for i in range(nTraces):
    start = np.random.randint(0, len(tx_signal) - nSamples)
    trace = matched_output[start: start + nSamples].real
    ax.plot(trace, color='b', alpha=0.3)

# Display the eye diagram
ax.set_title('Eye Diagram of BPSK Signal')
ax.set_xlabel('Time (samples)')
ax.set_ylabel('Amplitude')
ax.grid(True)
plt.show()
