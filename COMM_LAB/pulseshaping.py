import numpy as np
import cv2
from scipy.signal import convolve
import matplotlib.pyplot as plt

# Parameters
L = 4  # Upsampling factor
T_sym = 1
N_sym = 8  # Pulse shaping filter length in symbols
SNR = 10  # Signal-to-noise ratio

# Function: SRRC pulse shaping filter
def srrc_pulse(T_sym, L, N_sym, beta=0.25):
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
            denom = np.pi * t[i] / T_sym * (1 - (4 * beta * t[i] / T_sym) ** 2)
            p[i] = num / denom
    
    return p / np.sqrt(T_sym)

# Step 1: Load and convert image
image = cv2.imread("D:/cameraman.png", cv2.IMREAD_GRAYSCALE)

img_flat = image.flatten()
bits = np.unpackbits(img_flat)
# Step 2: BPSK modulation
bpsk_symbols = np.array([-1 if bit == 0 else 1 for bit in bits])


# Step 3: Upsample by inserting L-1 zeros
upsampled = np.zeros(len(bpsk_symbols) * L)
upsampled[::L] = bpsk_symbols

# Step 4: Pulse shaping
srrc_filter = srrc_pulse(T_sym, L, N_sym)
tx_signal = convolve(upsampled, srrc_filter, mode='same')

print(tx_signal)
# Step 5: Add Gaussian noise
noise_power = np.mean(tx_signal**2) / (10**(SNR / 10))
noise = np.sqrt(noise_power / 2) * (np.random.randn(len(tx_signal)) + 1j * np.random.randn(len(tx_signal)))
rx_signal = tx_signal + noise

# Step 6: Matched filtering
matched_output = convolve(rx_signal, srrc_filter, mode='same')

# Step 7: Downsampling
delay = (N_sym * L - 1) // 2
downsampled = matched_output[delay::L].real

# Step 8: Demapping
recovered_bits = (downsampled > 0).astype(int)
# Step 9: Reconstruct the image
recovered_pixels = np.packbits(recovered_bits)[:len(img_flat)]
reconstructed_image = recovered_pixels.reshape(image.shape)

# Display images
fig, axes = plt.subplots(1, 2, figsize=(12, 6))
axes[0].imshow(image, cmap='gray')
axes[0].set_title("Original Image")
axes[1].imshow(reconstructed_image, cmap='gray')
axes[1].set_title("Reconstructed Image")
plt.show()
