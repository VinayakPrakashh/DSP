import numpy as np
import matplotlib.pyplot as plt
import cv2
from google.colab import files

print("arya jestin \n Roll No: 70  \n ECE")

# Upload file
uploaded = files.upload()
file_name = list(uploaded.keys())[0]  # Get the uploaded file name

# Parameters
T_sym = 1
beta = 0.8  # Rolloff factor
L = 4
SNR_values = np.arange(0, 11, 2)  # SNR range for BER graph
BER = []  # To store Bit Error Rate values

# Square Root Raised Cosine (SRRC) Filter
def SRRC(t, T_sym, beta):
    if t == 0:
        return (1 / np.sqrt(T_sym)) * (1 - beta + (4 * beta / np.pi))

    if np.abs(t) == T_sym / (4 * beta):
        term_1 = (1 + 2 / np.pi) * np.sin(np.pi / (4 * beta))
        term_2 = (1 - 2 / np.pi) * np.cos(np.pi / (4 * beta))
        return (beta / np.sqrt(2 * T_sym)) * (term_1 + term_2)

    C = 4 * beta * t / T_sym
    num_sin = np.sin(np.pi * t * (1 - beta) / T_sym)
    num_cos = np.cos(np.pi * t * (1 + beta) / T_sym)

    den = (np.pi * t / T_sym) * (1 - (C ** 2))
    den = np.where(den == 0, 1e-10, den)  # Avoid division by zero

    num = num_sin + C * num_cos
    return (1 / np.sqrt(T_sym)) * (num / den)

# Convert image to bit array
def im_to_arr(file_name):
    arr = cv2.imread(file_name, cv2.IMREAD_GRAYSCALE)
    shape = arr.shape
    arr = arr.flatten()
    arr = np.array([[pixel >> i & 1 for i in range(7, -1, -1)] for pixel in arr])
    return arr.flatten(), shape

# Convert bit array back to image
def arr_to_im(file_name, arr, shape):
    arr = [arr[i: i + 8] for i in range(0, len(arr), 8)]
    pixel_arr = []
    for pixel_list in arr:
        pixel = 0
        for bit in pixel_list:
            pixel = (pixel << 1) | bit
        pixel_arr.append(pixel)
    arr = np.array(pixel_arr)
    cv2.imwrite(file_name, arr.reshape(shape))

# Process Image
d, shape = im_to_arr(file_name)
print(f"Image '{file_name}' loaded successfully!")

# BPSK Modulation (0 → -1, 1 → 1)
u = np.array([1 if i else -1 for i in d])

# Upsampling
v = []
for symbol in u:
    v.append([symbol] + [0] * (L - 1))
v = np.array(v).flatten()

# Generate SRRC pulse
time_step = T_sym / L
t = np.arange(-8 / 2, 8 / 2, 1 / L)
p = np.array([SRRC(t_, T_sym, beta) for t_ in t])

# Plot SRRC impulse response
plt.plot(t, p)
plt.title("Impulse Response of SRRC")
plt.xlabel("Time")
plt.ylabel("Amplitude")
plt.grid(True)
plt.show()

# Transmit signal using convolution
s = np.convolve(v, p)

# Function to add noise
def noise(SNR, n):
    N0 = 1 / (10 ** (SNR / 10))  # Assuming Es = 1
    real_noise = np.random.normal(0, 1, size=n)
    imag_noise = np.random.normal(0, 1, size=n)
    return np.sqrt(N0 / 2) * (real_noise + 1j * imag_noise)

# BER Calculation
for SNR in SNR_values:
    w = noise(SNR, len(s))
    r = s + w  # Received signalg

    # Matched filtering (correlation with SRRC)
    v_bar = np.convolve(r, p)

    # Downsampling
    filt_delay = int((len(p) - 1) / 2)
    u_bar = v_bar[2 * filt_delay + 1 : -(2 * filt_delay + 1) : L] / L

    # Decision rule for demodulation
    d_bar = np.array([1 if i > 0 else 0 for i in u_bar])

    # Compute BER
    errors = np.sum(d_bar != d[:len(d_bar)])
    BER.append(errors / len(d_bar))

# Plot BER vs. SNR graph
plt.figure()
plt.semilogy(SNR_values, BER, marker='o', linestyle='-', color='b')
plt.title("Bit Error Rate (BER) vs. SNR")
plt.xlabel("SNR (dB)")
plt.ylabel("BER (log scale)")
plt.grid(True)
plt.show()

# Reconstruct image
arr_to_im("reconstructed_image.png", d_bar, shape)
print("Reconstructed image saved as 'reconstructed_image.png'")

# Display original and reconstructed images
original_img = cv2.imread(file_name, cv2.IMREAD_GRAYSCALE)
reconstructed_img = cv2.imread("reconstructed_image.png", cv2.IMREAD_GRAYSCALE)

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(original_img, cmap="gray")
plt.title("Original Image")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(reconstructed_img, cmap="gray")
plt.title("Reconstructed Image")
plt.axis("off")
plt.show()