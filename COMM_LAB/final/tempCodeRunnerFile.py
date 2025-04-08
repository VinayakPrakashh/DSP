'''
Pulse Shaping

Aim: Use pulse shaping with square root raised cosine filter and reconstruct the signal
     using a matched filter

Linto Jomon
Date - 13 March 2025
'''

import numpy as np
import matplotlib.pyplot as plt

print("LINTO JOMON \nROLL NO: 44 \nECE")

# Parameters
T_sym = 1
beta = 0.8  # Rolloff factor
L = 4
#SNR = 20  # Used for image transmission
n_samples = 3 * L
n_traces = 100

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

    # Avoid division by zero
    den = (np.pi * t / T_sym) * (1 - (C ** 2))
    den = np.where(den == 0, 1e-10, den)  # Replace zero with a small number

    num = num_sin + C * num_cos
    return (1 / np.sqrt(T_sym)) * (num / den)

def modulate(bits, M):
    k = int(np.log2(M))
    bit_groups = bits.reshape(-1, k)
    symbols = np.array([int("".join(map(str, b)), 2) for b in bit_groups])
    angles = 2 * np.pi * symbols / M
    return np.cos(angles) + 1j * np.sin(angles)

# Generate white noise based on SNR
def noise(SNR, n):
    N0 = 1 / (10 ** (SNR / 10))  # assuming Es = 1
    real_noise = np.random.normal(0, 1, size=n)
    imag_noise = np.random.normal(0, 1, size=n)
    return np.sqrt(N0 / 2) * (real_noise + 1j * imag_noise)

# Function to demodulate MPSK
def demodulate(received_signal, M):
    angles = np.angle(received_signal)
    decoded_symbols = np.round((angles / (2 * np.pi)) * M) % M
    return decoded_symbols.astype(int)

M=int(input("Enter value for M: "))
bits_per_symbol = int(np.log2(M))
d = np.random.randint(0,2,10000)

d = d[:len(d) - (len(d) % bits_per_symbol)]  # Ensure correct length

u=modulate(d,M)

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
# BER Calculation for SNR range from -5 dB to 10 dB
SNR_values = np.arange(-5, 7, 2)  # Steps of 2 dB
BER_values = []

for SNR in SNR_values:
    # Add noise at specified SNR
    w = noise(SNR, len(s))
    r = s + w  # Received signal

    # Matched filtering (correlation with SRRC)
    v_bar = np.convolve(r, p)

    # Downsampling
    filt_delay = int((len(p) - 1) / 2)
    u_bar = v_bar[2 * filt_delay + 1 : -(2 * filt_delay + 1) : L] / L

    # Decision rule for demodulation
    d_bar = demodulate(u_bar , M)
    decoded_bits = np.array([list(np.binary_repr(s, width=bits_per_symbol)) for s in d_bar]).astype(int).flatten()

    bit_errors = np.sum(decoded_bits[:len(d)] != d)
   # symbol_errors = np.sum(decoded_symbols[:len(transmitted_symbols)] != np.round(np.angle(transmitted_symbols) / (2 * np.pi) * M) % M)


    # Compute BER
    BER = bit_errors / len(d)
    BER_values.append(BER)
    
# Plot BER vs. SNR
print(d_bar[:10])
plt.figure(figsize=(8, 6))
plt.semilogy(SNR_values, BER_values, marker="o", linestyle="-")
plt.title("BER vs. SNR")
plt.xlabel("SNR (dB)")
plt.ylabel("Bit Error Rate (BER)")
plt.grid(True, which="both", linestyle="--")
plt.show()