import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve

def srrc_pulse(beta, Nsym, Tsym, L):
    t = np.arange(-Nsym / 2, Nsym / 2, 1 / L)
    p = np.zeros_like(t)
    for i in range(len(t)):
        if t[i] == 0:
            p[i] = (1 - beta + 4 * beta / np.pi) / np.sqrt(Tsym)
        elif abs(t[i]) == Tsym / (4 * beta):
            t1 = (1 + 2 / np.pi) * np.sin(np.pi / (4 * beta))
            t2 = (1 - 2 / np.pi) * np.cos(np.pi / (4 * beta))
            t3 = beta / np.sqrt(2 * Tsym)
            p[i] = t3 * (t1 + t2)
        else:
            num = (np.sin(np.pi * t[i] * (1 - beta) / Tsym) +
                   4 * beta * t[i] / Tsym * np.cos(np.pi * t[i] * (1 + beta) / Tsym))
            denom = (np.pi * t[i] / Tsym * (1 - (4 * beta * t[i] / Tsym) ** 2))
            p[i] = num / denom
    return p / np.sqrt(np.sum(p ** 2))  # Normalize energy

def upsample(bpsk_symbols, pulse, L):
    upsampled_data = np.zeros(len(bpsk_symbols) * L)
    upsampled_data[::L] = bpsk_symbols
    return convolve(upsampled_data, pulse)

def awgn_channel(signal, snr_db):
    snr_linear = 10 ** (snr_db / 10)
    noise_std = np.sqrt(1 / snr_linear)
    noise = noise_std * np.random.randn(*signal.shape)
    return signal + noise

def downsampling(received_signal, pulse, num_bits, L):
    matched_filter_out = convolve(received_signal, pulse)
    # delay = (len(pulse) - 1) // 2
    # downsampled = matched_filter_out[2 * delay + 1::L]
    # detected_bpsk_symbols = np.where(downsampled >= 0, 1, -1)
    return  matched_filter_out

# Parameters
num_bits = 10000
input_bits = np.random.randint(0, 2, num_bits)
bpsk_symbols = 2 * input_bits - 1
Tsym, beta, L, Nsym = 1, 0.3, 4, 8
pulse = srrc_pulse(beta, Nsym, Tsym, L)
transmitted_signal = upsample(bpsk_symbols, pulse, L)

snr_values = [-10, 0, 10]  # Eye diagram for 3 different SNRs

for snr in snr_values:
    received_signal = awgn_channel(transmitted_signal, snr)
    matched_output = downsampling(received_signal, pulse, num_bits, L)

    # Plot Eye Diagram
    num_traces = 100
    samples_per_eye = 3* L
    eye_data = matched_output[:num_traces * samples_per_eye].reshape(num_traces, samples_per_eye)
    plt.figure(figsize=(6, 4))
    for trace in eye_data:
        plt.plot(trace, color='blue', alpha=0.5)
    plt.title(f"Eye Diagram at SNR = {snr} dB")
    plt.xlabel("Sample Index")
    plt.ylabel("Amplitude")
    plt.grid(True)
    plt.show()
