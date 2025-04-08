import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.signal import convolve

print("Name:Aaron Jacob")
print("Class:S6 ECE")
print("Roll no:4")

def srrc_pulse(Tsym, beta, L, Nsym):
    """Generates a Square-Root Raised Cosine (SRRC) pulse while handling singularities."""
    t = np.arange(-Nsym / 2, Nsym / 2, 1 / L)
    p = np.zeros_like(t)
    
    for i in range(len(t)):
        if t[i] == 0:
            p[i] = (1 - beta + 4 * beta / np.pi) / np.sqrt(Tsym)
        elif abs(t[i]) == Tsym / (4 * beta):
            p[i] = (beta / np.sqrt(2 * Tsym)) * (
                (1 + 2 / np.pi) * np.sin(np.pi / (4 * beta)) + (1 - 2 / np.pi) * np.cos(np.pi / (4 * beta))
            )
        else:
            num = (np.sin(np.pi * t[i] * (1 - beta) / Tsym) +
                   4 * beta * t[i] / Tsym * np.cos(np.pi * t[i] * (1 + beta) / Tsym))
            denom = (np.pi * t[i] / Tsym * (1 - (4 * beta * t[i] / Tsym) ** 2))
            p[i] = num / denom
    
    return p / np.sqrt(np.sum(p ** 2))

def upsample_and_filter(symbols, pulse, L):
    """Upsamples the symbols and applies the SRRC filter."""
    upsampled = np.zeros(len(symbols) * L)
    upsampled[::L] = symbols
    return convolve(upsampled, pulse, mode='full')

def add_awgn(signal, snr_db):
    """Adds AWGN noise to the signal based on SNR (dB)."""
    snr_linear = 10 ** (snr_db / 10)
    noise_power = 1 / (2 * snr_linear)
    noise = np.sqrt(noise_power) * np.random.randn(*signal.shape)
    return signal + noise

def downsample_and_filter(received_signal, pulse, L):
    """Performs matched filtering and downsampling."""
    matched_output = convolve(received_signal, pulse, mode='full')
    delay = (len(pulse) - 1) // 2
    return matched_output[2 * delay + 1::L]

def plot_eye_diagram(signal, L, nSamples, nTraces, snr, beta, ax):
    """Plots the eye diagram for given parameters."""
    total_samples = nSamples * nTraces
    signal = signal[:total_samples]  # Trim signal for visualization
    reshaped_signal = signal.reshape(nTraces, nSamples)
    
    for trace in reshaped_signal:
        ax.plot(trace, color='b', alpha=0.7)
    ax.set_title(f"Eye Diagram (SNR={snr} dB, β={beta})")
    ax.set_xlabel("Time")
    ax.set_ylabel("Amplitude")
    ax.grid()

def simulate_eye_diagram():
    """Runs the full simulation and plots eye diagrams for multiple SNR values and roll-off factors."""
    # Load image and convert to binary
    image = cv2.imread("D:\cameraman.png", cv2.IMREAD_GRAYSCALE)
    bits = np.unpackbits(image.flatten())
    symbols = np.where(bits == 0, -1, 1)  # BPSK mapping

    # Define parameters
    Tsym, L, Nsym = 1, 4, 8
    snr_values = [-5, 0, 10, 20]
    beta_values = [0.2, 0.8]

    # Create figure for eye diagrams
    fig, axes = plt.subplots(len(snr_values), len(beta_values), figsize=(12, 10))
    fig.suptitle("Eye Diagrams for Different SNRs and Roll-off Factors", fontsize=14)

    for i, snr in enumerate(snr_values):
        for j, beta in enumerate(beta_values):
            pulse = srrc_pulse(Tsym, beta, L, Nsym)
            transmitted_signal = upsample_and_filter(symbols, pulse, L)
            received_signal = add_awgn(transmitted_signal, snr)
            filtered_signal = downsample_and_filter(received_signal, pulse, L)
            
            plot_eye_diagram(filtered_signal, L, nSamples=3 * L, nTraces=100, snr=snr, beta=beta, ax=axes[i, j])

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.show()

simulate_eye_diagram()