import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

r = 74
sigma = 1
nsamples = 100
X = np.random.normal(r, sigma, nsamples)

def perform_pcm(samples, n_bits):
    """Performs Pulse Code Modulation on a set of samples.

    Args:
        samples (np.ndarray): The input samples.
        n_bits (int): The number of bits for quantization.

    Returns:a
        tuple: A tuple containing:
            - quantized_samples (np.ndarray): The quantized samples.
            - quantization_error (np.ndarray): The difference between original and quantized samples.
    """
    levels = 2**n_bits
    min_val = np.min(samples)
    max_val = np.max(samples)
    step_size = (max_val - min_val) / levels

    # Create quantization levels
    quantization_boundaries = np.linspace(min_val, max_val, levels + 1)
    quantized_levels = (quantization_boundaries[:-1] + quantization_boundaries[1:]) / 2

    # Quantize the samples
    quantized_indices = np.digitize(samples, quantization_boundaries[1:-1])
    quantized_samples = quantized_levels[quantized_indices]
    quantization_error = samples - quantized_samples

    return quantized_samples, quantization_error

def calculate_sqnr(original, quantized):
    """Calculates the Signal-to-Quantization Noise Ratio (SQNR) in dB.

    Args:
        original (np.ndarray): The original samples.
        quantized (np.ndarray): The quantized samples.

    Returns:
        float: The SQNR in dB.
    """
    signal_power = np.mean(original**2)
    noise_power = np.mean((original - quantized)**2)
    if noise_power == 0:
        return np.inf  # Avoid division by zero
    sqnr_linear = signal_power / noise_power
    sqnr_db = 10 * np.log10(sqnr_linear)
    return sqnr_db

if __name__ == "__main__":
    n_bits_range = range(2, 7)  # N varies from 2 to 6
    sqnr_values = []

    for n_bits in n_bits_range:
        quantized_samples, quantization_error = perform_pcm(X, n_bits)
        sqnr_db = calculate_sqnr(X, quantized_samples)
        sqnr_values.append(sqnr_db)
        print(f"Number of bits (N): {n_bits}, SQNR: {sqnr_db:.2f} dB")

    # Plot SQNR vs. Number of Bits
    plt.figure(figsize=(8, 6))
    plt.plot(n_bits_range, sqnr_values, marker='o')
    plt.xlabel("Number of Bits (N)")
    plt.ylabel("Signal-to-Quantization Noise Ratio (SQNR) [dB]")
    plt.title("SQNR vs. Number of Bits for PCM (Mean=74, SD=1)")
    plt.grid(True)
    plt.xticks(n_bits_range)
    plt.show()

    # Optional: Visualize the original and quantized signals for a specific N
    n_bits_visualize = 3
    quantized_visualize, _ = perform_pcm(X, n_bits_visualize)

    plt.figure(figsize=(10, 5))
    plt.plot(np.arange(nsamples), X, label='Original Samples')
    plt.plot(np.arange(nsamples), quantized_visualize, label=f'Quantized Samples (N={n_bits_visualize})', linestyle='--')
    plt.xlabel("Sample Index")
    plt.ylabel("Amplitude")
    plt.title(f"Original and Quantized Signals (N={n_bits_visualize})")
    plt.legend()
    plt.grid(True)
    plt.show()