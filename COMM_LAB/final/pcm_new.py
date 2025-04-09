import numpy as np
import matplotlib.pyplot as plt

# Generate a normally distributed random variable
np.random.seed(42)  # For reproducibility
num_samples = 10000
mean = 0  # Mean of the distribution
std_dev = 1  # Standard deviation of the distribution
random_signal = np.random.normal(mean, std_dev, num_samples)
# Define bit depths (N values)
N_values = range(2, 7)  # N varies from 2 to 6
SQNR_values = []

# Perform PCM for each N
for N in N_values:
    L = 2 ** N  # Number of quantization levels
    minval, maxval = np.min(random_signal), np.max(random_signal)
    stepsize = (maxval - minval) / (L - 1)
    
    # Quantize the signal
    quantized_signal = np.round((random_signal - minval) / stepsize) * stepsize + minval
    
    # Compute SQNR
    quantization_error = random_signal - quantized_signal
    signal_power = np.mean(random_signal ** 2)
    noise_power = np.mean(quantization_error ** 2)
    SQNR = 10 * np.log10(signal_power / noise_power)
    SQNR_values.append(SQNR)
    
    # Display PCM encoded output for N=4
    if N == 4:
        quantized_indices = np.round((random_signal - minval) / stepsize).astype(int)
        bits_per_sample = N
        pcm_output = []
        for value in quantized_indices:
            binary = bin(value)[2:].zfill(bits_per_sample)  # Convert to binary and pad with zeros
            pcm_output.extend([int(bit) for bit in binary])
        print("PCM Encoded Output for N=4:")
        print(pcm_output)
        print("Length of PCM output:", len(pcm_output))

# Plot SQNR vs Number of Bits
plt.figure()
plt.plot(N_values, SQNR_values, marker='o')
plt.xlabel("Number of Bits (N)")
plt.ylabel("SQNR (dB)")
plt.title("Signal-to-Quantization Noise Ratio vs Number of Bits")
plt.grid(True)
plt.show()