import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# Part (a): Generate normally distributed random variable and plot histogram
mean = 0
std_dev = 1
num_samples = 10000
data = np.random.normal(mean, std_dev, num_samples)

# Plot histogram and overlay Gaussian PDF
plt.figure(figsize=(10, 5))
plt.hist(data, bins=50, density=True, color='skyblue', alpha=0.7, label='Histogram')
x_vals = np.linspace(mean - 4*std_dev, mean + 4*std_dev, 1000)
pdf = norm.pdf(x_vals, mean, std_dev)
plt.plot(x_vals, pdf, 'r-', label='Gaussian PDF')
plt.title('Histogram with Gaussian Overlay')
plt.xlabel('Value')
plt.ylabel('Density')
plt.legend()
plt.grid()
plt.show()

# Part (b): Perform PCM and plot SQNR vs Number of Bits
bit_depths = [2, 3, 4, 5, 6]
sqnr = []
pcm_encoded_output_bits = None

for N in bit_depths:
    # Quantization
    max_val, min_val = np.max(data), np.min(data)
    num_levels = 2**N
    step_size = (max_val - min_val) / (num_levels - 1)
    quantized_indices = np.round((data - min_val) / step_size).astype(int)

    if N == 4:  # Save PCM-encoded output in bits for N=4
        pcm_encoded_output_bits = [format(index, '0{}b'.format(N)) for index in quantized_indices]

    # Reconstruct quantized signal (optional, for verification)
    quantized_signal = quantized_indices * step_size + min_val

    # SQNR calculation
    quantization_error = data - quantized_signal
    signal_power = np.mean(data**2)
    noise_power = np.mean(quantization_error**2)
    sqnr.append(10 * np.log10(signal_power / noise_power))

# Plot SQNR vs Number of Bits
plt.figure(figsize=(10, 5))
plt.plot(bit_depths, sqnr, marker='o', label='SQNR vs Bit Depth')
plt.title('SQNR vs Number of Bits')
plt.xlabel('Bit Depth (N)')
plt.ylabel('SQNR (dB)')
plt.legend()
plt.grid()
plt.show()

# Display PCM encoded output for N=4 in bits (first 20 values for brevity)
print("First 20 PCM Encoded Values (in bits) for N=4:")
if pcm_encoded_output_bits:
    print(pcm_encoded_output_bits[:20])
else:
    print("PCM encoded output for N=4 was not generated.")