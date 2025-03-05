import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy import special

# Function to add AWGN noise
def add_awgn_noise(signal, snr_db):
    """Add AWGN noise to the modulated BPSK signal"""
    snr_linear = 10 ** (snr_db / 10)  # Convert SNR from dB to linear scale
    noise_power = np.var(signal) / snr_linear  # Assume signal power = 1
    noise = np.sqrt(noise_power / 2) * (np.random.randn(len(signal)) + 1j * np.random.randn(len(signal)))
    return signal + noise

# Step 1: Read and process the image
image = cv2.imread('"D:\cameraman.png"', cv2.IMREAD_GRAYSCALE)
plt.imshow(image, cmap='gray')
plt.title('Original Image')
plt.axis('off')
plt.show()

# Convert image to binary bit stream
flattened_image = image.flatten()  # Convert 2D image to 1D array
bit_array = np.unpackbits(flattened_image)  # Convert pixel values to binary bits``

# Step 2: BPSK Modulation (Mapping bits: 0 → 1, 1 → -1)
modulated_signal = -2 * bit_array + 1

# Plot constellation diagram of transmitted BPSK signal
plt.scatter(np.real(modulated_signal), np.imag(modulated_signal))
plt.title('Constellation Diagram (BPSK Transmitted Signal)')
plt.grid()
plt.show()

# Step 3: Transmission over AWGN Channel and BER Calculation
snr_db_values = np.arange(-10, 11, 1)  # SNR from -10dB to 10dB
ber_values = []
theoretical_ber = []

for snr_db in snr_db_values:
    # Transmit signal with AWGN
    received_signal = add_awgn_noise(modulated_signal, snr_db)

    # Plot received constellation diagram for each SNR value
    plt.scatter(np.real(received_signal), np.imag(received_signal))
    plt.title(f'Constellation Diagram (SNR = {snr_db} dB)')
    plt.grid()
    plt.show()

    # Step 4: Demodulation and BER Calculation
    received_bits = (np.real(received_signal) <= 0).astype(int)
    bit_errors = np.sum(bit_array != received_bits)
    ber = bit_errors / len(bit_array)
    ber_values.append(ber)

    # Theoretical BER for BPSK in AWGN
    theoretical_ber.append(0.5 * special.erfc(np.sqrt(10 ** (snr_db / 10))))

    # Step 5: Reconstructing the image from received bits
    received_pixels = np.packbits(received_bits)[:image.size]
    received_image = received_pixels.reshape(image.shape)
    
    plt.imshow(received_image, cmap='gray')
    plt.title(f'Reconstructed Image at SNR = {snr_db} dB')
    plt.axis('off')
    plt.show()

# Step 6: Plot SNR vs BER
plt.semilogy(snr_db_values, ber_values, 'o-', label='Practical BER')
plt.semilogy(snr_db_values, theoretical_ber, 's--', label='Theoretical BER')
plt.xlabel('SNR (dB)')
plt.ylabel('Bit Error Rate (BER)')
plt.title('SNR vs BER')
plt.grid(True, which='both')
plt.legend()
plt.show()
