import numpy as np
import cv2
import matplotlib.pyplot as plt
import scipy.special
import imageio

# Step 1: Read and Resize Image to 256x256
image_path = "D:/cameraman.png"
image = imageio.imread(image_path)
image = cv2.resize(image, (256, 256))
rows, cols = image.shape
plt.imshow(image, cmap='gray')
plt.title('Original Image')
plt.axis('off')
plt.show()

# Step 2: Convert Image to 1D Array
image_1d = image.flatten()

# Step 3: Convert Each Pixel to 8-bit Binary
binary_array = np.array([format(pixel, '08b') for pixel in image_1d])

# Convert Binary Strings to 1D Bit Array
binary_bits = np.array([int(bit) for binary in binary_array for bit in binary])

# Step 4: BPSK Mapping (0 → +1, 1 → -1)
bpsk_symbols = 1 - 2 * binary_bits  

# Step 5: Plot the Constellation Diagram (Transmitted)
plt.figure(figsize=(6, 6))
plt.scatter(bpsk_symbols, np.zeros_like(bpsk_symbols), marker='o', color='blue', s=100)
plt.title("BPSK Constellation Diagram (Transmitted Signal)")
plt.xlabel("In-phase")
plt.ylabel("Quadrature")
plt.xlim(-2, 2)
plt.ylim(-0.1, 0.1)
plt.grid()
plt.show()

# Define SNR range
snr_db_range = np.arange(-10, 11, 2)  # From -10 dB to 10 dB in steps of 2
theoretical_ber = []
simulated_ber = []

for snr_db in snr_db_range:
    snr_linear = 10 ** (snr_db / 10)
    noise_std = np.sqrt(1 / (2 * snr_linear))
    
    # Generate Complex Gaussian Noise
    noise =      noise_std    * (np.random.randn(len(bpsk_symbols)) + 1j * np.random.randn(len(bpsk_symbols)))
    
    # Add Noise to BPSK Symbols
    received_signal = bpsk_symbols + noise
    
    # Plot Received Constellation Diagram
    plt.figure(figsize=(6, 6))
    plt.scatter(received_signal.real, received_signal.imag, marker='o', color='b')
    plt.title(f"BPSK Constellation Diagram (Received Signal, SNR={snr_db} dB)")
    plt.xlabel("In-phase")
    plt.ylabel("Quadrature")
    plt.xlim(-2, 2)
    plt.ylim(-2, 2)
    plt.grid()
    plt.show()
    
    # Decision Rule: BPSK Demodulation
    decoded_symbols = np.where(received_signal.real >= 0, 1, -1)
    
    # Convert Symbols Back to Bits
    decoded_bits = ['0' if symbol == 1 else '1' for symbol in decoded_symbols]
    
    # Convert Binary Bits Back to 8-bit Values
    decoded_binary_values = ["".join(decoded_bits[i:i+8]) for i in range(0, len(decoded_bits), 8)]
    decoded_pixel_values = np.array([int(b, 2) for b in decoded_binary_values if len(b) == 8], dtype=np.uint8)
    
    # Reshape Back to Image Dimensions
    decoded_image = decoded_pixel_values.reshape(rows, cols)
    
    # Show Reconstructed Image
    plt.figure(figsize=(6, 6))
    plt.imshow(decoded_image, cmap='gray')
    plt.title(f"Reconstructed Image (SNR={snr_db} dB)")
    plt.axis("off")
    plt.show()
    
    # Compute BER
    original_bits = [bit for binary in binary_array for bit in binary]  # Flatten original bits
    bit_errors = sum(1 for o, d in zip(original_bits, decoded_bits) if o != d)
    ber_sim = bit_errors / len(original_bits)
    ber_theo = scipy.special.erfc(np.sqrt(2 * snr_linear)) / 2
    
    simulated_ber.append(ber_sim)
    theoretical_ber.append(ber_theo)
    
    print(f"SNR: {snr_db} dB, Simulated BER: {ber_sim}, Theoretical BER: {ber_theo}")

# Plot SNR vs BER
plt.figure(figsize=(8, 6))
plt.semilogy(snr_db_range, simulated_ber, 'ro-', label='Simulated BER')
plt.semilogy(snr_db_range, theoretical_ber, 'b*-', label='Theoretical BER')
plt.xlabel("SNR (dB)")
plt.ylabel("Bit Error Rate (BER)")
plt.title("SNR vs BER for BPSK")
plt.legend()
plt.grid(True, which='both', linestyle='--')
plt.show()