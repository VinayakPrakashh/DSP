
import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.signal import convolve
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
def visualize_pulse_shaping():
    """Visualizes the pulse shaping process and its effects."""
    
    # Load image and convert to binary
    image = cv2.imread(r"D:/cameraman.png", cv2.IMREAD_GRAYSCALE)
    bits = np.unpackbits(image.flatten())
    symbols = np.where(bits == 0, -1, 1)  # BPSK Mapping
    print("First 30 symbols:", symbols[:30])
    
    # Define parameters
    Tsym, beta, L, Nsym = 1, 0.3, 4, 8
    pulse = srrc_pulse(Tsym, beta, L, Nsym)
    
    # Plot the SRRC pulse
    plt.figure()
    plt.plot(pulse)
    plt.title("SRRC Pulse")
    plt.xlabel("Sample Index")
    plt.ylabel("Amplitude")
    plt.grid()
    plt.show()
    
    # Transmit signal
    transmitted_signal = upsample_and_filter(symbols, pulse, L)
    print("First 30 samples of transmitted signal:", transmitted_signal[:30])
    
    # Plot the transmitted signal
    plt.figure()
    plt.plot(np.real(transmitted_signal), label="Real Part")
    plt.plot(np.imag(transmitted_signal), label="Imaginary Part")
    plt.title("Transmitted Signal (Pulse Shaped)")
    plt.xlabel("Sample Index")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.grid()
    plt.show()
    
    # Add noise and visualize received signal
    snr = 10  # Example SNR in dB
    received_signal = add_awgn(transmitted_signal, snr)
    print("First 30 samples of received signal:", received_signal[:30])
    
    plt.figure()
    plt.plot(np.real(received_signal), label="Real Part")
    plt.plot(np.imag(received_signal), label="Imaginary Part")
    plt.title(f"Received Signal with AWGN (SNR={snr} dB)")
    plt.xlabel("Sample Index")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.grid()
    plt.show()
    
    # Matched filtering and demodulation
    detected_symbols = downsample_and_demodulate(received_signal, pulse, L, len(bits))
    recovered_bits = (detected_symbols == 1).astype(np.uint8)
    recovered_bits = np.pad(recovered_bits, (0, 8 - len(recovered_bits) % 8), mode='constant')[:len(bits)]
    recovered_image = np.packbits(recovered_bits).reshape(image.shape)
    
    # Plot the reconstructed image
    plt.figure()
    plt.imshow(recovered_image, cmap='gray')
    plt.title(f"Reconstructed Image at SNR={snr} dB")
    plt.axis('off')
    plt.show()

# Call the visualization function
visualize_pulse_shaping()