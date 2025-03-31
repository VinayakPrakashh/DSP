
import numpy as np
import matplotlib.pyplot as plt

# Define parameters
r= 74 #roll number
modr = (r % 5) + 1  #mod(r,5) + 1
fs = 32000 #fm=4, nyquist=2*4=8, 4times nyquist=4*8=32
t = np.linspace(0, 1, fs)  # Time vector=time b/w 0,1

# Generate raised sine wave signal
s = modr * (1 + np.cos(8 * np.pi * t)) / 2

# Quantization levels
L= [4, 8, 16, 32, 64]
SQNR_values = []#to store square to quantization noise ratio val for each val

plt.figure(figsize=(12, 6))

for i in L:
    
    # Perform uniform quantization
    minval, maxval = np.min(s), np.max(s)
    stepsize = (maxval - minval) / (i - 1)
    quantizedsignal = np.round((s - minval) / stepsize) * stepsize + minval#Maps signal values into integer levels (0 to L-1)* step_size + min_val:Converts back to original amplitude scale.
    
    # Compute quantization noise and SQNR
    quantizationerror = s - quantizedsignal
    signal_power = np.mean(s ** 2)
    noise_power = np.mean(quantizationerror ** 2)
    SQNR = 6.02*np.log2(i)+1.76
    SQNR_values.append(SQNR)

    # Plot quantized signals for visualization
  
    plt.plot(t, quantizedsignal, label=f"L={L}")
   

plt.plot(t, s, label="Original Signal")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.title("Quantized Signal for Different L values")
plt.legend()
plt.show()

# Plot SQNR vs N
N_values = np.log2(L)
plt.figure()

plt.plot(N_values, SQNR_values,)
plt.xlabel("Number of Bits (N = log2(L))")
plt.ylabel("SQNR (dB)")
plt.title("Signal-to-Quantization Noise Ratio vs Number of Bits")
plt.grid(True)
plt.show()
# PCM Modulation for L = 32
L_pcm = 32
minval, maxval = np.min(s), np.max(s)
stepsize = (maxval - minval) / (L_pcm - 1)
quantizedsignal_pcm = np.round((s - minval) / stepsize) * stepsize + minval

# Binary encoding
pcm_encoded = ((quantizedsignal_pcm - minval) / stepsize).astype(int)
pcm_binary = [np.binary_repr(val, width=int(np.log2(L_pcm))) for val in pcm_encoded]

# Display first 10 PCM encoded values as an example
print("First 10 PCM encoded values:")
for i in range(10):
    print(f"Sample {i}: {pcm_binary[i]}")