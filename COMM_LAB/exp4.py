"""
Program: Pulse code modulation
Description: Program to 
1. Generate raised sine wave of equation in question by sampling it at four times the Nyquist rate
  Quantize the samples of s(t) with L = 4, 8, 16, 32, 64 where L is the number of quantization levels.
2. Compute the signal to quantization noise ratio and plot it against N, where N = log2(L)
  is the number of bits used for quantization.
3. Generate the PCM modulated output for L = 32 using binary encoding

Author: VINAYAK PRAKASH
Date: 27/02/2025
"""
# Student Information
print("prasoon pradeep")
print("Roll No:73")

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
    SQNR = 10 * np.log10(signal_power / noise_power)#Converts power ratio to decibels (dB)
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

# PCM Encoding for L = 32
L = 32
stepsize = (maxval - minval) / (L - 1)
quantized_signal = np.round((s - minval) / stepsize).astype(int)  # Convert to integer levels

# Convert to binary representation
N = int(np.log2(L))#Determines number of bits for PCM encoding
pcm_encoded = [format(val, f'0{N}b') for val in quantized_signal]#Converts integer levels to binary strings,format(val, f'0{N}b') converts the integer val to an N-bit binary string,Pad with leading zeros if necessary.
print("First 10 PCM Encoded Samples:")
for i in range(10):
    print(f"Sample {i+1}: {pcm_encoded[i]}")