w, H_rectangular = freqz(h_rectangular, worN=8000)
w, H_hamming = freqz(h_hamming, worN=8000)

frequencies = w * fs / (2 * np.pi)

plt.figure(figsize=(14, 6))

plt.subplot(1, 2, 1)

plt.plot(frequencies, 20*np.log10(np.abs(H_rectangular)),
         label='Rectangular Window')
plt.plot(frequencies, 20*np.log10(np.abs(H_hamming)),
         label='Hamming Window')

plt.xlabel('Frequency (Hz)')
plt.ylabel('Magnitude (dB)')
plt.title('Frequency Response of Rectangular and Hamming Windows')
plt.legend()