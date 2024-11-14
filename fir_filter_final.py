from numpy import *
from matplotlib.pyplot import *
from scipy.signal import freqz
N = 40
wc = 0.2*pi
fs =1000
n =arange(N+1)
def fir_rectangular(N,wc):
    n =arange(N+1)
    hn =  np.sinc(2*100/fs*(n-N/2))
    window_func = ones(N+1)
    h = hn*window_func
    return h
def fir_hamming(N,wc):
    n =arange(N+1)
    hn =  np.sinc(2*100/fs*(n-N/2))
    hamming_window = 0.54 - 0.46 * np.cos(2 * np.pi * n / N)
    h = hn*hamming_window
    return h
h_rect = fir_rectangular(N,wc)
h_hamming = fir_hamming(N,wc)

w,H = freqz(h_rect,worN=8000)
w2,H2 =freqz(h_hamming,worN=8000)
f = w*fs/(2*pi) #hz
f2 = w2*fs/(2*pi) #hz
subplot(1,2,1)
plot(f,20*log10(abs(H)),label='Rectangular Window')
plot(f2,20*log10(abs(H2)),label='Hamming Window')
xlabel('Frequency (Hz)')
ylabel('Magnitude (dB)')
title('Frequency response')
legend()
subplot(1,2,2)
stem(n, h_rect, label='Rectangular Window')
stem(n, h_hamming, label='Hamming Window')
xlabel('Sample Index')
ylabel('Amplitude')
title('Impulse Response of Rectangular and Hamming Windows')
legend()
show()
plot(f,abs(H),label='Rectangular Window')
plot(f2,abs(H2),label='Hamming Window')
xlabel('Frequency (Hz)')
ylabel('Magnitude')
show()