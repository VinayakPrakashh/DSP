from numpy import *
from matplotlib.pyplot import *
from scipy.signal import freqz
N = int(input("ENTER VALUE OF N: "))
wc = 0.2*pi
fs =1000
def fir_rectangular(N,wc):
    n =arange(-N//2,N//2+1)
    hn = sin(wc*n)/pi*n
    window_func = ones(N+1)
    h = hn*window_func
    return h
def fir_hamming(N,wc):
    n =arange(-N//2,N//2+1)
    hn = sin(wc*n)/pi*n
    hamming_window = 0.54 - 0.46 * np.cos(2 * np.pi * n / N)
    h = hn*hamming_window
    return h
h_rect = fir_rectangular(N,wc)
h_hamming = fir_hamming(N,wc)

w,H = freqz(h_rect,worN=8000)
w2,H2 =freqz(h_hamming,worN=8000)
f = w*fs/(2*pi) #hz
subplot(2,2,1)
plot(f,20*log10(abs(H)))
title('Frequency  response of rectangular window')
subplot(2,2,2)
plot(f,abs(H))
title('Linear Frequency  response of rectangular window')
subplot(2,2,3)
plot(f,20*log10(abs(H2)))
title('Frequency  response of hamming window')
subplot(2,2,4)
plot(f,abs(H))
title('Linear Frequency  response of hammming window')
show()