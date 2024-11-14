from numpy import *
import time
from matplotlib.pyplot import *

def DFT(N):
    x = random.random(N)
    n = arange(N)
    k = n[:,None]
    e = exp(-2j * pi * k * n / N)
    X = dot(e, x)
    return X
def custom_fft(N):
    x = random.random(N)
    res = fft.fft(x)
    return res
def time_complexity():
    dft_time = []
    fft_time = []
    gamma = arange(0, 15, 1)
    for i in gamma:
        N = 2**i
        start_time = time.time()
        DFT(N)
        end_time = time.time()
        dft_time.append(end_time-start_time)
        start_time = time.time()
        custom_fft(N)
        end_time = time.time()
        fft_time.append(end_time-start_time)
    plot(gamma,dft_time,label = 'DFT_time')
    plot(gamma,fft_time,label = 'FFT_time')
    legend()
    show()
time_complexity()
    
