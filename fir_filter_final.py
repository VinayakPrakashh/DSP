from numpy import *
from matplotlib.pyplot import *
from scipy.signal import freqz
N = 40
wc = 2*pi*100
fs = 1000
n  = arange(N+1)
def rectangular(N,wc):
    fc = wc/(pi*2)
    n = arange(N+1)
    hn = sinc(2*fc/fs*(n-N/2))
    rectangular = ones(N+1)
    h_rect = hn * rectangular
    return h_rect
def fir_hamming(N,wc):
    fc = wc/(pi*2)
    n = arange(N+1)
    hn = sinc(2*fc/fs*(n-N/2))
    hamming = 0.54 - 0.46 *cos(2*pi*n/N)
    h_ham = hn * hamming
    return h_ham
h_rect = rectangular(N,wc)
h_hamming = fir_hamming(N,wc)

w,H = freqz(h_rect,worN=8000)
w2,H2 = freqz(h_hamming,worN=8000)
f = w*fs /2*pi
f2 = w2*fs /2*pi
plot(f,20*log10(abs(H)),label='rect')
plot(f2,20*log10(abs(H2)),label='hamming')
legend()
show()

plot()