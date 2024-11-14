from numpy import *
from scipy.fft import fft,ifft
xn=[1,2,3,4]
hn=[1,2]
def circular_conv(xn,hn):
    N1=len(xn)
    N2=len(hn)
    N =max([N1,N2])
    f1 = fft(xn,N)
    f2 = fft(hn,N)

    result = zeros(N)
    result = ifft(f1*f2)
    return result.real

print(circular_conv(xn,hn))