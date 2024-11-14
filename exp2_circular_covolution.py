from numpy import *
from matplotlib.pyplot import *
from scipy.fft import fft,ifft

def find_cir_conv(x1,x2):
    N1 = len(x1)
    N2 = len(x2)
    N =max([N1,N2])
    x1 = pad(x1, (0, N - len(x1)))
    x2 = pad(x2, (0, N - len(x2)))
    conv=zeros((N,N),dtype=complex)
    
    for i in range(N):
        conv[:,i]=roll(x1,i)
    result = dot(conv,x2)
    return result
def find_cir_conv2(xn,hn):
    N = max([len(xn),len(hn)])
    xn1 = fft(xn,N)
    hn1=fft(hn,N)
    yn = zeros(N,dtype=complex)
    yn = ifft(xn1*hn1)
    return yn

x1 = eval(input("Enter the input sequence : "))
x2 = eval(input("Enter the impulse sequence : "))
print(x1,"    ",x2)
y = find_cir_conv(x1,x2)
y2 = find_cir_conv2(x1,x2)
print("y(n)=",y)
print("y(n)=",y2)