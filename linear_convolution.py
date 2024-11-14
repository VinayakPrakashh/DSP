from numpy import *

xn = input("enter the input sequence: ")
xn = list(map(int,xn.split()))
hn = input("Enter the impusle response: ")
hn = list(map(int,hn.split()))
def linear_conv(xn,hn):
    N1 = len(xn)
    N2 = len(hn)

    N = N1+N2-1

    y = zeros(N)

    for n in range(N):
        for k in range(N1):
            if n-k>=0 and n-k<N2:
                y[n] += xn[k]*hn[n-k]
    return y

print("using user defined funtion:",linear_conv(xn,hn))
print("using funtion",convolve(xn,hn))
