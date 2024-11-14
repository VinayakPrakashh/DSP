from matplotlib.pyplot import *
from numpy import *
N = int(input("Enter the value of N: "))
def dft_matrix(N):
    n=arange(N)
    k = n[:,None]

    omega = exp(-2j * pi * n * k / N)
    return omega
if((N & (N - 1)) == 0):
    print("N is a power of 2")
    omega = dft_matrix(N)
    print(omega)
    figure(figsize=(12,6))
    subplot(1,2,1)
    title(f"real values of DFT with N={N}")
    imshow(real(omega))
    colorbar()
    subplot(1,2,2)
    title(f"imaginary values of DFT with N={N}")
    imshow(imag(omega))
    colorbar()
    show()
else:
    print("N should be a power of 2")
    N = int(input("Enter the value of N: "))
