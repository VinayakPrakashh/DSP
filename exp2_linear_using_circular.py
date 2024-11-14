from numpy import *
def circular_conv(x1,h1):
    h1 = h1[:, newaxis]
    x2 = zeros([len(x1),len(x1)])

    for n in range(len(x1)):
        x2[:,n] = roll(x1,n)
    return dot(x2,h1)

x1 = [1,2,3,4]
h1 = [1,1,1]

n1 = len(x1)
n2 = len(h1)

n = n1+n2-1
x1 = pad(x1,(0,n-n1))
h1 = pad(h1,(0,n-n2))
print(circular_conv(x1,h1))