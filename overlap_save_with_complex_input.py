from numpy import *

# Taking user input for xn and hn
xn = [1+1j, 2+2j, 3+3j, 6+4j, 5+5j, 2+6j, 7+7j, 8+8j, 9+9j, 2+2j, 3+3j, 4+4j, 5+5j, 6+6j, 7+7j]
hn = [1+1j, 1+1j, 1+1j]
N = int(input("Enter the value of N: "))
M = len(hn)
L = N-M+1
def circular_conv(xn,hn):
    N1=len(xn)
    N2=len(hn)
    N = max([N1,N2])
    xn = pad(xn,(0,(N-N1)))
    hn = pad(hn,(0,(N-N2)))
    result = zeros([N,N],dtype=complex)
    for i in range(N):
        result[:,i] = roll(xn,i)
    return dot(result,hn)

def overlap_save(xn,hn,M,L):
    padded_hn = pad(hn,(0,L-1))
    splitted_x1 = [xn[i:i+L] for i in range(0,len(xn),L)]
    splitted_x1[0] = [0]*(M-1) + splitted_x1[0]
    for i in range(1,len(splitted_x1)):
        splitted_x1[i] = splitted_x1[i-1][-(M-1):]+splitted_x1[i]
    splitted_x1 = [sublist+[0]*(len(splitted_x1[0])-len(splitted_x1[i])) for i,sublist in enumerate(splitted_x1)]
    result =[]
    for sublist in splitted_x1:
        conv_res = circular_conv(sublist,padded_hn)
        conv_res = conv_res[M-1:]
        result.extend(conv_res)
    result = [complex(x) for x in result]
    return result[:len(xn)+len(hn)-1]
print(overlap_save(xn,hn,M,L))