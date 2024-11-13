from numpy import *

xn=[1,2,3,4,5,6,7,8,9,10,11,12,13,14]
hn=[1,1,1]
N=8 # put any values of N
M = len(hn)
L = N-M+1
def circular_conv(xn,hn):
    N1=len(xn)
    N2=len(hn)
    N = max([N1,N2])
    xn = pad(xn,(0,(N-N1)))
    hn = pad(hn,(0,(N-N2)))
    result = zeros([N,N])
    for i in range(N):
        result[:,i] = roll(xn,i)
    return dot(result,hn)

def overlap_save(xn,hn,M,L):
    padded_hn = pad(hn,(0,L-1))
    print(padded_hn)
    splitted_x1 = [xn[i:i+L] for i in range(0,len(xn),L)]
    splitted_x1[0] = [0]*(M-1) + splitted_x1[0]
    for i in range(1,len(splitted_x1)):
        splitted_x1[i] = splitted_x1[i-1][-(M-1):]+splitted_x1[i]
    splitted_x1 = [sublist+[0]*(len(splitted_x1[0])-len(splitted_x1[i])) for i,sublist in enumerate(splitted_x1)]
    result =[]
    for sublist in splitted_x1:
        conv_res = circular_conv(sublist,hn)
        conv_res = conv_res[M-1:]
        result.extend(conv_res)
    result = [int(x) for x in result]
    return result[:len(xn)+len(hn)-1]
print(overlap_save(xn,hn,M,L))