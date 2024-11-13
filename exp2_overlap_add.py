from numpy import *

xn = [1,2,3,4,5,6,7,8,12]
hn = [1,1,1]
N = 8# put any values of N

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

def overlap_add(xn,hn,M,L):
    padded_h1 = pad(hn,(0,L-1))
    splitted_x1 = [xn[i:i+L] for i in range(0,len(xn),L)]
    padded_x1 = [i+[0]*(M-1) for i in splitted_x1]
    result_length = len(xn)+len(padded_h1)-1
    result = zeros(result_length)
    for i,sublist in enumerate(padded_x1):
        conv_res = circular_conv(sublist,padded_h1)
        start_index = i*L
        end_index = start_index+len(conv_res)
        result[start_index:end_index]+= conv_res
    print(result[:len(result)-L+1])

overlap_add(xn,hn,M,L)