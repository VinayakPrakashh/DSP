from numpy import *
from matplotlib.pyplot import *
from scipy.signal import convolve


L = 4
n_traces = 100
SNR_values = [-10, -5, 0, 5]
N_Symbols = 3*L

N_samples = 1000
#generate bits
bits = random.randint(0,2,N_samples)

#bpsk_mapping
symbols = where(bits ==0,-1,1)

# upsample

transmitted_symbols = zeros(len(bits)*L)
transmitted_symbols[::L] = symbols


#add noise
def add_awgn(snr,num_bits):
    N0 = 1/(10**(snr/10))
    return random.normal(0,sqrt(N0/2),num_bits)

for idx,snr in enumerate(SNR_values):
    recieved = transmitted_symbols + add_awgn(snr,len(bits))
    subplot(2,2,idx+1)
    for k in n_traces:
        start = k*N_Symbols
        end = start + N_Symbols + 1
        if end >= len(bits):
            break
        segment = recieved[start:end]
        t = linspace(0,len(segment)-1,len(segment)) / L
