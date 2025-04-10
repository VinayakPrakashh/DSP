from numpy import *
from matplotlib.pyplot import *


L = 4
n_samples = 3*L
n_bits = 1000
snr_range = [-10,-5]
n_traces = 100

bits = random.randint(0,2,1000)

bpsk_mapped  = where(bits == 0,-1,1)

#upsample
symbols = zeros(len(bits)*L)

symbols[::L] = bpsk_mapped

def add_awgn(snr_db,n):
    snr = 10**(snr_db/10)
    N0 = 1/snr
    return random.normal(0,sqrt(N0/2),n)

for idx,snr_db in enumerate(snr_range):
    recieved_signal = symbols + add_awgn(snr_db,len(symbols))
    subplot(2,2,idx+1)
    for k in range(n_traces):
        start = k* n_samples
        end = start + n_samples +1
        if end>=len(recieved_signal):
            break
        segment = recieved_signal[start:end]
        t = linspace(0,len(segment)-1,len(segment)) / L
        plot(t,segment,alpha=0.6,color='blue')
show()
