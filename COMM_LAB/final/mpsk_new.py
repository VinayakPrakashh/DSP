from numpy import *
from matplotlib.pyplot import *
from scipy.special import erfc

M_values = [2,4,8,16]

def modulate(bitstream,M):
    k = int(log2(M))
    bit_group = bitstream.reshape(-1,k)
    symbols = array([int("".join(map(str,b)),2) for b in bit_group])
    angle = 2*pi*symbols/M
    return cos(angle) + 1j*sin(angle)

def add_awgn(signal,snr_db,bits_per_symbol):
    snr_linear = 10**(snr_db/10)
    noise_std = np.sqrt(1 / (2 * bits_per_symbol * snr_linear))
    noise = noise_std * (np.random.randn(*signal.shape) + 1j * np.random.randn(*signal.shape))
    return signal + noise
snr_db_range = arange(-10,10,1)
for M in M_values:
    bits_per_symbol = int(log2(M))
    bitstream = random.randint(0,2,10000)
    bitstream = bitstream[:len(bitstream)-( len(bitstream) % bits_per_symbol )]
    modulated = modulate(bitstream,M)
    
    for snr_db in snr_db_range:
      recieved_signal = add_awgn(modulated,snr_db,int(log2(M)))
      print(recieved_signal)
      
