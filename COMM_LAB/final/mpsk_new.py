from numpy import *
from matplotlib.pyplot import *
from scipy.special import erfc

M_values = [2,4,8,16]
bit_error_array = []
theoretical_ber_array = []
def theoretical_ber(snr_db):
    snr_linear = 10**(snr_db/10)
    return 0.5*erfc(sqrt(2*snr_linear))
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
def demod(signal,M):
    angles = angle(signal)
    decoded_symbols = np.round( ( angles/(2*pi) )*M )%M
    return decoded_symbols.astype(int)
for M in M_values:
    bits_per_symbol = int(log2(M))
    bitstream = random.randint(0,2,10000)
    bitstream = bitstream[:len(bitstream)-( len(bitstream) % bits_per_symbol )]
    modulated = modulate(bitstream,M)
    
    for snr_db in snr_db_range:
      recieved_signal = add_awgn(modulated,snr_db,int(log2(M)))
      decoded_symbols = demod(recieved_signal,M)
      decoded_bits = array([list(binary_repr(s,width=bits_per_symbol)) for s in decoded_symbols]).astype(int).flatten()
      bit_error = sum(decoded_bits[:len(bitstream)] != bitstream)
      ber = bit_error/len(bitstream)
      bit_error_array.append(ber)
      theoretical = theoretical_ber(snr_db)
      t_ber = theoretical/len(bitstream)
      theoretical_ber_array.append(t_ber)
    semilogy(snr_db_range,bit_error_array)
    semilogy(snr_db_range,theoretical_ber_array)
    show()



