from numpy import *
from matplotlib.pyplot import *
from scipy.signal import convolve
def SRRC(Tsym, Nsym, L, beta):
    t = arange(-Nsym / 2, Nsym / 2, 1 / L)
    p = zeros_like(t)
    comm = 1 / sqrt(Tsym)
    beta_4 = (4 * beta) / pi
    abs_1 = 1 + (2 / pi)
    abs_2 = 1 - (2 / pi)
    sc = pi / (4 * beta)

    for i in range(len(t)):
        ti = t[i]
        if ti == 0:
            p[i] = ((1 - beta) + beta_4) * comm
        elif abs(ti - Tsym / (4 * beta)) < 1e-6:
            p[i] = (abs_1 * sin(sc) + abs_2 * cos(sc)) * comm * (beta / sqrt(2))
        else:
            sin_t = pi * ti * (1 - beta) / Tsym
            cos_t = pi * ti * (1 + beta) / Tsym
            beta_t = 4 * beta * ti / Tsym
            pi_t = pi * ti / Tsym
            num = sin(sin_t) + (beta_t * cos(cos_t))
            denom = pi_t * (1 - beta_t ** 2)
            p[i] = comm * (num / denom)
    return p
# import matplotlib.pyplot as plt

# pulse = SRRC(Tsym=1, Nsym=6, L=8, beta=0.35)
# plt.plot(pulse)
# plt.title("SRRC Pulse")
# plt.grid(True)
# plt.show()

def upsample_filter(signal,pulse,L):
    upsampled = zeros(len(signal)*L)
    upsampled[::L] = signal
    return convolve(signal,pulse,mode='full')
def add_noise(signal,snr_db):
    snr_linear = 10**(snr_db/10)
    noise_pow = 1/(2*snr_linear)
    awgn = random.randn(*signal.shape) +(1j*random.randn(*signal.shape))
    noise = sqrt(noise_pow)* awgn
    return signal + noise

def downsampled(signal,pulse,L):




bitstream = random.randint(0,2,10000)

Tsym,Nsym,L,beta = 1,8,4,0.3
bpsk_mapping = where(bitstream == 0,-1,1)
print(bpsk_mapping[0:30])
pulse = SRRC(Tsym,Nsym,L,beta)
upsampled_signal = upsample_filter(bpsk_mapping,pulse,L)
snr_db_range = arange(-10,10,5)

for snr_db in snr_db_range:
    recieved_signal = add_noise(upsampled_signal,snr_db)
    