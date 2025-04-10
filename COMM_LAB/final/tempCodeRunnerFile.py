from numpy import *
from matplotlib.pyplot import *
from scipy.signal import convolve
ber_values = []
def srrc(Tsym,Nsym,L,beta):
    t = arange(-Nsym/2,Nsym/2,1/L)
    p = zeros_like(t)
    # t==0
    x2 = 4*beta/pi
    x1 = 1-beta
    comm = 1/sqrt(Tsym)
    check = Tsym/(4*beta)

    #t == tsym/4beta
    y1 = 1-(2/pi)
    y2 = 1+(2/pi)
    op = pi/(4*beta)
    comm1 = beta/sqrt(2)

    # p[t]




    for i in range(len(t)):
       if t[i] == 0:
           
           p[i] = ( x1 + x2 ) * comm
       elif t[i] == abs(check):
           p[i] = ( y1*sin(op) + y2*cos(op) )*comm*comm1
       else:
            w1 = pi*t[i]*(1-beta) / Tsym
            w2 = pi*t[i]*(1+beta) / Tsym
            z = pi*t[i] / Tsym
            c = 4*beta*t[i] / Tsym 
            num =  sin(w1) + c*cos(w2) 
            denom = z*(1-c**2)
            
            p[i] =( num/denom ) * comm
    plot(t,p)
    show()
    return p/(sqrt(sum(p**2)))
           
def upsample_filter(signal,pulse,L):
    symbol = zeros(len(signal)*L)
    symbol[::L] = signal
    print(symbol)
    return convolve(symbol,pulse,mode='full')

def add_noise(signal,snr_db):
    snr = 10**(snr_db/10)
    noise_pow = 1/(2*snr)
    noise = sqrt(noise_pow) * (random.randn(*signal.shape) + 1j*random.randn(*signal.shape))
    return signal + noise

def demod(signal,L,len_bits):
    matched_out = convolve(signal,pulse,mode='full')
    # delay = (len(pulse)-1)//2
    # sampled = demodulated[2*delay+1::L]
    # detected_symbols = np.where(sampled >= 0, 1, -1)
    return matched_out

snr_db_range = arange(-10,10,.2)


bits = random.randint(0,2,10000)

bpsk_mapping = where(bits == 0,-1,1)

Tsym,Nsym,L,beta = 1,8,4,0.3

pulse = srrc(Tsym,Nsym,L,beta)

transmitted_symbols = upsample_filter(bpsk_mapping,pulse,L)

for snr_db in snr_db_range:
    recieved_signal = add_noise(transmitted_symbols,snr_db)
    matched_out = demod(recieved_signal,L,len(bits))

    n_samples = 3*L
    traces = 100
    eye_data = matched_out[:n_samples*traces].reshape(traces,n_samples)
    for trac in eye_data:
        plot(trac,color='blue',alpha=0.5)
    show()

