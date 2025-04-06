from numpy import *
from matplotlib.pyplot import *
from scipy.signal import convolve


def srrc_filter(Tsym,L,beta,Nsym):
  t=arange(-Nsym/2 , Nsym/2, 1/L)
  p=zeros_like(t)
  for i in range (len(t)):
    if t[i]==0 :
      p[i] = (1 - beta + 4 * beta / np.pi) / np.sqrt(Tsym)
    elif abs(t[i]) == Tsym / (4 * beta):
      t1 = (1 + 2 / np.pi) * np.sin(np.pi / (4 * beta))
      t2 = (1 - 2 / np.pi) * np.cos(np.pi / (4 * beta))
      t3 = beta / np.sqrt(2 * Tsym)
      p[i] = t3 * (t1 + t2) 
    else:
      num = (np.sin(np.pi * t[i] * (1 - beta) / Tsym) +
      4 * beta * t[i] / Tsym * np.cos(np.pi * t[i] * (1 + beta) / Tsym))
      denom = (np.pi * t[i] / Tsym * (1 - (4 * beta * t[i] / Tsym) ** 2))
      p[i] = num / denom
  
  return p/sqrt(sum(p**2))


def upsample(bpsk_stream,pulse,L):
  upsampled = zeros(len(bpsk_stream)* L)
  upsampled[::L] = bpsk_stream
  return convolve(upsampled,pulse,mode='full')

def awgn_noise(transmitted_signal,snr):
  snr_linear = 10**(snr/10)
  noise_pwr = 1/(2*snr_linear)
  noise = sqrt(noise_pwr)*(random.randn(*transmitted_signal.shape) + 1j*random.randn(*transmitted_signal.shape))
  return transmitted_signal + noise
def convo(s,p):
  return convolve(s,p)



Tsym,L,beta,Nsym = 1,4,0.3,8
bit_stream = random.randint(0,2,10000)
bpsk_stream = where(bit_stream == 0 ,1 ,-1)

pulse=srrc_filter(Tsym,L,beta,Nsym)

transmitted_signal = upsample(bpsk_stream,pulse,L)

SNR_db = arange(-10,11,1)

for snr in SNR_db :
  received_signal = awgn_noise(transmitted_signal , snr)

  matched_filter_out = convo(received_signal,pulse)
  #print(matched_filter_out)
  #k=arange(0,len(matched_filter_out),1)
   
  n = 100
  samples = 3*L

  eye_data = matched_filter_out[:n*samples].reshape(n,samples)

  for trace in eye_data :
    plot(trace,color='blue',alpha=0.5)