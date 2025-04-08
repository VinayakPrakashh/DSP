from numpy import *
from matplotlib.pyplot import *
from scipy.signal import convolve
from scipy.special import erfc

def theoretical_ber(snr_db):
    k = np.log2(2)
    
    return erfc(np.sqrt(k * 10**(snr_db / 10)) / np.sqrt(2)) / k

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
    return p / np.sqrt(np.sum(p ** 2))

def upsample_filter(signal, pulse, L):
    upsampled = zeros(len(signal) * L)
    upsampled[::L] = signal
    return convolve(upsampled, pulse, mode='full')

def add_noise(signal, snr_db):
    snr_linear = 10 ** (snr_db / 10)
    noise_pow = 1 / ( 2*snr_linear)
    awgn = random.randn(*signal.shape)
    noise = sqrt(noise_pow) * awgn
    return signal + noise

def downsampled(signal, pulse, L, bits):
    matched_out = convolve(signal, pulse, mode='full')
    delay = (len(pulse) - 1) // 2
    sampled_out = matched_out[2 * delay + 1::L]
    detected_symbols = where(sampled_out >= 0, 1, -1)
    return detected_symbols[:bits]  # Ensure correct length

# Main simulation
ber_values = []
theor_ber = []

bitstream = random.randint(0, 2, 10000)  # Generate random bitstream
Tsym, Nsym, L, beta = 1, 8, 4, 0.3  # Parameters
bpsk_mapping = where(bitstream == 0, -1, 1)  # Map bits to BPSK symbols
pulse = SRRC(Tsym, Nsym, L, beta)  # Generate SRRC pulse
upsampled_signal = upsample_filter(bpsk_mapping, pulse, L)  # Pulse shaping
snr_db_range = arange(-10, 10, 1)  # SNR range in dB

for snr_db in snr_db_range:
    recieved_signal = add_noise(upsampled_signal, snr_db)  # Add noise
    downsampled_bits = downsampled(recieved_signal, pulse, L, len(bitstream))  # Matched filtering and downsampling
    recovered_bits = where(downsampled_bits ==-1,0,1)
    errors = sum(recovered_bits != bitstream)  # Count bit errors
    err_t = theoretical_ber(snr_db)
    ber = errors / len(bitstream)  # Calculate BER
    ber_t = err_t/len(bitstream)

    ber_values.append(ber)
    theor_ber.append(ber_t)

# Plot BER curve
semilogy(snr_db_range, ber_values, 'o-', label="Simulated BER")  # Simulated BER
semilogy(snr_db_range, theor_ber, 'o--', label="Theoretical BER")  # Theoretical BER
xlabel("SNR (dB)")
ylabel("Bit Error Rate (BER)")
title("BER vs SNR Curve")
grid(True, which="both", linestyle="--")
legend()
show()