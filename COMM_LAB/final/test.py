import numpy as np
import matplotlib.pyplot as plt
# Square Root Raised Cosine (SRRC) Filter
L = 4
def SRRC(t, T_sym, beta):
    if t == 0:
        return (1 / np.sqrt(T_sym)) * (1 - beta + (4 * beta / np.pi))

    if np.abs(t) == T_sym / (4 * beta):
        term_1 = (1 + 2 / np.pi) * np.sin(np.pi / (4 * beta))
        term_2 = (1 - 2 / np.pi) * np.cos(np.pi / (4 * beta))
        return (beta / np.sqrt(2 * T_sym)) * (term_1 + term_2)

    C = 4 * beta * t / T_sym
    num_sin = np.sin(np.pi * t * (1 - beta) / T_sym)
    num_cos = np.cos(np.pi * t * (1 + beta) / T_sym)

    # Avoid division by zero
    den = (np.pi * t / T_sym) * (1 - (C ** 2))
    den = np.where(den == 0, 1e-10, den)  # Replace zero with a small number

    num = num_sin + C * num_cos
    return (1 / np.sqrt(T_sym)) * (num / den)

t = np.arange(-8 / 2, 8 / 2, 1 / L)
p = np.array([SRRC(t_, 1, 0.3) for t_ in t])
# Plot SRRC impulse response
plt.plot(t, p)
plt.title("Impulse Response of SRRC")
plt.xlabel("Time")
plt.ylabel("Amplitude")
plt.grid(True)
plt.show()