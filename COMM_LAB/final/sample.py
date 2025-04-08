import numpy as np
import matplotlib.pyplot as plt

# Parameters
Ts = 1  # Symbol duration
samples_per_symbol = 100  # Resolution
t = np.linspace(-2*Ts, 2*Ts, 4*samples_per_symbol)  # Time axis

# Create rectangular pulse centered at t=0
rect_pulse = np.where(np.abs(t) <= Ts/2, 1.0, 0.0)

# Plot
plt.figure(figsize=(8, 3))
plt.plot(t, rect_pulse, label='Rectangular Pulse', color='blue')
plt.title('Rectangular Pulse in Time Domain')
plt.xlabel('Time (t)')
plt.ylabel('Amplitude')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
