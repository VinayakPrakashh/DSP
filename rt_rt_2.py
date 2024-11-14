import numpy as np
import matplotlib.pyplot as plt

# Define the ramp function
def ramp(t):
    return np.where(t>=0,t,0)

# Define the time range
t = np.linspace(-1, 5, 1000)

# Compute r(t) and r(t-2)
r_t = ramp(t)
r_t_minus_2 = ramp(t - 2)

# Compute r(t) - r(t-2)
result = r_t - r_t_minus_2

# Plot r(t) - r(t-2)
plt.plot(t, result, label='r(t) - r(t-2)')
plt.xlabel('Time (t)')
plt.ylabel('Amplitude')
plt.title('Plot of r(t) - r(t-2)')
plt.grid(True)
plt.legend()
plt.show()