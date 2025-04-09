import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# Generate a normally distributed random variable
np.random.seed(42)  # For reproducibility
num_samples = 10000
mean = 0  # Mean of the distribution
std_dev = 1  # Standard deviation of the distribution
random_variable = np.random.normal(mean, std_dev, num_samples)

# Plot the histogram
plt.figure(figsize=(8, 5))
count, bins, _ = plt.hist(random_variable, bins=50, density=True, alpha=0.6, color='blue', label="Histogram")

# Overlay the theoretical Gaussian curve
x = np.linspace(min(bins), max(bins), 1000)
pdf = norm.pdf(x, mean, std_dev)  # Gaussian PDF
plt.plot(x, pdf, 'r-', label="Theoretical Gaussian Curve")

# Add labels and legend
plt.title("Histogram of Normally Distributed Random Variable")
plt.xlabel("Value")
plt.ylabel("Density")
plt.legend()
plt.grid(True)
plt.show()