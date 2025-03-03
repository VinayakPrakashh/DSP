import imageio
from numpy import *

image_path = "D:/cameraman.png"

SNR_in_db = -10
SNR = 10**(SNR_in_db/10)
Pn = 1/SNR
variance_ =Pn
mean_ =0
def image_to_transformed_bit_array(image_path):
    """
    Reads an image from the specified path, converts it into a 1D array,
    and then converts each element into individual bits (0s and 1s).
    Finally, transforms 1s to -1 and 0s to +1.

    Parameters:
    image_path (str): The path to the image file.

    Returns:
    list: A list of transformed bits (-1s and +1s) representing the image.
    """
    # Read the image
    image = imageio.imread(image_path)

    # Convert the image to a 1D array
    image_1d = image.flatten()

    # Convert each element to an 8-bit binary string and then to individual bits
    image_1d_bits = [int(bit) for pixel in image_1d for bit in format(pixel, '08b')]

    # Transform 1s to -1 and 0s to +1
    transformed_bits = [-1 if bit == 1 else 1 for bit in image_1d_bits]

    return transformed_bits

def awgn_channel(bit_array, mean, variance, Pn):
    """
    Adds Additive White Gaussian Noise (AWGN) to each bit of the bit array.

    Parameters:
    bit_array (list): The input bit array with values -1 and +1.
    mean (float): The mean of the normal distribution.
    variance (float): The variance of the normal distribution.
    Pn (float): The noise power.

    Returns:
    np.ndarray: The noisy bit array.
    """
    # Convert the bit array to a numpy array
    bit_array = array(bit_array)

    # Generate AWGN noise
    noise = sqrt(Pn / 2) * (random.normal(mean, sqrt(variance), len(bit_array)) + 
                               1j * random.normal(mean, sqrt(variance), len(bit_array)))

    # Add the noise to each bit of the bit array
    noisy_bit_array = bit_array + noise

    return noisy_bit_array

bit_array = image_to_transformed_bit_array(image_path)

print(awgn_channel(bit_array,mean_,variance_,Pn))