from numpy import *

# Function to take complex number input from user
def input_complex_array(prompt):
    input_str = input(prompt)
    return array([complex(num) for num in input_str.split()], dtype=complex)

# Taking user input for xn and hn
xn = input_complex_array("Enter the complex numbers for xn separated by spaces: ")
hn = input_complex_array("Enter the complex numbers for hn separated by spaces: ")
N = int(input("Enter the value of N: "))

M = len(hn)
L = N - M + 1

def circular_conv(xn, hn):
    N1 = len(xn)
    N2 = len(hn)
    N = max([N1, N2])
    xn = pad(xn, (0, (N - N1)), 'constant', constant_values=(0,))
    hn = pad(hn, (0, (N - N2)), 'constant', constant_values=(0,))
    result = zeros([N, N], dtype=complex)
    for i in range(N):
        result[:, i] = roll(xn, i)
    return dot(result, hn)

def overlap_add(xn, hn, M, L):
    padded_h1 = pad(hn, (0, L - 1), 'constant', constant_values=(0,))
    splitted_x1 = [xn[i:i + L] for i in range(0, len(xn), L)]
    padded_x1 = [list(i) + [0] * (M - 1) for i in splitted_x1]
    result_length = len(xn) + len(padded_h1) - 1
    result = zeros(result_length, dtype=complex)
    for i, sublist in enumerate(padded_x1):
        conv_res = circular_conv(sublist, padded_h1)
        start_index = i * L
        end_index = start_index + len(conv_res)
        result[start_index:end_index] += conv_res
    print(result[:len(result) - L + 1])

overlap_add(xn, hn, M, L)