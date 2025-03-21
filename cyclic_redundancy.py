data = 101010
Divisor = 1101

def xor_fixed(a, b):
    result = []
    min_len = min(len(a), len(b))
    
    for i in range(min_len):  
        result.append('0' if a[i] == b[i] else '1')
    
    # Append remaining bits if the strings have different lengths
    if len(a) > len(b):
        result.extend(a[min_len:])
    elif len(b) > len(a):
        result.extend(b[min_len:])
    
    return ''.join(result)

def mod2_div(dividend, divisor):
    # Append zeros to the dividend (divisor length - 1)
    dividend += '0' * (len(divisor) - 1)

    n = len(divisor)
    temp = dividend[:n]


    while n < len(dividend):

        if temp[0] == '1':
            temp = xor_fixed(temp, divisor)
        else:
            temp = xor_fixed(temp, '0' * len(divisor))

        temp = temp[1:] + dividend[n]
        n += 1

    # Final XOR step
    if temp[0] == '1':
        temp = xor_fixed(temp, divisor)
    else:
        temp = xor_fixed(temp, '0' * len(divisor))

    # Extract the remainder
    remainder = temp[-(len(divisor) - 1):]

    return remainder

def encodeData(data, key):
    remainder = mod2_div(data, key)
    codeword = data + remainder
    print("Encoded Data (Data + CRC):", codeword)
    return codeword

def decodeData(data, key):
    remainder = mod2_div(data, key)
    print("Remainder after decoding:", remainder)
    if remainder == '0' * (len(key) - 1):
        print("No error detected in received data.")
    else:
        print("Error detected in received data.")

data = "101010"
Divisor = "1101"
print("Original Data:", data)
encoded_data = encodeData(data, Divisor)

# Simulate transmission (no error introduced)
received_data = encoded_data
print("Received Data:", received_data)

decodeData(received_data, Divisor)