import numpy as np

class BCH:
    def __init__(self, n, k):
        """
        Initializes the BCH code parameters.
        """
        self.n = n                  # Codeword length
        self.k = k                  # Message length
        self.t = (n - k) // 2       # Error-correcting capability
        self.generator_poly = self.generate_polynomial()

    def generate_polynomial(self):
        """
        Generates a BCH generator polynomial.
        """
        # Basic BCH generator polynomial (example for demo purposes)
        # In a real BCH, you need Galois field operations
        g = np.array([1], dtype=int)
        for _ in range(1, self.t * 2 + 1):
            g = np.polymul(g, [1, 1]) % 2
        return np.trim_zeros(g, 'f')

    def encode(self, message):
        """
        Encodes the message into a BCH codeword.
        """
        if len(message) != self.k:
            raise ValueError(f"Message length must be {self.k} bits.")

        # Append parity bits (n-k zeros)
        message_poly = np.concatenate((message, np.zeros(self.n - self.k, dtype=int)))

        # Polynomial division (for parity calculation)
        quotient, remainder = np.polydiv(message_poly, self.generator_poly)
        remainder = np.mod(remainder, 2)

        # Ensure remainder has the correct length
        remainder = np.pad(remainder, (self.n - len(remainder), 0), 'constant')

        # Generate the final codeword
        codeword = (message_poly + remainder) % 2
        return codeword.astype(int)

    def decode(self, received):
        """
        Decodes the received codeword, detects, and corrects errors.
        """
        if len(received) != self.n:
            raise ValueError(f"Received codeword should be {self.n} bits.")

        # Syndrome calculation
        _, syndrome = np.polydiv(received, self.generator_poly)
        syndrome = np.mod(syndrome, 2)

        print("\nSyndrome:", syndrome)

        if np.all(syndrome == 0):
            print("No errors detected.")
            return received[:self.k]

        # Error correction using Berlekamp-Massey algorithm
        print("Errors detected!")
        error_locator_poly = self.berlekamp_massey(syndrome)
        error_positions = self.find_error_positions(error_locator_poly)

        corrected = received.copy()
        for pos in error_positions:
            corrected[pos] ^= 1

        return corrected[:self.k]

    def berlekamp_massey(self, syndrome):
        """
        Berlekamp-Massey algorithm to find the error locator polynomial.
        """
        n = len(syndrome)
        c = np.zeros(n, dtype=int)
        b = np.zeros(n, dtype=int)
        c[0] = 1
        b[0] = 1
        l = 0
        m = 1
        for i in range(n):
            discrepancy = syndrome[i]
            for j in range(1, l + 1):
                discrepancy ^= c[j] & syndrome[i - j]
            if discrepancy != 0:
                t = c.copy()
                for j in range(i + 1, n):
                    c[j] ^= b[j - m]
                if 2 * l <= i:
                    l = i + 1 - l
                    b = t
                    m = i + 1
            b = np.roll(b, 1)
        return c[:l + 1]

    def find_error_positions(self, error_locator_poly):
        """
        Find error positions using the error locator polynomial.
        """
        error_positions = []
        for i in range(self.n):
            if np.polyval(error_locator_poly[::-1], 2 ** i) == 0:
                error_positions.append(self.n - 1 - i)
        return error_positions

# Example Usage
bch = BCH(15, 7)  # BCH(15, 7) with 2-bit error correction

# Example message
message = np.array([1, 0, 1, 1, 0, 0, 1])
print("\nOriginal Message:", message)

# Encoding the message
codeword = bch.encode(message)
print("\nEncoded Codeword:", codeword)

# Introduce an error
codeword_with_error = codeword.copy()
codeword_with_error[2] ^= 1  # Flip a bit to simulate error
print("\nReceived Codeword with Error:", codeword_with_error)

# Decode the received message
decoded = bch.decode(codeword_with_error)
print("\nDecoded Message:", decoded)