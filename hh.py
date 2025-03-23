import numpy as np

class BCH:
    def _init_(self, n, k):
        self.n = n  # Code length
        self.k = k  # Message length
        self.t = (n - k) // 2  # Error-correcting capability
        self.generator_poly = self.generate_polynomial()

    def generate_polynomial(self):
        """Generate the generator polynomial for the BCH code."""
        g = np.array([1], dtype=int)
        
        # Generating polynomial (simplified, binary polynomial multiplication)
        for _ in range(self.t * 2):
            g = np.polymul(g, [1, 1]) % 2  # Multiply with (x + 1) in binary

        # Ensure the generator polynomial has the correct length
        while len(g) < self.n - self.k + 1:
            g = np.append(g, 0)
        
        return np.trim_zeros(g, 'f')

    def encode(self, message):
        """Encode the message using BCH encoding."""
        if len(message) != self.k:
            raise ValueError(f"Message length should be {self.k} bits.")
        
        # Append zeros for parity bits
        message_poly = np.concatenate((message, np.zeros(self.n - self.k, dtype=int)))

        # Polynomial division to get the remainder
        _, remainder = np.polydiv(message_poly, self.generator_poly)
        remainder = np.mod(remainder, 2)

        # Ensure remainder has the correct length
        parity = np.zeros(self.n - self.k, dtype=int)
        remainder = remainder.astype(int)
        parity[-len(remainder):] = remainder

        # Create the final codeword (message + parity)
        codeword = np.concatenate((message, parity))
        return codeword

    def introduce_error(self, codeword, num_errors=1):
        """Introduce random bit-flip errors in the codeword."""
        if num_errors > self.n:
            raise ValueError("Number of errors exceeds the code length.")
        
        error_positions = np.random.choice(self.n, num_errors, replace=False)
        corrupted = codeword.copy()

        for pos in error_positions:
            corrupted[pos] ^= 1  # Flip the bit

        print(f"Introduced errors at positions: {error_positions}")
        return corrupted

    def decode(self, received):
        """Decode the received codeword using BCH decoding."""
        _, syndrome = np.polydiv(received, self.generator_poly)
        syndrome = np.mod(syndrome, 2)

        print("Syndrome:", syndrome.astype(int))

        # No errors detected
        if np.all(syndrome == 0):
            print("No errors detected.")
            return received[:self.k]

        # Naive error correction (flip bits where syndrome indicates)
        corrected = received.copy()
        corrected_positions = []

        for i in range(len(syndrome)):
            if syndrome[i] == 1:
                pos = len(received) - 1 - i
                if 0 <= pos < self.n:
                    corrected[pos] ^= 1
                    corrected_positions.append(pos)

        if corrected_positions:
            print(f"Error corrected at positions: {corrected_positions}")
        
        return corrected[:self.k]