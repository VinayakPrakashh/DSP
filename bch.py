import numpy as np

class BCH:
    def __init__(self, n, k):
        self.n = n  # Code length
        self.k = k  # Message length
        self.t = (n - k) // 2  # Error-correcting capability
        self.generator_poly = self.generate_polynomial()

    def generate_polynomial(self):
        """
        Generates a simple BCH generator polynomial.
        """
        g = np.array([1], dtype=int)
        for _ in range(1, self.t * 2 + 1):
            g = np.polymul(g, [1, 1]) % 2
        return np.trim_zeros(g, 'f')

    def encode(self, message):
        """
        Encodes the message into a BCH codeword.
        """
        if len(message) != self.k:
            raise ValueError(f"Message length should be {self.k} bits.")

        # Append parity bits (n-k zeros)
        message_poly = np.concatenate((message, np.zeros(self.n - self.k, dtype=int)))

        # Polynomial division to get the remainder (parity bits)
        _, remainder = np.polydiv(message_poly, self.generator_poly)
        remainder = np.mod(remainder, 2)

        # Ensure the remainder has the correct length
        remainder = np.pad(remainder, (self.n - self.k - len(remainder), 0), 'constant')

        # Final codeword = message + parity bits
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

        # If the syndrome is zero, no error
        if np.all(syndrome == 0):
            print("No errors detected.")
            return received[:self.k]

        # If errors are detected (basic correction logic)
        print("Errors detected!")

        # Simple error correction (flips bits at random positions for demo purposes)
        corrected = received.copy()

        # Basic flipping for testing (not Berlekamp-Massey algorithm)
        for i in range(self.t):
            corrected[i] ^= 1  # Flip bits (for demo only)

        return corrected[:self.k]


