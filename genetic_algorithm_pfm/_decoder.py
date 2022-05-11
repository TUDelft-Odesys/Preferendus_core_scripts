"""
Decoding object for the GA

(c) Harold van Heukelum, 2022
"""


class _Decoding:
    def __init__(self, bounds, n_bits, approach):
        """
        Decode bitstring to numbers.

        For real valued variables:
            The accuracy of the variables is determined by the number of bits. The bitstring represents an integer with
            a max value of 2**n_bits. The bitstring is converted to integer via the build-in function int(). To get to
            floats in the range of the boundaries, the integer value is normalized.

            Note that the accuracy is thus influenced by the number of bits!

        :param bounds: list containing boundaries of variables
        :param n_bits: number of bits per variable
        :return: variables as floats (instead of bits)
        """
        self.bounds = bounds
        self.n_bits = n_bits

        self.largest_value = 2 ** n_bits
        self.approach = approach

    def decode(self, member):
        """
        Decode member of population

        :param member: member of population
        :return: list with decoded variables
        """
        decoded = list()
        for i in range(len(self.bounds)):
            if self.approach[i] == 'real':
                decoded.append(self._decode_real(member[i], self.bounds[i]))
            else:
                decoded.append(member[i])
        return decoded

    def _decode_real(self, substring, bounds):
        """
        Method to go from list of bits to real valued float

        :param substring: list of bits
        :param bounds: lower and upper bound of variable
        :return: real valued float
        """
        # convert bitstring to a string of chars
        chars = ''.join(map(str, substring))

        # convert string to integer
        integer = int(chars, 2)

        # scale integer to desired range
        return bounds[0] + (integer / self.largest_value) * (bounds[1] - bounds[0])
