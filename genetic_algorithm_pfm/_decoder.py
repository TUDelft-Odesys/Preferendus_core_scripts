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
        ret = bounds[0] + (integer / self.largest_value) * (bounds[1] - bounds[0])
        if ret > 7000:
            print(len(substring))
            print(bounds[0])
            print(integer)
            print(self.largest_value)
            print(bounds[1])
            raise ValueError('X2 > 7000. Je script is kapot!')
        return bounds[0] + (integer / self.largest_value) * (bounds[1] - bounds[0])

    def inverse_decode(self, decoded):
        """

        :param decoded:
        :return:
        """
        bitstring = list()
        for i in range(len(self.bounds)):
            if self.approach[i] == 'real':
                integer = int(((decoded[i] - self.bounds[i][0]) * self.largest_value) / (
                        self.bounds[i][1] - self.bounds[i][0])) - 1
                bits = self.n_bits  # int(max(8, math.log(integer, 2) + 1))
                bitstring.append([1 if integer & (1 << (bits - 1 - n)) else 0 for n in range(bits)])
            else:
                bitstring.append(decoded[i])
        return bitstring


if __name__ == '__main__':
    cls = _Decoding([[0, 3000], [0, 7000]], 16, ['real', 'real'])
    bs = cls.inverse_decode([58, 7000])
    print(bs)
    print(cls.decode(bs))
