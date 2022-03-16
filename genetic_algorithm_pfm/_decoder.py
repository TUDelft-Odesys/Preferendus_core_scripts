"""Function to go from bitstring to floats

Copyright (c) Harold van Heukelum, 2021
"""


def _decode(bounds, n_bits, bitstring):
    """
    Decode bitstring to numbers.

    The accuracy of the variables is determined by the number of bits. The bitstring represents an integer with the max
    value of 2**n_bits. The bitstring is converted to integer via the build-in function int(). To get to a float in the
    range of the boundaries, the integer value is scaled by using the fraction integer / largest value and the given
    boundary conditions.

    Note that the accuracy is thus influenced by the number of bits!

    :param bounds: list containing boundaries of variables
    :param n_bits: number of bits per variable
    :param bitstring: list of len(boundaries) * n_bits that make up the population
    :return: variables as floats (instead of bits)
    """
    decoded = []
    largest_value = 2 ** n_bits
    for i in range(len(bounds)):
        # extract the substring from the bitstring
        start, end = i * n_bits, (i * n_bits) + n_bits
        try:
            substring = bitstring[start:end]
        except TypeError as err:
            print(bitstring)
            raise err

        # convert bitstring to a string of chars
        chars = ''.join(map(str, substring))

        # convert string to integer
        integer = int(chars, 2)

        # scale integer to desired range and store to list
        try:
            value = bounds[i][0] + (integer / largest_value) * (bounds[i][1] - bounds[i][0])
        except TypeError:
            value = integer + 1
        decoded.append(value)
    return decoded
