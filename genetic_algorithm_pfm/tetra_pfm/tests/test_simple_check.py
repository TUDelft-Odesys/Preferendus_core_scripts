"""Copyright (c) 2022. Harold Van Heukelum"""

import pathlib

from scipy.interpolate import pchip_interpolate

from genetic_algorithm_pfm.tetra_pfm import TetraSolver

solver = TetraSolver()

HERE = pathlib.Path(__file__).parent


def test_check_new_tetra():
    """
    Test to see if the TetraSolver is accurate up to 3 digits. The comparison is made to the 'verified' result of
    the Matlab code. Trows AssertionError when the result is not the same.

    :return: None
    """

    a = [1]

    p1 = pchip_interpolate([0, 5, 10], [0, 50, 100], a)
    p2 = pchip_interpolate([0, 5, 10], [100, 50, 0], a)

    w = [0.5, 0.5]
    ret = solver.request(w=w, p=[p1, p2])
    assert round(ret[0], 4) == -50.
    return


if __name__ == '__main__':
    test_check_new_tetra()
