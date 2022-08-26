"""
Test function to see if the combination of the GA and Tetra is still functioning correctly.
"""
import numpy as np
from scipy.interpolate import pchip_interpolate

from genetic_algorithm_pfm import GeneticAlgorithm
from tetra_pfm import TetraSolver

w1 = 0.5
w2 = 0.5

solver = TetraSolver()


def objective(variables):
    x1 = variables[:, 0]
    x2 = variables[:, 1]

    area = x1 * x2
    vector_length = np.sqrt(x1 ** 2 + x2 ** 2)

    min_vl = np.sqrt(4 ** 2 + 2 ** 2)
    max_vl = np.sqrt(8 ** 2 + 10 ** 2)
    mid_vl = np.sqrt(6 ** 2 + 6 ** 2)

    p1 = pchip_interpolate([8, 36, 80], [0, 100, 0], area)
    p2 = pchip_interpolate([min_vl, mid_vl, max_vl], [0, 100, 0], vector_length)

    return solver.request([w1, w2], [p1, p2])


def constraint(variables):
    """First constraint for problem 2 of the reference paper"""
    x1 = variables[:, 0]
    x2 = variables[:, 1]
    return x1 * x2 - 70


bounds = [[4, 8], [2, 10]]
cons = [['ineq', constraint]]


def test_tetra(n=1):
    options = {
        'n_bits': 12,
        'n_iter': 400,
        'n_pop': 100,
        'r_cross': 0.85,
        'max_stall': 15,
        'tetra': True,
        'tetra_method': 1
    }

    ga = GeneticAlgorithm(objective=objective, constraints=cons, bounds=bounds, options=options)
    print('Start test')
    for _ in range(n):
        score, decoded, _ = ga.run(verbose=False)

        np.testing.assert_allclose(actual=decoded, desired=[6, 6], atol=0.005, rtol=0.005)

    print('Test was successful')
    return


if __name__ == '__main__':
    test_tetra(n=3)
