"""
Test function with simple constraint handler to verify the GA.See the paper of Kuri & Quezada (1998) for the referenced
functions and solutions.

Sources:
Kuri, Angel & Quezada, Carlos. (1998). A universal eclectic genetic algorithm for constrained optimization. Proceedings
6th European Congress on Intelligent Techniques & Soft Computing, EUFIT'98.
"""
import numpy as np
from scipy.interpolate import pchip_interpolate

from genetic_algorithm_pfm import GeneticAlgorithm

w1 = 0.5
w2 = 0.5


def objective(variables):
    """Objective problem 2 of the reference paper"""
    x1 = variables[:, 0]
    x2 = variables[:, 1]

    area = x1 * x2
    vector_length = np.sqrt(x1 ** 2 + x2 ** 2)

    min_vl = np.sqrt(4 ** 2 + 2 ** 2)
    max_vl = np.sqrt(8 ** 2 + 10 ** 2)
    mid_vl = np.sqrt(6 ** 2 + 6 ** 2)

    p1 = pchip_interpolate([8, 36, 80], [0, 100, 0], area)
    p2 = pchip_interpolate([min_vl, mid_vl, max_vl], [0, 100, 0], vector_length)

    return [w1, w2], [p1, p2]


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
        'n_pop': 250,
        'r_cross': 0.9,
        'max_stall': 15,
        'tetra': True,
        'aggregation': 'tetra'
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
