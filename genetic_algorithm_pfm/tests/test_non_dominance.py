"""
Test function with non-dominance constraint handler to verify the GA. See the paper of Coello (2000) for the referenced
functions and solutions.

References:
    Coello, C. A. C. (2000). Use of a self-adaptive penalty approach for engineering optimization problems.
    Computers in Industry, 41(2), 113-127.
"""

from pprint import pprint

import numpy as np

from genetic_algorithm_pfm import GeneticAlgorithm


class Colors:
    """Class to allow for printing in color on the console"""
    OK = '\033[92m'  # GREEN
    WARNING = '\033[93m'  # YELLOW
    FAIL = '\033[91m'  # RED
    RESET = '\033[0m'  # RESET COLOR


def objective(x):
    """Objective from example 4 of referenced paper"""
    x1 = x[:, 0]
    x3 = x[:, 2]
    x5 = x[:, 4]

    return 5.3578547 * x3 ** 2 + 0.8356891 * x1 * x5 + 37.293239 * x1 - 40792.141


def cons_1(x):
    """First constraint from example 4 of referenced paper"""
    x1 = x[:, 0]
    x2 = x[:, 1]
    x3 = x[:, 2]
    x4 = x[:, 3]
    x5 = x[:, 4]

    g_1 = 85.334407 + 0.0056858 * x2 * x5 + 0.00026 * x1 * x4 - 0.0022053 * x3 * x5
    return g_1 - 92


def cons_2(x):
    """Second constraint from example 4 of referenced paper"""
    x1 = x[:, 0]
    x2 = x[:, 1]
    x3 = x[:, 2]
    x4 = x[:, 3]
    x5 = x[:, 4]

    g_1 = 85.334407 + 0.0056858 * x2 * x5 + 0.00026 * x1 * x4 - 0.0022053 * x3 * x5
    return -1 * g_1


def cons_3(x):
    """Third constraint from example 4 of referenced paper"""
    x1 = x[:, 0]
    x2 = x[:, 1]
    x3 = x[:, 2]
    x5 = x[:, 4]

    g_2 = 80.51249 + 0.0071317 * x2 * x5 + 0.0029955 * x1 * x2 + 0.0021813 * x3 ** 2
    return g_2 - 110


def cons_4(x):
    """Fourth constraint from example 4 of referenced paper"""
    x1 = x[:, 0]
    x2 = x[:, 1]
    x3 = x[:, 2]
    x5 = x[:, 4]

    g_2 = 80.51249 + 0.0071317 * x2 * x5 + 0.0029955 * x1 * x2 + 0.0021813 * x3 ** 2
    return 90 - g_2


def cons_5(x):
    """Fifth constraint from example 4 of referenced paper"""
    x1 = x[:, 0]
    x3 = x[:, 2]
    x4 = x[:, 3]
    x5 = x[:, 4]

    g_3 = 9.300961 + 0.0047026 * x3 * x5 + 0.0012547 * x1 * x3 + 0.0019085 * x3 * x4
    return g_3 - 25


def cons_6(x):
    """Sixth constraint from example 4 of referenced paper"""
    x1 = x[:, 0]
    x3 = x[:, 2]
    x4 = x[:, 3]
    x5 = x[:, 4]

    g_3 = 9.300961 + 0.0047026 * x3 * x5 + 0.0012547 * x1 * x3 + 0.0019085 * x3 * x4
    return 20 - g_3


bounds = [[78, 102], [33, 45], [27, 45], [27, 45], [27, 45]]
cons = [['ineq', cons_1], ['ineq', cons_2], ['ineq', cons_3], ['ineq', cons_4], ['ineq', cons_5], ['ineq', cons_6]]


def test_non_dominance(verbose=False):
    """Function to run the test automatically via pytest"""
    options = {
        'n_bits': 8,
        'n_iter': 400,
        'n_pop': 150,
        'r_cross': 0.9,
        'max_stall': 200,
        'tol': 1e-15,
        'elitism percentage': 20
    }
    ga = GeneticAlgorithm(objective=objective, constraints=cons, bounds=bounds, cons_handler='CND', options=options)
    save_array = list()
    for _ in range(4):
        score, decoded, _ = ga.run(verbose=verbose)
        assert score < -30710.00
        save_array.append(score)

    worst = max(save_array)
    best = min(save_array)
    mean = np.mean(save_array)
    std = np.std(save_array)

    if verbose:
        pprint(save_array)
        print()
        print(worst)
        print(best)
        print(mean)
        print(std)

    assert std < 100.00

    return


if __name__ == '__main__':
    test_non_dominance(verbose=True)

