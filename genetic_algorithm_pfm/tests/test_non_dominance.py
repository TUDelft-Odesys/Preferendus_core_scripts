"""
See the paper of Kuri & Quezada (1998) for the reference functions and solutions.

Sources:
Kuri, Angel & Quezada, Carlos. (1998). A universal eclectic genetic algorithm for constrained optimization. Proceedings
6th European Congress on Intelligent Techniques & Soft Computing, EUFIT'98.
"""

from numpy.random import rand

from genetic_algorithm_pfm import genetic_algorithm


class Colors:
    OK = '\033[92m'  # GREEN
    WARNING = '\033[93m'  # YELLOW
    FAIL = '\033[91m'  # RED
    RESET = '\033[0m'  # RESET COLOR


def objective(x):
    x1 = x[:, 0]
    x3 = x[:, 2]
    x5 = x[:, 4]

    return 5.3578547 * x3 ** 2 + 0.8356891 * x1 * x5 + 37.293239 * x1 - 40792.141


def cons_1(x):
    x1 = x[:, 0]
    x2 = x[:, 1]
    x3 = x[:, 2]
    x4 = x[:, 3]
    x5 = x[:, 4]

    g_1 = 85.334407 + 0.0056858 * x2 * x5 + 0.00026 * x1 * x4 - 0.0022053 * x3 * x5
    return g_1 - 92


def cons_2(x):
    x1 = x[:, 0]
    x2 = x[:, 1]
    x3 = x[:, 2]
    x4 = x[:, 3]
    x5 = x[:, 4]

    g_1 = 85.334407 + 0.0056858 * x2 * x5 + 0.00026 * x1 * x4 - 0.0022053 * x3 * x5
    return -1 * g_1


def cons_3(x):
    x1 = x[:, 0]
    x2 = x[:, 1]
    x3 = x[:, 2]
    x5 = x[:, 4]

    g_2 = 80.51249 + 0.0071317 * x2 * x5 + 0.0029955 * x1 * x2 + 0.0021813 * x3 ** 2
    return g_2 - 110


def cons_4(x):
    x1 = x[:, 0]
    x2 = x[:, 1]
    x3 = x[:, 2]
    x5 = x[:, 4]

    g_2 = 80.51249 + 0.0071317 * x2 * x5 + 0.0029955 * x1 * x2 + 0.0021813 * x3 ** 2
    return 90 - g_2


def cons_5(x):
    x1 = x[:, 0]
    x3 = x[:, 2]
    x4 = x[:, 3]
    x5 = x[:, 4]

    g_3 = 9.300961 + 0.0047026 * x3 * x5 + 0.0012547 * x1 * x3 + 0.0019085 * x3 * x4
    return g_3 - 25


def cons_6(x):
    x1 = x[:, 0]
    x3 = x[:, 2]
    x4 = x[:, 3]
    x5 = x[:, 4]

    g_3 = 9.300961 + 0.0047026 * x3 * x5 + 0.0012547 * x1 * x3 + 0.0019085 * x3 * x4
    return 20 - g_3


bounds = [[78, 102], [33, 45], [27, 45], [27, 45], [27, 45]]
cons = [['ineq', cons_1], ['ineq', cons_2], ['ineq', cons_3], ['ineq', cons_4], ['ineq', cons_5], ['ineq', cons_6]]


def test_non_dominance():
    options = {
        'n_bits': 16,
        'n_iter': 500,
        'n_pop': 250,
        'r_cross': 0.8,
        'max_stall': 35,
        'tol': 1e-15
    }
    score, decoded= genetic_algorithm(objective=objective, constraints=cons,
                                          bounds=bounds, cons_handler='CND', options=options, tetra=False)
    assert score < -30760

    return score, decoded


if __name__ == '__main__':
    best_score, variables = test_non_dominance()

    dec = rand(2, 5)
    dec[0, :] = variables
    g1 = cons_1(dec)[0] + 92
    g2 = cons_2(dec)[0]
    g3 = cons_3(dec)[0] + 110
    g4 = cons_4(dec)[0] - 90
    g5 = cons_5(dec)[0] + 25
    g6 = cons_6(dec)[0] - 20

    print()
    print(f'The optimal result = {best_score}')
    print(f'Value of constraint 1: {g1}')
    print(f'Value of constraint 2: {g2}')
    print(f'Value of constraint 3: {g3}')
    print(f'Value of constraint 4: {g4}')
    print(f'Value of constraint 5: {g5}')
    print(f'Value of constraint 6: {g6}')
    print(f'For the combination of variables: {variables}')
