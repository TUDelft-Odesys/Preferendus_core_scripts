"""
Test function with simple constraint handler to verify the GA.See the paper of Kuri & Quezada (1998) for the referenced
functions and solutions.

Sources:
Kuri, Angel & Quezada, Carlos. (1998). A universal eclectic genetic algorithm for constrained optimization. Proceedings
6th European Congress on Intelligent Techniques & Soft Computing, EUFIT'98.
"""

from matplotlib import pyplot as plt
from numpy import array
from numpy.testing import assert_allclose

from genetic_algorithm_pfm import GeneticAlgorithm


def objective_p2(variables):
    """Objective problem 2 of the reference paper"""
    x = variables[:, 0]
    y = variables[:, 1]
    return -x - y


def constraint_1_p2(variables):
    """First constraint for problem 2 of the reference paper"""
    x = variables[:, 0]
    y = variables[:, 1]
    return y - (2 * x ** 4 - 8 * x ** 3 + 8 * x ** 2 + 2)


def constraint_2_p2(variables):
    """Second constraint for problem 2 of the reference paper"""
    x = variables[:, 0]
    y = variables[:, 1]
    return y - (4 * x ** 4 - 32 * x ** 3 + 88 * x ** 2 - 96 * x + 36)


bounds_p2 = [[0, 3], [0, 4]]
cons_p2 = [['ineq', constraint_1_p2], ['ineq', constraint_2_p2]]


def test_objective_2(verbose=False):
    n = 5
    sol_p2 = list()

    options = {
        'n_bits': 24,
        'n_iter': 400,
        'n_pop': 250,
        'r_cross': 0.9,
        'max_stall': 32,
        'tol': 1e-15,
    }

    ga = GeneticAlgorithm(objective=objective_p2, constraints=cons_p2, bounds=bounds_p2, options=options)
    for _ in range(n):
        score_p2, decoded_p2, _ = ga.run(verbose=False)

        assert_allclose(actual=decoded_p2, desired=[2.3295, 3.1738], atol=0.05, rtol=0.05)
        assert_allclose(actual=score_p2, desired=-5.5079, atol=0.05, rtol=0.05)
        sol_p2.append(decoded_p2)

    if verbose:
        # fill figure
        x_fill_p2 = [bounds_p2[0][0], bounds_p2[0][1], bounds_p2[0][1], bounds_p2[0][0]]
        y_fill_p2 = [bounds_p2[1][0], bounds_p2[1][0], bounds_p2[1][1], bounds_p2[1][1]]

        fig, ax = plt.subplots()
        ax.set_xlim((-1, 4))
        ax.set_ylim((-1, 5))
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_title('Optimal solution in solution space')
        ax.fill(x_fill_p2, y_fill_p2, color='#539ecd', label='Solution space')
        ax.scatter([array(sol_p2)[:, 0]], [array(sol_p2)[:, 1]], color='r', label='Solutions algorithm')
        ax.scatter([2.3295], [3.1738], color='k', marker='*', label='Optimal solution')
        fig.legend()
    return


if __name__ == '__main__':
    test_objective_2(verbose=True)
    plt.show()
