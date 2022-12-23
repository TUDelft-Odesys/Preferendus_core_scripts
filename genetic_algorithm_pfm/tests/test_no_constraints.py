"""
Test function without constraints to verify the GA

Source:
        https://en.wikipedia.org/wiki/Test_functions_for_optimization#Test_functions_for_multi-objective_optimization
"""
import matplotlib.pyplot as plt
from numpy import exp, sqrt, cos, pi, e, array
from numpy.testing import assert_allclose

from genetic_algorithm_pfm import GeneticAlgorithm


def test_ackley(verbose=False):
    """Test to see if algorithm is correct. runs automatically via pytest."""

    def test_objective(x):
        """Ackley function. Minimal at x,y = 0,0 for 5 < x,y < 5"""
        x1 = x[:, 0]
        x2 = x[:, 1]
        return -20 * exp(-0.2 * sqrt(0.5 * (x1 ** 2 + x2 ** 2))) - exp(
            0.5 * (cos(2 * pi * x1) + cos(2 * pi * x2))) + e + 20

    bounds = [[-5, 5], [-5, 5]]
    cons = []

    options = {
        'n_bits': 16,
        'n_iter': 400,
        'n_pop': 250,
        'r_cross': 0.9,
        'max_stall': 32,
        'tol': 1e-15,
    }

    sol = list()
    ga = GeneticAlgorithm(objective=test_objective, constraints=cons, bounds=bounds, options=options)
    for _ in range(10):
        score, decoded, _ = ga.run(verbose=verbose)

        assert_allclose(actual=decoded, desired=[0, 0], rtol=1e-4, atol=5e-4, err_msg='Algorithm is broken')
        sol.append(decoded)
    print('Test was successful')

    if verbose:
        x_fill = [bounds[0][0], bounds[0][1], bounds[0][1], bounds[0][0]]
        y_fill = [bounds[1][0], bounds[1][0], bounds[1][1], bounds[1][1]]

        fig, ax = plt.subplots()
        ax.set_xlim((-6, 6))
        ax.set_ylim((-6, 6))
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_title('Optimal solution in solution space')
        ax.fill(x_fill, y_fill, color='#539ecd', label='Solution space')
        ax.scatter([array(sol)[:, 0]], [array(sol)[:, 1]], color='r', label='Solutions algorithm')
        ax.scatter([0], [0], color='k', marker='*', label='Optimal solution')
        fig.legend()

        plt.show()
    return


def test_rastrigin(verbose=False):
    """Test to see if algorithm is correct. Runs automatically via pytest"""
    a = 10
    n = 6

    def test_objective(x):
        """Rastrigin function with six variables"""
        x1 = x[:, 0]
        x2 = x[:, 1]
        x3 = x[:, 2]
        x4 = x[:, 3]
        x5 = x[:, 4]
        x6 = x[:, 5]

        res = a * n + (x1 ** 2 - a * cos(x1)) + (x2 ** 2 - a * cos(x2)) + \
              (x3 ** 2 - a * cos(x3)) + (x4 ** 2 - a * cos(x4)) + (x5 ** 2 - a * cos(x5)) + \
              (x6 ** 2 - a * cos(x6))

        return res

    bounds = [[-5.12, 5.12]] * 6
    cons = []

    options = {
        'n_bits': 16,
        'n_iter': 400,
        'n_pop': 100,
        'r_cross': 0.9,
        'max_stall': 15,
        'tol': 1e-15,
        'sexes_divider': 0.85
    }
    sol = list()
    ga = GeneticAlgorithm(objective=test_objective, constraints=cons, bounds=bounds, options=options)
    for _ in range(10):
        score, decoded, _ = ga.run(verbose=verbose)

        assert_allclose(actual=decoded, desired=[0] * 6, rtol=1e-4, atol=5e-4, err_msg='Algorithm is broken')
        sol.append(decoded)
    print('Test was successful')

    return


if __name__ == '__main__':
    test_ackley()
    test_rastrigin()
