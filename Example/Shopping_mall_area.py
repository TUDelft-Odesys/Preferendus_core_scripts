"""Example from Preference-based optimization reader. Credits to Dmitry Zhilyaev for creating the example

Code adapted by Harold van Heukelum
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import pchip_interpolate

from genetic_algorithm_pfm import genetic_algorithm
from tetra_pfm import TetraSolver

# create arrays for plotting continuous preference curves
c1 = np.linspace(0, 1200000)
c2 = np.linspace(0, 750000)
c3 = np.linspace(0, 360000)

p1 = pchip_interpolate([0, 400000, 1200000], [0, 50, 100], c1)
p2 = 100 - (1 / 7500) * c2
p3 = (1 / 3600) * c3

# import Tetra solver
solver = TetraSolver()

# set weights
w1 = 0.6
w2 = 0.2
w3 = 0.2


def objective(variables):
    """Objective function that calculates three preference scores for all members in the population. These preference
    scores are aggegated by using the Tetra solver.

    :param variables: ndarray of n-by-m, with n the population size of the GA and m the number of variables.
    :return: list with aggregated preference scores (size n-by-1)
    """
    x1 = variables[:, 0]
    x2 = variables[:, 1]
    p_1 = pchip_interpolate([0, 400000, 1200000], [0, 50, 100], (160 * x1 + 80 * x2))
    p_2 = 100 - (1 / 7500) * (120 * x1 + 30 * x2)
    p_3 = (1 / 3600) * (15 * x1 + 45 * x2)
    return solver.request([w1, w2, w3], [p_1, p_2, p_3])


# set bounds for all variables
b1 = [0, 5000]  # x1
b2 = [0, 7000]  # x2
bounds = [b1, b2]


def constraint_1(variables):
    """Constraint that checks if the sum of the areas x1 and x2 is not higher than 10,000 m2.

    :param variables: ndarray of n-by-m, with n the population size of the GA and m the number of variables.
    :return: list with scores of the constraint
    """
    x1 = variables[:, 0]
    x2 = variables[:, 1]

    return x1 + x2 - 10000  # < 0


def constraint_2(variables):
    """Constraint that checks if the sum of the areas x1 and x2 is not lower than 3,000 m2.

    :param variables: ndarray of n-by-m, with n the population size of the GA and m the number of variables.
    :return: list with scores of the constraint
    """
    x1 = variables[:, 0]
    x2 = variables[:, 1]

    return 3000 - (x1 + x2)  # < 0


# define list with constraints
cons = [['ineq', constraint_1], ['ineq', constraint_2]]

if __name__ == '__main__':

    # make dictionary with parameter settings for the GA
    options = {
        'n_bits': 12,
        'n_iter': 400,
        'n_pop': 750,
        'r_cross': 0.9,
        'max_stall': 15,
        'tol': 1e-15
    }

    # run the GA several times, print results to console, and save results to allow for plotting
    save_array = list()
    for i in range(3):
        print(f'Initialize run {i + 1}')
        score, decoded = genetic_algorithm(objective=objective, constraints=cons, bounds=bounds, options=options)

        print(f'Optimal result for x1 = {round(decoded[0], 2)}m2 and x2 = {round(decoded[1], 2)}m2 '
              f'(sum = {round(sum(decoded))}m2)')

        save_array.append([decoded[0], decoded[1], sum(decoded)])
        print(f'Finished run {i + 1}')

    # create figure that shows the results in the solution space
    x_fill = [0, 3000, 5000, 5000, 3000, 0]
    y_fill = [7000, 7000, 5000, 0, 0, 3000]

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.set_xlim((0, 9000))
    ax.set_ylim((0, 9000))
    ax.set_xlabel('x1 [m2]')
    ax.set_ylabel('x2 [m2]')
    ax.set_title('Solution space')
    ax.fill_between(x_fill, y_fill, color='#539ecd', label='Solution space')
    ax.scatter(np.array(save_array)[:, 0], np.array(save_array)[:, 1], label='Optimal solutions')
    fig.legend()

    # calculate individual preference scores for the results of the GA, to plot them on the preference curves
    variable = np.array(save_array)

    c1_res = (160 * variable[:, 0] + 80 * variable[:, 1])
    p1_res = pchip_interpolate([0, 400000, 1200000], [0, 50, 100], c1_res)

    c2_res = (120 * variable[:, 0] + 30 * variable[:, 1])
    p2_res = 100 - (1 / 7500) * c2_res

    c3_res = (15 * variable[:, 0] + 45 * variable[:, 1])
    p3_res = (1 / 3600) * c3_res

    # create figure that plots all preference curves and the preference scores of the returned results of the GA
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle('Preference functions')

    ax1.plot(c1, p1)
    ax1.scatter(c1_res, p1_res, label='Optimal solutions')
    ax1.set_xlim((0, 1200000))
    ax1.set_ylim((0, 100))
    ax1.set_title('Profit')
    ax1.set_xlabel('Profit [€]')
    ax1.set_ylabel('Preference score')

    ax2.plot(c2, p2)
    ax2.scatter(c2_res, p2_res, label='Optimal solutions')
    ax2.set_xlim((0, 750000))
    ax2.set_ylim((0, 100))
    ax2.set_title('CO2 Emission')
    ax2.set_xlabel('Emissions [kg]')
    ax2.set_ylabel('Preference score')

    ax3.plot(c3, p3)
    ax3.scatter(c3_res, p3_res, label='Optimal solutions')
    ax3.set_xlim((0, 360000))
    ax3.set_ylim((0, 100))
    ax3.set_title('Shopping potential')
    ax3.set_xlabel('Shopping potential [people]')
    ax3.set_ylabel('Preference score')

    plt.show()
