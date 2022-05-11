"""
Credits to Max Driessen and Laurens Visser for the creation of this example.
Code adaptation by Harold van Heukelum, 2022
"""

import matplotlib.pyplot as plt
import numpy as np

from genetic_algorithm_pfm import GeneticAlgorithm
from tetra_pfm import TetraSolver


def objective_municipality(x1, x2):
    return 500000 * x1 - 25000 * (-1 * x2 + 10)


def objective_inhabitants(x1, x2):
    return 60 * 1 / x2 + 1.5 * x1


def objective_operator(x1, x2):
    return 120000 * x1 + 0.1 * 120000 * (x2 - 10) / 10


def objective_project_team(x1):
    return 0.5 * x1


# arrays for plotting continuous preference curves
c1 = np.linspace(1300000, 5250000)
c2 = np.linspace(7.5, 45)
c3 = np.linspace(350000, 1212000)
c4 = np.linspace(1.5, 5)

p1 = c1 / 39500 - 2600 / 79
p2 = 120 - 8 / 3 * c2
p3 = 50500 / 359 - c3 / 8616
p4 = 142.857 - 28.5714 * c4

# import Tetra solver
solver = TetraSolver()

# weights setting, weights can be adjusted between runs to check for outcome changes
w1 = 0.2  # weight of municipality
w2 = 0.4  # weight of users and inhabitants
w3 = 0.35  # weight of light rail operator
w4 = 0.05  # weight of project organisation


def objective(variables):
    """Objective function that calculates three preference scores for all members in the population. These preference
    scores are aggegated by using the Tetra solver.

    :param variables: ndarray of n-by-m, with n the population size of the GA and m the number of variables.
    :return: list with aggregated preference scores (size n-by-1)
    """
    x1 = variables[:, 0]  # number of stops
    x2 = variables[:, 1]  # number of trains

    p_1 = (20 * x1 + x2 - 62) * 50 / 79
    p_2 = 120 - 4 * x1 - 160 / x2
    p_3 = (-120000 * x1 + 1200 * (x2 - 10)) / 8616 + 50500 / 359
    p_4 = 142.857 - 14.2857 * x1

    return solver.request([w1, w2, w3, w4], [p_1, p_2, p_3, p_4])


# bounds setting for the 2 variables
b1 = [3, 10]  # x1 #
b2 = [2, 20]  # x2 # adjusted from [0,20] to [2,20] because the code cannot divide by 0
bounds = [b1, b2]

if __name__ == '__main__':
    n_runs = 3

    # make dictionary with parameter settings for the GA, no changes have been made here.
    options = {
        'n_bits': 20,
        'n_iter': 400,
        'n_pop': 100,
        'r_cross': 0.9,
        'max_stall': 15,
        'tetra': True,
        'var_type_mixed': ['int', 'real']
    }

    save_array = list()
    ga = GeneticAlgorithm(objective=objective, constraints=[], bounds=bounds, options=options)
    for i in range(n_runs):
        print(f'Initialize run {i + 1}')

        score, decoded, _ = ga.run()

        print(f'Optimal result for x1 = {decoded[0]} stations and x2 = {round(decoded[1], 2)} trains')
        save_array.append([decoded[0], decoded[1], sum(decoded)])

        print(f'Finished run {i + 1}')

    # Create figure that shows the results in the solution space
    x_fill = [3, 10, 10, 3]
    y_fill = [20, 20, 2, 2]

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.set_xlim((0, 12))
    ax.set_ylim((0, 22))
    ax.set_xlabel('x1: Number of stations')
    ax.set_ylabel('x2: Number of trains per hour')
    ax.set_title('Solution space')
    ax.fill_between(x_fill, y_fill, color='#539ecd', label='Solution space')
    ax.scatter(np.array(save_array)[:, 0], np.array(save_array)[:, 1], label='Optimal solutions Tetra')
    ax.scatter([9], [12], label='As-built')
    ax.legend()
    ax.set_yticks(np.arange(0, 23, 2))
    ax.grid(linewidth=0.3)

    # calculate individual preference scores for the results of the GA, to plot them on the preference curves
    variable = np.array(save_array)

    c1_res = objective_municipality(variable[:, 0], variable[:, 1])
    c1_res_actual = objective_municipality(9, 12)

    c2_res = objective_inhabitants(variable[:, 0], variable[:, 1])
    c2_res_actual = objective_inhabitants(9, 12)

    c3_res = objective_operator(variable[:, 0], variable[:, 1])
    c3_res_actual = objective_operator(9, 12)

    c4_res = objective_project_team(variable[:, 0])
    c4_res_actual = objective_project_team(9)

    p1_res = c1_res / 39500 - 2600 / 79
    p2_res = 120 - 8 / 3 * c2_res
    p3_res = 50500 / 359 - c3_res / 8616
    p4_res = 142.857 - 28.5714 * c4_res

    p1_res_actual = c1_res_actual / 39500 - 2600 / 79
    p2_res_actual = 120 - 8 / 3 * c2_res_actual
    p3_res_actual = 50500 / 359 - c3_res_actual / 8616
    p4_res_actual = 142.857 - 28.5714 * c4_res_actual

    # create figure that plots all preference curves and the preference scores of the returned results of the GA
    fig2, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(15, 5))

    ax1.plot(c1, p1, label='Preference curve')
    ax1.scatter(c1_res, p1_res, label='Optimal solutions Tetra')
    ax1.scatter(c1_res_actual, p1_res_actual, label='As-built')
    ax1.set_xlim((1000000, 5500000))
    ax1.set_ylim((0, 100))
    ax1.set_title('Income municipality')
    ax1.set_xlabel('Income [€]')
    ax1.set_ylabel('Preference score')
    ax1.grid()
    fig2.legend()

    ax2.plot(c2, p2)
    ax2.scatter(c2_res, p2_res)
    ax2.scatter(c2_res_actual, p2_res_actual)
    ax2.set_xlim((0, 55))
    ax2.set_ylim((0, 100))
    ax2.set_title('Travel time')
    ax2.set_xlabel('Travel time [min]')
    ax2.set_ylabel('Preference score')
    ax2.grid()

    ax3.plot(c3, p3)
    ax3.scatter(c3_res, p3_res)
    ax3.scatter(c3_res_actual, p3_res_actual)
    ax3.set_xlim((0, 1300000))
    ax3.set_ylim((0, 100))
    ax3.set_title('Operational costs')
    ax3.set_xlabel('Operational costs [€]')
    ax3.set_ylabel('Preference score')
    ax3.grid()

    ax4.plot(c4, p4, )
    ax4.scatter(c4_res, p4_res)
    ax4.scatter(c4_res_actual, p4_res_actual)
    ax4.set_xlim((0, 6))
    ax4.set_ylim((0, 100))
    ax4.set_title('Building time')
    ax4.set_xlabel('Building time [years]')
    ax4.set_ylabel('Preference score')
    ax4.grid()

    plt.show()
