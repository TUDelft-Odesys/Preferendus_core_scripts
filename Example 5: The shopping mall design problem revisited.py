"""Example from Preference-based optimization reader. Credits to Dmitry Zhilyaev for creating the example

Code adapted by Harold van Heukelum
"""
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as ss
from scipy.interpolate import pchip_interpolate

from genetic_algorithm_pfm import GeneticAlgorithm
from tetra_pfm import TetraSolver
from weighted_minmax.algorithm import aggregate_max

# create arrays for plotting continuous preference curves
c1 = np.linspace(0, 1200000)
c2 = np.linspace(0, 750000)
c3 = np.linspace(0, 360000)

# p1 = 1 / 9600 * (c1 - 240000)
p1 = pchip_interpolate([0, 400000, 1200000], [0, 50, 100], c1)
p2 = 100 - (1 / 6600) * (c2 - 90000)
p3 = (1 / 3150) * (c3 - 45000)

# import Tetra solver
solver = TetraSolver()

# set weights
w1 = 1 / 3
w2 = 1 / 3
w3 = 1 / 3


def objective_p1(x1, x2):
    return pchip_interpolate([0, 400000, 1400000], [0, 50, 100], (160 * x1 + 80 * x2))
    # return 1 / 9600 * ((160 * x1 + 80 * x2) - 240000)


def objective_p2(x1, x2):
    return 100 - (1 / 6600) * (120 * x1 + 30 * x2 - 90000)


def objective_p3(x1, x2):
    return (1 / 3150) * (15 * x1 + 45 * x2 - 45000)


def objective(variables, method='tetra'):
    x1 = variables[:, 0]
    x2 = variables[:, 1]
    p_1 = objective_p1(x1, x2)
    p_2 = objective_p2(x1, x2)
    p_3 = objective_p3(x1, x2)

    if method == 'minmax':
        return aggregate_max([w1, w2, w3], [p_1, p_2, p_3], 100)
    else:
        return solver.request([w1, w2, w3], [p_1, p_2, p_3])


x_array = np.array([
    [3000, 0],
    [5000, 0],
    [5000, 5000],
    [3000, 7000],
    [0, 7000],
    [0, 3000]
])

results_p1 = objective_p1(x_array[:, 0], x_array[:, 1])
results_p2 = objective_p2(x_array[:, 0], x_array[:, 1])
results_p3 = objective_p3(x_array[:, 0], x_array[:, 1])

alternatives = ['X1 = 3,000; X2= 0',
                'X1 = 5,000; X2= 0',
                'X1 = 5,000; X2= 5,000',
                'X1 = 3,000; X2= 7,000',
                'X1 = 0; X2= 7,000',
                'X1 = 0; X2= 3,000',
                ]

data = {'Alternatives': alternatives, 'P1': np.round_(results_p1, 2), 'P2': np.round_(results_p2),
        'P3': np.round_(results_p3)}
df = pd.DataFrame(data)
print(df)
print()

aggregation_results = objective(x_array)
data = {'Alternatives': alternatives, 'rank': ss.rankdata(aggregation_results, method='min'),
        'Aggregated scores': np.round_(np.multiply(aggregation_results, -1), 2)}

df = pd.DataFrame(data)
print(df)
print()

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
        'n_pop': 250,
        'r_cross': 0.9,
        'max_stall': 15,
        'tetra': True,
        'var_type_mixed': ['real', 'real']
    }

    # run the GA several times, print results to console, and save results to allow for plotting
    save_array = list()
    ga = GeneticAlgorithm(objective=objective, constraints=cons, bounds=bounds, options=options)
    for i in range(1):
        print(f'Initialize run {i + 1}')
        score, decoded, _ = ga.run()

        print(f'Optimal result for x1 = {round(decoded[0], 2)}m2 and x2 = {round(decoded[1], 2)}m2 '
              f'(sum = {round(sum(decoded))}m2)')

        save_array.append([decoded[0], decoded[1], sum(decoded)])
        print(f'Finished run {i + 1}')

    # make dictionary with parameter settings for the GA
    options = {
        'n_bits': 12,
        'n_iter': 400,
        'n_pop': 150,
        'r_cross': 0.9,
        'max_stall': 15,
        'tetra': False,
        'var_type_mixed': ['real', 'real']
    }

    # run the GA several times, print results to console, and save results to allow for plotting
    save_array_mm = list()
    ga = GeneticAlgorithm(objective=objective, constraints=cons, bounds=bounds, options=options, args=('minmax',))
    for i in range(1):
        print(f'Initialize run {i + 1}')
        score, decoded, _ = ga.run()

        print(f'Optimal result for x1 = {round(decoded[0], 2)}m2 and x2 = {round(decoded[1], 2)}m2 '
              f'(sum = {round(sum(decoded))}m2)')

        save_array_mm.append([decoded[0], decoded[1], sum(decoded)])
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
    ax.scatter(np.array(save_array)[:, 0], np.array(save_array)[:, 1], label='Optimal solution')
    ax.scatter(np.array(save_array_mm)[:, 0], np.array(save_array_mm)[:, 1], label='Optimal solution MinMax')
    ax.grid()
    fig.legend()

    # calculate individual preference scores for the results of the GA, to plot them on the preference curves
    variable = np.array(save_array)
    variable_mm = np.array(save_array_mm)

    c1_res = (160 * variable[:, 0] + 80 * variable[:, 1])
    p1_res = 1 / 9600 * (c1_res - 240000)

    c2_res = (120 * variable[:, 0] + 30 * variable[:, 1])
    p2_res = 100 - (1 / 6600) * (c2_res - 90000)

    c3_res = (15 * variable[:, 0] + 45 * variable[:, 1])
    p3_res = (1 / 3150) * (c3_res - 45000)

    c1_res_mm = (160 * variable_mm[:, 0] + 80 * variable_mm[:, 1])
    p1_res_mm = 1 / 9600 * (c1_res_mm - 240000)

    c2_res_mm = (120 * variable_mm[:, 0] + 30 * variable_mm[:, 1])
    p2_res_mm = 100 - (1 / 6600) * (c2_res_mm - 90000)

    c3_res_mm = (15 * variable_mm[:, 0] + 45 * variable_mm[:, 1])
    p3_res_mm = (1 / 3150) * (c3_res_mm - 45000)

    # create figure that plots all preference curves and the preference scores of the returned results of the GA
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle('Preference functions')

    ax1.plot(c1, p1)
    ax1.scatter(c1_res, p1_res, label='Tetra')
    ax1.scatter(c1_res_mm, p1_res_mm, label='MinMax')
    ax1.set_xlim((0, 1200000))
    ax1.set_ylim((0, 100))
    ax1.set_title('Profit')
    ax1.set_xlabel('Profit [€]')
    ax1.set_ylabel('Preference score')
    ax1.grid()
    ax1.legend()

    ax2.plot(c2, p2)
    ax2.scatter(c2_res, p2_res)
    ax2.scatter(c2_res_mm, p2_res_mm)
    ax2.set_xlim((0, 750000))
    ax2.set_ylim((0, 100))
    ax2.set_title('CO2 Emission')
    ax2.set_xlabel('Emissions [kg]')
    ax2.set_ylabel('Preference score')
    ax2.grid()

    ax3.plot(c3, p3)
    ax3.scatter(c3_res, p3_res)
    ax3.scatter(c3_res_mm, p3_res_mm)
    ax3.set_xlim((0, 360000))
    ax3.set_ylim((0, 100))
    ax3.set_title('Shopping potential')
    ax3.set_xlabel('Shopping potential [people]')
    ax3.set_ylabel('Preference score')
    ax3.grid()

    plt.show()
