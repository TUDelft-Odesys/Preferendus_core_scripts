"""Credits to Harold van Heukelum for creating the code and the model"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as ss
from scipy.interpolate import pchip_interpolate

from genetic_algorithm_pfm import GeneticAlgorithm
from tetra_pfm import TetraSolver
from weighted_minmax import aggregate_max


# global information about this model
# The P1 and C1 functions regard the municipality, and it's preference function --> tax income
# The P2 and C2 functions regard the users and its preference function --> travel time
# The P3 and C3 functions regard the light rail operator, and it's preference function --> maintenance costs
# The P4 and C4 functions regard the project organisation, and it's preference function --> building time

def objective_municipality(x1, x2):
    """

    :param x1:
    :param x2:
    :return:
    """
    return 500000 * x1 - 25000 * (-1 * x2 + 10)


def objective_inhabitants(x1, x2):
    """

    :param x1:
    :param x2:
    :return:
    """
    return 60 * 1 / x2 + 1.5 * x1


def objective_operator(x1, x2):
    """

    :param x1:
    :param x2:
    :return:
    """
    return 120000 * x1 + 0.1 * 120000 * (x2 - 10) / 10


def objective_project_team(x1):
    """

    :param x1:
    :param x2:
    :return:
    """
    return 0.5 * x1


# arrays for plotting continuous preference curves
c1 = np.linspace(1300000, 5250000)
c2 = np.linspace(7.5, 45)
c3 = np.linspace(350400, 1212000)
c4 = np.linspace(1.5, 5)

p1_min, p1_mid, p1_max = [1300000, 4000000, 5250000]
p2_min, p2_mid1, p2_mid2, p2_max = [7.5, 20, 35, 45]
p3_min, p3_mid, p3_max = [350400, 750000, 1212000]
p4_min, p4_mid, p4_max = [1.5, 2, 5]

p1 = pchip_interpolate([p1_min, p1_mid, p1_max], [0, 100, 60], c1)
# the preference function for the income of the municipality

p2 = pchip_interpolate([p2_min, p2_mid1, p2_mid2, p2_max], [100, 80, 10, 0], c2)
# the preference function for the travel time for the users

p3 = pchip_interpolate([p3_min, p3_mid, p3_max], [60, 100, 0], c3)
# the preference function for the maintenance costs

p4 = pchip_interpolate([p4_min, p4_mid, p4_max], [100, 95, 0], c4)
# the preference function for the building time

# import Tetra solver
solver = TetraSolver()

# weights setting, weights can be adjusted between runs to check for outcome changes
w1 = 0.2  # weight of municipality
w2 = 0.4  # weight of users and inhabitants
w3 = 0.3  # weight of light rail operator
w4 = 0.1  # weight of project organisation


def objective(variables, method='tetra'):
    """Objective function that calculates three preference scores for all members in the population. These preference
    scores are aggegated by using the Tetra solver.

    :param variables: ndarray of n-by-m, with n the population size of the GA and m the number of variables.
    :param method:
    :return: list with aggregated preference scores (size n-by-1)
    """
    x1 = variables[:, 0]  # number of stops
    x2 = variables[:, 1]  # number of trains

    p_1 = pchip_interpolate([p1_min, p1_mid, p1_max], [0, 100, 60], objective_municipality(x1, x2))
    p_2 = pchip_interpolate([p2_min, p2_mid1, p2_mid2, p2_max], [100, 80, 10, 0], objective_inhabitants(x1, x2))
    p_3 = pchip_interpolate([p3_min, p3_mid, p3_max], [60, 100, 0], objective_operator(x1, x2))
    p_4 = pchip_interpolate([p4_min, p4_mid, p4_max], [100, 95, 0], objective_project_team(x1))

    if method == 'minmax':
        ret = aggregate_max([w1, w2, w3, w4], [p_1, p_2, p_3, p_4], 100)
    else:
        ret = solver.request([w1, w2, w3, w4], [p_1, p_2, p_3, p_4])
    return ret


x_array = np.array([
    [3, 2],
    [3, 20],
    [10, 2],
    [10, 20]
])

results_p1 = pchip_interpolate([p1_min, p1_mid, p1_max], [0, 100, 60],
                               objective_municipality(x_array[:, 0], x_array[:, 1]))
results_p2 = pchip_interpolate([p2_min, p2_mid1, p2_mid2, p2_max], [100, 80, 10, 0],
                               objective_inhabitants(x_array[:, 0], x_array[:, 1]))
results_p3 = pchip_interpolate([p3_min, p3_mid, p3_max], [60, 100, 0], objective_operator(x_array[:, 0], x_array[:, 1]))
results_p4 = pchip_interpolate([p4_min, p4_mid, p4_max], [100, 95, 0], objective_project_team(x_array[:, 0]))

alternatives = ['X1 = 3; X2= 2',
                'X1 = 3; X2= 20',
                'X1 = 10; X2= 2',
                'X1 = 10; X2= 20'
                ]

data = {'Alternatives': alternatives, 'P1': np.round_(results_p1, 2), 'P2': np.round_(results_p2),
        'P3': np.round_(results_p3), 'P4': np.round_(results_p3)}
df = pd.DataFrame(data)
print(df)
print()

aggregation_results = objective(x_array)
data = {'Alternatives': alternatives, 'rank': ss.rankdata(aggregation_results, method='min'),
        'Aggregated scores': np.round_(np.multiply(aggregation_results, -1), 2)}

df = pd.DataFrame(data)
print(df)
print()

# bounds setting for the 2 variables
b1 = [3, 10]  # x1
b2 = [2, 20]  # x2
bounds = [b1, b2]

if __name__ == '__main__':
    n_runs = 1

    # make dictionary with parameter settings for the GA, no changes have been made here.
    print('Run Tetra')
    options = {
        'n_bits': 20,
        'n_iter': 400,
        'n_pop': 500,
        'r_cross': 0.8,
        'max_stall': 10,
        'tetra': True,
        'method_tetra': 1,
        'var_type_mixed': ['int', 'real']
    }

    save_array = list()
    ga = GeneticAlgorithm(objective=objective, constraints=[], bounds=bounds, options=options, args=('tetra',))
    for i in range(n_runs):
        print(f'Initialize run {i + 1}')
        score, decoded, plot_array = ga.run()
        print(f'Optimal result for x1 = {decoded[0]} stations and x2 = {round(decoded[1], 2)} trains')
        save_array.append([decoded[0], decoded[1], sum(decoded)])
        print(f'Finished run {i + 1}')

    # # make dictionary with parameter settings for the GA, no changes have been made here.
    # print('Run MinMax')
    # options = {
    #     'n_bits': 20,
    #     'n_iter': 400,
    #     'n_pop': 500,
    #     'r_cross': 0.9,
    #     'max_stall': 15,
    #     'var_type_mixed': ['int', 'real'],
    #     'tetra': False
    # }

    # save_array_mm = list()
    # ga = GeneticAlgorithm(objective=objective, constraints=[], bounds=bounds, options=options, args=('minmax',))
    # for i in range(n_runs):
    #     print(f'Initialize run {i + 1}')
    #     score, decoded, _ = ga.run()
    #     print(f'Optimal result for x1 = {decoded[0]} stations and x2 = {round(decoded[1], 2)} trains')
    #     save_array_mm.append([decoded[0], decoded[1], sum(decoded)])
    #     print(f'Finished run {i + 1}')

    save_array = save_array
    # save_array_mm = save_array_mm

    # Create figure that shows the results in the solution space, the solution space is also
    # shown in figure 3 in the report. The optimal results are determined in this model.
    x_fill = [3, 10, 10, 3]
    y_fill = [20, 20, 2, 2]

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.set_xlim((0, 15))
    ax.set_ylim((0, 25))
    ax.set_xlabel('x1 [Number of stations]')
    ax.set_ylabel('x2 [Number of trains per hour]')
    ax.set_title('Solution space')
    ax.fill_between(x_fill, y_fill, color='#539ecd', label='Solution space')
    ax.scatter(np.array(save_array)[:, 0], np.array(save_array)[:, 1], label='Optimal solutions Tetra')
    # ax.scatter(np.array(save_array_mm)[:, 0], np.array(save_array_mm)[:, 1], label='Optimal solutions MinMax',
    #            marker='^')
    ax.scatter([9], [12], label='As-built', marker='p', color='r')
    ax.grid()
    fig.legend()

    # calculate individual preference scores for the results of the GA, to plot them on the preference curves
    variable = np.array(save_array)
    # variable_mm = np.array(save_array_mm)

    c1_res = objective_municipality(variable[:, 0], variable[:, 1])
    c2_res = objective_inhabitants(variable[:, 0], variable[:, 1])
    c3_res = objective_operator(variable[:, 0], variable[:, 1])
    c4_res = objective_project_team(variable[:, 0])

    # c1_res_mm = objective_municipality(variable_mm[:, 0], variable_mm[:, 1])
    # c2_res_mm = objective_inhabitants(variable_mm[:, 0], variable_mm[:, 1])
    # c3_res_mm = objective_operator(variable_mm[:, 0], variable_mm[:, 1])
    # c4_res_mm = objective_project_team(variable_mm[:, 0])

    c1_res_actual = objective_municipality(9, 12)
    c2_res_actual = objective_inhabitants(9, 12)
    c3_res_actual = objective_operator(9, 12)
    c4_res_actual = objective_project_team(9)

    p1_res = pchip_interpolate([p1_min, p1_mid, p1_max], [0, 100, 60], c1_res)
    p2_res = pchip_interpolate([p2_min, p2_mid1, p2_mid2, p2_max], [100, 80, 10, 0], c2_res)
    p3_res = pchip_interpolate([p3_min, p3_mid, p3_max], [60, 100, 0], c3_res)
    p4_res = pchip_interpolate([p4_min, p4_mid, p4_max], [100, 95, 0], c4_res)

    # p1_res_mm = pchip_interpolate([p1_min, p1_mid, p1_max], [0, 100, 60], c1_res_mm)
    # p2_res_mm = pchip_interpolate([p2_min, p2_mid1, p2_mid2, p2_max], [100, 80, 10, 0], c2_res_mm)
    # p3_res_mm = pchip_interpolate([p3_min, p3_mid, p3_max], [60, 100, 0], c3_res_mm)
    # p4_res_mm = pchip_interpolate([p4_min, p4_mid, p4_max], [100, 95, 0], c4_res_mm)

    p1_res_actual = pchip_interpolate([p1_min, p1_mid, p1_max], [0, 100, 60], c1_res_actual)
    p2_res_actual = pchip_interpolate([p2_min, p2_mid1, p2_mid2, p2_max], [100, 80, 10, 0], c2_res_actual)
    p3_res_actual = pchip_interpolate([p3_min, p3_mid, p3_max], [60, 100, 0], c3_res_actual)
    p4_res_actual = pchip_interpolate([p4_min, p4_mid, p4_max], [100, 95, 0], c4_res_actual)

    print(c1_res, p1_res)
    print(c2_res, p2_res)
    print(c3_res, p3_res)
    print(c4_res, p4_res)

    # create figure that plots all preference curves and the preference scores of the returned results of the GA
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(15, 5))

    ax1.scatter(c1_res, p1_res, label='Optimal solutions Tetra', color='tab:purple', zorder=1)
    # ax1.scatter(c1_res_mm, p1_res_mm, label='Optimal solutions MinMax', marker='^', color='tab:orange', zorder=1)
    ax1.scatter(c1_res_actual, p1_res_actual, label='As built', marker='p', color='r', zorder=1)
    ax1.plot(c1, p1, zorder=3)
    ax1.set_xlim((0, 5500000))
    ax1.set_ylim((0, 100))
    ax1.set_title('Income municipality')
    ax1.set_xlabel('Income [€]')
    ax1.set_ylabel('Preference score')
    fig.legend()
    ax1.grid()

    ax2.scatter(c2_res, p2_res, color='tab:purple', zorder=1)
    # ax2.scatter(c2_res_mm, p2_res_mm, marker='^', color='tab:orange', zorder=1)
    ax2.scatter(c2_res_actual, p2_res_actual, marker='p', color='tab:red', zorder=1)
    ax2.plot(c2, p2, zorder=3)
    ax2.set_xlim((0, 60))
    ax2.set_ylim((0, 100))
    ax2.set_title('Travel time')
    ax2.set_xlabel('Travel time [min]')
    ax2.set_ylabel('Preference score')
    ax2.grid()

    ax3.scatter(c3_res, p3_res, color='tab:purple', zorder=1)
    # ax3.scatter(c3_res_mm, p3_res_mm, marker='^', color='tab:orange', zorder=1)
    ax3.scatter(c3_res_actual, p3_res_actual, marker='p', color='tab:red', zorder=1)
    ax3.plot(c3, p3, zorder=3)
    ax3.set_xlim((0, 1500000))
    ax3.set_ylim((0, 100))
    ax3.set_title('Operational costs')
    ax3.set_xlabel('Operational costs [€]')
    ax3.set_ylabel('Preference score')
    ax3.grid()

    ax4.scatter(c4_res, p4_res, color='tab:purple')
    # ax4.scatter(c4_res_mm, p4_res_mm, marker='^', color='tab:orange')
    ax4.scatter(c4_res_actual, p4_res_actual, marker='p', color='tab:red')
    ax4.plot(c4, p4, zorder=3)
    ax4.set_xlim((0, 7))
    ax4.set_ylim((0, 100))
    ax4.set_title('Building time')
    ax4.set_xlabel('Building time [years]')
    ax4.set_ylabel('Preference score')
    ax4.grid()

    plt.show()
