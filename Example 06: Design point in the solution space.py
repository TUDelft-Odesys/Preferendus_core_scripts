"""Example from Preference-based optimization reader

Code adapted by Harold van Heukelum
"""
import sys

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import gridspec
from scipy.interpolate import pchip_interpolate

from genetic_algorithm_pfm import GeneticAlgorithm
from tetra_pfm import TetraSolver
from weighted_minmax.algorithm import aggregate_max

# create arrays for plotting continuous preference curves
c1 = np.linspace(0, 2, 100)
c2 = np.linspace(0, 2, 100)

norm_1 = np.linspace(0, 1, 100)
norm_2 = np.linspace(0, 1, 100)
N1, N2 = np.meshgrid(norm_1, norm_2)

p1 = 100 * (np.sqrt(N1 ** 2 + N2) ** 2)
m = p1 > 100
p1[m] = 0
p2 = pchip_interpolate([0, 1, 2], [0, 100, 60], c2)

# import Tetra solver
solver = TetraSolver()

# set weights
w1 = 0.65
w2 = 0.35


def objective_p1(x1, x2):
    normalized_x1 = 1 - (x1 - 100) / (1000 - 100)
    normalized_x2 = (x2 - 800) / (30_000 - 800)

    ret = 100 * (np.sqrt(normalized_x1 ** 2 + normalized_x2) ** 2)
    mask = ret > 100
    ret[mask] = 0

    return ret


def objective_p2(x1, x2):
    return pchip_interpolate([0, 1, 2], [0, 100, 60], x2 / (20_000 * x1 / 400))


def objective(variables, method='tetra'):
    x1 = variables[:, 0]
    x2 = variables[:, 1]
    p_1 = objective_p1(x1, x2)
    p_2 = objective_p2(x1, x2)

    mask1 = np.array(p_1) > 100
    mask2 = np.array(p_1) < 0
    p_1[mask1] = 100
    p_1[mask2] = 0

    mask1 = np.array(p_2) > 100
    mask2 = np.array(p_2) < 0
    p_2[mask1] = 100
    p_2[mask2] = 0

    if method == 'minmax':
        return aggregate_max([w1, w2], [p_1, p_2], 100)
    else:
        return solver.request([w1, w2], [p_1, p_2])


def constraint_1(variables):
    x1 = variables[:, 0]
    x2 = variables[:, 1]

    normalized_x1 = 1 - (x1 - 100) / (1000 - 100)
    normalized_x2 = (x2 - 800) / (30_000 - 800)
    return normalized_x1 + normalized_x2 - 0.80


# set bounds for all variables
b1 = [100, 1000]  # x1
b2 = [800, 30_000]  # x2
bounds = [b1, b2]

cons = []

if __name__ == '__main__':

    # make dictionary with parameter settings for the GA
    options = {
        'n_bits': 12,
        'n_iter': 400,
        'n_pop': 500,
        'r_cross': 0.9,
        'max_stall': 15,
        'tetra': True,
        'var_type_mixed': ['real', 'int']
    }

    # run the GA several times, print results to console, and save results to allow for plotting
    save_array = list()
    ga = GeneticAlgorithm(objective=objective, constraints=cons, bounds=bounds, options=options)
    for i in range(1):
        print(f'Initialize run {i + 1}')
        score, decoded, _ = ga.run()

        print(f'Optimal result for a distance of {round(decoded[0], 2)} meters and {decoded[1]} products')

        save_array.append([decoded[0], decoded[1]])
        print(f'Finished run {i + 1}')

    # make dictionary with parameter settings for the GA
    options = {
        'n_bits': 12,
        'n_iter': 400,
        'n_pop': 1000,
        'r_cross': 0.9,
        'max_stall': 15,
        'tetra': False,
        'var_type_mixed': ['real', 'int']
    }

    # run the GA several times, print results to console, and save results to allow for plotting
    save_array_mm = list()
    ga = GeneticAlgorithm(objective=objective, constraints=cons, bounds=bounds, options=options,
                          args=('minmax',))
    for i in range(1):
        print(f'Initialize run {i + 1}')
        score, decoded, _ = ga.run()

        print(f'Optimal result for a distance of {round(decoded[0], 2)} meters and {decoded[1]} products')

        save_array_mm.append([decoded[0], decoded[1]])
        print(f'Finished run {i + 1}')

    # create figure that shows the results in the solution space
    x_fill = [100, 1000, 1000, 100]
    y_fill = [30_000, 30_000, 800, 800]

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.set_xlim((0, 1200))
    ax.set_ylim((600, 32_000))
    ax.set_xlabel('x1 [m]')
    ax.set_ylabel('x2 [-]')
    ax.set_title('Solution space')
    ax.fill_between(x_fill, y_fill, color='#539ecd', label='Solution space')
    ax.scatter(np.array(save_array)[:, 0], np.array(save_array)[:, 1], label='Optimal solution Tetra')
    ax.scatter(np.array(save_array_mm)[:, 0], np.array(save_array_mm)[:, 1], label='Optimal solution MinMax',
               marker='*')
    ax.grid()
    fig.legend()

    # calculate individual preference scores for the results of the GA, to plot them on the preference curves
    variable = np.array(save_array)
    variable_mm = np.array(save_array_mm)

    normalized_c1x1 = 1 - (variable[:, 0] - 100) / (1000 - 100)
    normalized_c1x2 = (variable[:, 1] - 800) / (30_000 - 800)

    c2_res = variable[:, 1] / (20_000 * variable[:, 0] / 400)

    p1_res = objective_p1(variable[:, 0], variable[:, 1])
    p2_res = objective_p2(variable[:, 0], variable[:, 1])

    normalized_c1x1_mm = 1 - (variable[:, 0] - 100) / (1000 - 100)
    normalized_c1x2_mm = (variable[:, 1] - 800) / (30_000 - 800)

    c2_res_mm = variable_mm[:, 1] / (20_000 * variable_mm[:, 0] / 400)

    p1_res_mm = objective_p1(variable_mm[:, 0], variable_mm[:, 1])
    p2_res_mm = objective_p2(variable_mm[:, 0], variable_mm[:, 1])

    # create figure that plots all preference curves and the preference scores of the returned results of the GA
    plt.figure()
    ax2 = plt.subplot(122)
    ax2.plot(c2, p2, label='Preference curve')
    ax2.scatter(c2_res, p2_res, label='Solution Tetra')
    ax2.scatter(c2_res_mm, p2_res_mm, label='Solution MinMax', marker='*')
    ax2.set(xlabel='Sustainability index', ylabel='Preference')
    ax2.set_title('Preference Curve Transport Sustainability & Wasted')
    ax2.legend()
    ax2.grid()

    ax1 = plt.subplot(221, projection='3d')
    surf = ax1.plot_surface(N1, N2, p1, label='Preference curve')
    ax1.scatter(normalized_c1x1, normalized_c1x2, p1_res, label='Solution Tetra')
    ax1.scatter(normalized_c1x1_mm, normalized_c1x2_mm, p1_res_mm, label='Solution MinMax', marker='*')
    ax1.set(xlabel='Normalized travel distance', ylabel='Normalized items in assortiment', zlabel='Preference')
    ax1.set_title('Preference Curves Shopping Added Value')
    ax1.view_init(elev=15, azim=160)
    surf._facecolors2d = surf._facecolor3d
    surf._edgecolors2d = surf._edgecolor3d
    ax1.grid()

    ax3 = plt.subplot(223)
    fig = ax3.imshow(p1, cmap='GnBu', interpolation='nearest')
    ax3.scatter(normalized_c1x1 * 100, normalized_c1x2 * 100, label='Solution Tetra')
    ax3.scatter(normalized_c1x1_mm * 100, normalized_c1x2_mm * 100, label='Solution MinMax', marker='*')
    ax3.set(xlabel='Normalized travel distance', ylabel='Normalized items in assortiment')
    ax3.grid()

    plt.colorbar(fig, ax=ax3, location='left')

    plt.show()
