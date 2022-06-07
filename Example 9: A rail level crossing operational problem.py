"""Credits to Harold van Heukelum for creating the code and the model"""

import warnings

import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import pchip_interpolate, interp2d

from genetic_algorithm_pfm import GeneticAlgorithm
from tetra_pfm import TetraSolver

data_force = np.loadtxt('Data_force.txt', delimiter=',')
data_acc = np.loadtxt('Data_acceleration.txt', delimiter=',')

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=RuntimeWarning)


def interpolate_data(data):
    """

    :param data:
    :return:
    """

    x, y, z = data[:, 0], data[:, 1], data[:, 2]
    x = x.reshape((10, 10))  # sleeper distance
    y = y.reshape((10, 10))  # nr of sleepers
    z = z.reshape((10, 10))

    return interp2d(x, y, z, kind='cubic')


fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.scatter(data_force[:, 0] * 0.05, data_force[:, 1], data_force[:, 2])
ax.set(xlabel='Sleeper Spacing', ylabel='Number of Sleepers', zlabel='RMS Force [kN]')

fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.scatter(data_acc[:, 0] * 0.05, data_acc[:, 1], data_acc[:, 2])
ax.set(xlabel='Sleeper Spacing', ylabel='Number of Sleepers', zlabel=r'RMS Acceleration [$m/s^2$]')


def objective_maintainance_costs(force, acc):
    return np.multiply(2, force) + np.multiply(20000, acc)


c1 = np.linspace(900, 6000)  # maintenance costs
c2 = np.linspace(0, 5)  # comfort
c3 = np.linspace(10000, 75000)  # investment costs

p1_min, p1_mid, p1_max = [900, 2500, 6000]
p2_min, p2_mid, p2_max = [0, 2, 5]
p3_min, p3_mid, p3_max = [10000, 35000, 75000]

p1 = pchip_interpolate([p1_min, p1_mid, p1_max], [100, 50, 0], c1)
p2 = pchip_interpolate([p2_min, p2_mid, p2_max], [100, 50, 0], c2)
p3 = pchip_interpolate([p3_min, p3_mid, p3_max], [100, 50, 0], c3)


# import Tetra solver
solver = TetraSolver()

# weights setting, weights can be adjusted between runs to check for outcome changes
w1 = 0.35
w2 = 0.2
w3 = 0.45

force_inter = interpolate_data(data_force)
acc_inter = interpolate_data(data_acc)


def objective(variables, method='tetra'):
    """Objective function that calculates three preference scores for all members in the population. These preference
    scores are aggegated by using the Tetra solver.

    :param variables: ndarray of n-by-m, with n the population size of the GA and m the number of variables.
    :param method:
    :return: list with aggregated preference scores (size n-by-1)
    """
    x1 = variables[:, 0]  # sleeper distance
    x2 = variables[:, 1]  # number of sleepers

    force_array = force_inter(x1, x2)
    acc_array = acc_inter(x1, x2)

    force = list()
    acc = list()

    for it in range(len(x1)):
        try:
            force.append(force_array[it][it])
            acc.append(acc_array[it][it])
        except IndexError:
            force.append(force_array[it])
            acc.append(acc_array[it])

    maintenance_costs = objective_maintainance_costs(force, acc)
    riding_comfort = np.multiply(acc, 8)
    investment_costs = x2 * 5000

    p_1 = pchip_interpolate([p1_min, p1_mid, p1_max], [100, 50, 0], maintenance_costs)
    p_2 = pchip_interpolate([p2_min, p2_mid, p2_max], [100, 50, 0], riding_comfort)
    p_3 = pchip_interpolate([p3_min, p3_mid, p3_max], [100, 50, 0], investment_costs)

    mask_higher = p_1 > 100
    mask_lower = p_1 < 0
    p_1[mask_higher] = 100
    p_1[mask_lower] = 0

    mask_higher = p_2 > 100
    mask_lower = p_2 < 0
    p_2[mask_higher] = 100
    p_2[mask_lower] = 0

    mask_higher = p_3 > 100
    mask_lower = p_3 < 0
    p_3[mask_higher] = 100
    p_3[mask_lower] = 0

    ret = solver.request([w1, w2, w3], [p_1, p_2, p_3])
    return ret


# bounds setting for the 2 variables
b1 = [6, 14]  # x1
b2 = [4, 15]  # x2
bounds = [b1, b2]

if __name__ == '__main__':
    n_runs = 1

    # make dictionary with parameter settings for the GA, no changes have been made here.
    options = {
        'n_bits': 20,
        'n_iter': 400,
        'n_pop': 500,
        'r_cross': 0.85,
        'max_stall': 10,
        'tetra': True,
        'method_tetra': 1,
        'var_type_mixed': ['real', 'int']
    }

    save_array = list()
    ga = GeneticAlgorithm(objective=objective, constraints=[], bounds=bounds, options=options, args=('tetra',))
    for i in range(n_runs):
        print(f'Initialize run {i + 1}')
        score, decoded, _ = ga.run()
        print(f'Optimal result for a sleeper distance of {round(decoded[0] * 0.05, 2)}m and {decoded[1]} sleepers')
        save_array.append([decoded[0], decoded[1]])
        print(f'Finished run {i + 1}')

    # Create figure that shows the results in the solution space, the solution space is also
    # shown in figure 3 in the report. The optimal results are determined in this model.
    x_fill = [0.3, 0.7, 0.7, 0.3]
    y_fill = [0, 0, 15, 15]

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.set_xlim((0, 1))
    ax.set_ylim((0, 17))
    ax.set_xlabel('x1: Sleeper spacing [m]')
    ax.set_ylabel('x2: Number of sleepers')
    ax.set_title('Solution space')
    ax.fill_between(x_fill, y_fill, color='#539ecd', label='Solution space')
    ax.scatter(np.array(save_array)[:, 0] * 0.05, np.array(save_array)[:, 1], label='Optimal solutions Tetra')
    ax.legend()
    ax.grid(linewidth=0.3)

    # calculate individual preference scores for the results of the GA, to plot them on the preference curves
    variable = np.array(save_array)

    f = force_inter(variable[:, 0], variable[:, 1])
    a = acc_inter(variable[:, 0], variable[:, 1])

    f_res = list()
    a_res = list()

    for it in range(len(variable[:, 0])):
        try:
            f_res.append(f[it][it])
            a_res.append(a[it][it])
        except IndexError:
            f_res.append(f[it])
            a_res.append(a[it])

    c1_res = objective_maintainance_costs(f, a)
    c2_res = a * 8
    c3_res = variable[:, 1] * 5000

    p1_res = pchip_interpolate([p1_min, p1_mid, p1_max], [100, 50, 0], c1_res)
    p2_res = pchip_interpolate([p2_min, p2_mid, p2_max], [100, 50, 0], c2_res)
    p3_res = pchip_interpolate([p3_min, p3_mid, p3_max], [100, 50, 0], c3_res)

    print(f)
    print(a)

    print(c1_res, p1_res)
    print(c2_res, p2_res)
    print(c3_res, p3_res)

    # create figure that plots all preference curves and the preference scores of the returned results of the GA
    fig2, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))

    ax1.plot(c1, p1, label='Preference curve')
    ax1.scatter(c1_res, p1_res, label='Optimal solutions Tetra')
    ax1.set_xlim((900, 6000))
    ax1.set_ylim((0, 100))
    ax1.set_title('Maintenance Costs')
    ax1.set_xlabel('Maintenance Costs [€/y]')
    ax1.set_ylabel('Preference')
    ax1.grid()
    fig2.legend()

    ax2.plot(c2, p2)
    ax2.scatter(c2_res, p2_res)
    ax2.set_xlim((0, 5))
    ax2.set_ylim((0, 100))
    ax2.set_title('Travel Comfort')
    ax2.set_xlabel(r'Travel Comfort [$m/s^2$]')
    ax2.set_ylabel('Preference')
    ax2.grid()

    ax3.plot(c3 / 1000, p3)
    ax3.scatter(c3_res / 1000, p3_res)
    ax3.set_xlim((10, 75))
    ax3.set_ylim((0, 100))
    ax3.set_title('Investment Costs')
    ax3.set_xlabel(r'Investment Costs [€$*10^3$]')
    ax3.set_ylabel('Preference')
    ax3.grid()

    plt.show()
