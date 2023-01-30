"""
Python code for the rail level crossing operational problem exemplar
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.interpolate import pchip_interpolate, interp2d

from genetic_algorithm_pfm import GeneticAlgorithm

# todo: fix DeprecationWarning from interp2d!
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)


def interpolate_data(data):
    """
    Function to get an interpolated function from the discrete input.

    :param data: numpy array with data points [x, y, z]
    :return: interp2d
    """
    z = data.transpose()
    x = np.arange(0.3, 0.75, 0.05)
    y = np.arange(0, 16, 1)  # nr of sleepers
    return interp2d(x, y, z, kind='cubic')


# import force and acceleration data
data_force = np.loadtxt('data/Data_force_2d.csv', delimiter=';', encoding='utf-8-sig')
data_acc = np.loadtxt('data/Data_acceleration_2d.csv', delimiter=';', encoding='utf-8-sig')

max_force = np.amax(data_force)
min_force = np.amin(data_force)
max_acc = np.amax(data_acc) + 0.01
min_acc = np.amin(data_acc) - 0.01

# get interpolated functions for force and acceleration
force_inter = interpolate_data(data_force)
acc_inter = interpolate_data(data_acc)

# weights setting, weights can be adjusted between runs to check for outcome changes
w1 = 0.4
w2 = 0.4
w3 = 0.2

# x_points: the outcomes of the objective for which a preference score is defined by the stakeholders
# p_points: the corresponding preference scores
x_points_1, p_points_1 = [[3_500, 8_000, 19_000], [100, 50, 0]]
x_points_2, p_points_2 = [[0, 0.3, 1], [0, 40, 100]]
x_points_3, p_points_3 = [[3_000, 7_000, 15_000], [100, 40, 0]]


def objective_maintenance_costs(force, acc):
    """
    Function to calculate the maintenance costs"""
    norm_force = (force - min_force) / (max_force - min_force)
    norm_acc = (acc - min_acc) / (max_acc - min_acc)
    agg = np.sqrt(norm_force ** 2 + norm_acc ** 2)
    return agg * 15000


def single_objective_p1(variables):
    """Function for single objective optimization of the maintenance costs"""
    var1 = variables[:, 0]  # sleeper distance
    var2 = variables[:, 1]  # number of sleepers

    force = list()
    acc = list()
    for ix in range(len(var1)):
        force.append(force_inter(var1[ix], var2[ix])[0])  # get force on rail for every member of the population
        acc.append(acc_inter(var1[ix], var2[ix])[0])  # get acceleration of rail for every member of the population

    return objective_maintenance_costs(np.array(force), np.array(acc))


def single_objective_p2(variables):
    """Function for single objective optimization of the travel comfort"""
    var1 = variables[:, 0]  # sleeper distance
    var2 = variables[:, 1]  # number of sleepers

    acc = list()
    for ix in range(len(var1)):
        acc.append(acc_inter(var1[ix], var2[ix])[0])  # get acceleration of rail for every member of the population

    return np.multiply(1 - (np.array(acc) - min_acc) / (max_acc - min_acc), -1)


def single_objective_p3(variables):
    """Function for single objective optimization of the investment costs"""
    var1 = variables[:, 0]  # sleeper distance
    var2 = variables[:, 1]  # number of sleepers

    costs = var2 * 1000 - np.multiply(var1, var2) * 350
    return costs


def check_p_score(p):
    """Function to mak sure all preference scores are in [0,100]"""
    mask1 = p < 0
    mask2 = p > 100
    p[mask1] = 0
    p[mask2] = 100
    return p


def objective(variables):
    """Objective function for the GA. Calculates all sub-objectives and their corresponding preference scores. The
    aggregation is done in the GA"""
    var1 = variables[:, 0]  # sleeper distance
    var2 = variables[:, 1]  # number of sleepers

    force = list()
    acc = list()
    for ix in range(len(var1)):
        force.append(force_inter(var1[ix], var2[ix])[0])  # get force on rail for every member of the population
        acc.append(acc_inter(var1[ix], var2[ix])[0])  # get acceleration of rail for every member of the population

    # calculate objectives
    maintenance_costs = objective_maintenance_costs(np.array(force), np.array(acc))
    riding_comfort = 1 - (np.array(acc) - min_acc) / (max_acc - min_acc)
    investment_costs = var2 * 1000 - np.multiply(var1, var2) * 350

    # calculate the preference scores
    p_1 = check_p_score(pchip_interpolate(x_points_1, p_points_1, maintenance_costs))
    p_2 = check_p_score(pchip_interpolate(x_points_2, p_points_2, riding_comfort))
    p_3 = check_p_score(pchip_interpolate(x_points_3, p_points_3, investment_costs))

    return [w1, w2, w3], [p_1, p_2, p_3]


# set the bounds for the 2 variables
b1 = [0.3, 0.7]  # x1
b2 = [4, 15]  # x2
bounds = [b1, b2]


def print_results(x):
    """Function that prints the results of the optimizations"""
    print(f'Optimal result for x1 = {round(x[0], 2)}m and x2 = {round(x[1], 2)} sleepers')


if __name__ == '__main__':
    ####################################################################################
    # run single objectives and save to save_array
    save_array = list()
    methods = list()

    # make dictionary with parameter settings for the GA
    options = {
        'n_bits': 16,
        'n_iter': 400,
        'n_pop': 250,
        'r_cross': 0.8,
        'max_stall': 10,
        'var_type_mixed': ['real', 'int']
    }

    # maintenance costs
    ga = GeneticAlgorithm(objective=single_objective_p1, constraints=[], bounds=bounds,
                          options=options)
    res, design_variables_P1, _ = ga.run()
    print_results(design_variables_P1)
    save_array.append(design_variables_P1)
    methods.append('SODO 1')
    print(f'SODO maintenance costs = €{round(res, 2)}')

    # riding comfort
    ga = GeneticAlgorithm(objective=single_objective_p2, constraints=[], bounds=bounds,
                          options=options)
    res, design_variables_P2, _ = ga.run()
    print_results(design_variables_P2)
    save_array.append(design_variables_P2)
    methods.append('SODO 2')
    print(f'SODO maintenance costs = {round(res, 2)}')

    # investment costs
    ga = GeneticAlgorithm(objective=single_objective_p3, constraints=[], bounds=bounds,
                          options=options)
    res, design_variables_P3, _ = ga.run()
    print_results(design_variables_P3)
    save_array.append(design_variables_P3)
    methods.append('SODO 3')
    print(f'SODO investment costs = €{round(res, 2)}')

    ####################################################################################
    # run multi-objective with minmax solver

    # change some entries in the options dictionary
    options['n_bits'] = 8
    options['n_pop'] = 1200
    options['r_cross'] = 0.8
    options['tetra'] = False
    options['aggregation'] = 'minmax'

    ga = GeneticAlgorithm(objective=objective, constraints=[], cons_handler='CND', bounds=bounds, options=options)
    _, design_variables_minmax, best_mm = ga.run()
    print_results(design_variables_minmax)
    save_array.append(design_variables_minmax)
    methods.append('Min-max')

    ####################################################################################
    # run multi-objective with tetra solver

    # change some entries in the options dictionary
    options['n_bits'] = 8
    options['n_pop'] = 350
    options['r_cross'] = 0.8
    options['tetra'] = True
    options['aggregation'] = 'tetra'
    options['mutation_rate_order'] = 3

    ga = GeneticAlgorithm(objective=objective, constraints=[], bounds=bounds, options=options,
                          start_points_population=[design_variables_minmax])
    _, design_variables_tetra, best_t = ga.run()
    print_results(design_variables_tetra)
    save_array.append(design_variables_tetra)
    methods.append('IMAP')

    ####################################################################################
    # evaluate all runs

    variable = np.array(save_array)  # make ndarray
    w, p = objective(variable)  # evaluate objective
    r = ga.solver.request(w, p)  # get aggregated scores to rank them

    # create pandas DataFrame and print it to console
    d = {'Method': methods,
         'Results': r,
         'Variable 1': variable[:, 0],
         'Variable 2': variable[:, 1]
         }
    print()
    print(pd.DataFrame(data=d).to_string())
    print()

    # create figure that shows the results in the design space
    markers = ['x', 'v', '1', 's', '+']
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_xlim((0.2, 0.8))
    ax.set_ylim((3, 16))
    ax.set_xlabel('x1: Sleeper spacing [m]')
    ax.set_ylabel('x2: Number of sleepers')
    ax.set_title('Design space')

    # define corner points of design space
    x_fill = [0.3, 0.7, 0.7, 0.3]
    y_fill = [4, 4, 15, 15]

    ax.fill_between(x_fill, y_fill, color='#539ecd', label='Design space')
    for i in range(len(variable)):
        ax.scatter(variable[i, 0], variable[i, 1], label=methods[i], marker=markers[i])

    ax.grid()  # show grid
    ax.legend()  # show legend

    # arrays for plotting continuous preference curves
    c1 = np.linspace(x_points_1[0], x_points_1[-1])  # maintenance costs
    c2 = np.linspace(x_points_2[0], x_points_2[-1])  # comfort
    c3 = np.linspace(x_points_3[0], x_points_3[-1])  # investment costs

    # calculate the preference functions
    p1 = pchip_interpolate(x_points_1, p_points_1, c1)
    p2 = pchip_interpolate(x_points_2, p_points_2, c2)
    p3 = pchip_interpolate(x_points_3, p_points_3, c3)

    f_res = list()
    a_res = list()
    for i in range(len(variable)):
        f_res.append(force_inter(variable[:, 0][i], variable[:, 1][i])[0])  # get force on rail
        a_res.append(acc_inter(variable[:, 0][i], variable[:, 1][i])[0])  # get acceleration of rail

    # calculate individual preference scores for the results of the GA, to plot them on the preference curves
    c1_res = objective_maintenance_costs(np.array(f_res), np.array(a_res))
    c2_res = 1 - (np.array(a_res) - min_acc) / (max_acc - min_acc)
    c3_res = variable[:, 1] * 1000 - np.multiply(variable[:, 0], variable[:, 1]) * 350

    p1_res = pchip_interpolate(x_points_1, p_points_1, c1_res)
    p2_res = pchip_interpolate(x_points_2, p_points_2, c2_res)
    p3_res = pchip_interpolate(x_points_3, p_points_3, c3_res)

    d = {'Method': methods,
         'Maintenance costs': np.round_(c1_res, 2),
         'Travel comfort': np.round_(c2_res, 2),
         'Investment costs': np.round_(c3_res, 2),
         }
    print()
    print(pd.DataFrame(data=d).to_string())
    print()

    d = {'Method': methods,
         'Maintenance costs': np.round_(p1_res),
         'Travel comfort': np.round_(p2_res),
         'Investment costs': np.round_(p3_res),
         }
    print()
    print(pd.DataFrame(data=d).to_string())
    print()

    # create figure that plots all preference curves and the preference scores of the returned results of the GA
    fig = plt.figure(figsize=(15, 5))
    ax1 = fig.add_subplot(1, 3, 1)
    ax1.plot(c1 * 1e-3, p1, label='Preference Function')
    for i in range(len(c1_res)):
        ax1.scatter(c1_res[i] * 1e-3, p1_res[i], label=methods[i], marker=markers[i])
    ax1.set_ylim((0, 100))
    ax1.set_title('Maintenance Costs')
    ax1.set_xlabel(r'Maintenance Costs [€$*10^3$]')
    ax1.set_ylabel('Preference function outcome')
    ax1.grid()
    fig.legend()

    ax2 = fig.add_subplot(1, 3, 2)
    ax2.plot(c2, p2, label='Preference Function')
    for i in range(len(c1_res)):
        ax2.scatter(c2_res[i], p2_res[i], marker=markers[i])
    ax2.set_ylim((0, 100))
    ax2.set_title('Travel Comfort')
    ax2.set_xlabel(r'Travel Comfort')
    ax2.set_ylabel('Preference function outcome')
    ax2.grid()

    ax3 = fig.add_subplot(1, 3, 3)
    ax3.plot(c3 / 1000, p3, label='Preference Function')
    for i in range(len(c1_res)):
        ax3.scatter(c3_res[i] * 1e-3, p3_res[i], marker=markers[i])
    ax3.set_ylim((0, 100))
    ax3.set_title('Investment Costs')
    ax3.set_xlabel(r'Investment Costs [€$*10^3$]')
    ax3.set_ylabel('Preference function outcome')
    ax3.grid()

    df = np.loadtxt('data/Data_force.txt', delimiter=',')
    da = np.loadtxt('data/Data_acceleration.txt', delimiter=',')

    # create figures and plot the force and acceleration data
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(projection='3d')
    ax.scatter(df[:, 0] * 0.05, df[:, 1], df[:, 2] * 1e-3)
    ax.set_xlabel('Sleeper Spacing [m]', labelpad=10)
    ax.set_ylabel('Number of Sleepers', labelpad=10)
    ax.set_zlabel('RMS Force [kN]', labelpad=10)

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(projection='3d')
    ax.scatter(da[:, 0] * 0.05, da[:, 1], da[:, 2])
    ax.set_xlabel('Sleeper Spacing [m]', labelpad=10)
    ax.set_ylabel('Number of Sleepers', labelpad=10)
    ax.set_zlabel(r'RMS Acceleration [$m/s^2$]', labelpad=10)

    plt.show()
