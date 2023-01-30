"""
Python code for the rail level crossing operational problem exemplar
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp2d

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


def objective_maintenance_costs(force, acc):
    """
    Function to calculate the maintenance costs"""
    norm_force = (force - min_force) / (max_force - min_force)
    norm_acc = (acc - min_acc) / (max_acc - min_acc)
    agg = np.sqrt(norm_force ** 2 + norm_acc ** 2)
    return agg * 15000


def single_objective_maintenance_costs(variables):
    """Function for single objective optimization of the maintenance costs"""
    var1 = variables[:, 0]  # sleeper distance
    var2 = variables[:, 1]  # number of sleepers

    force = list()
    acc = list()
    for ix in range(len(var1)):
        force.append(force_inter(var1[ix], var2[ix])[0])  # get force on rail for every member of the population
        acc.append(acc_inter(var1[ix], var2[ix])[0])  # get acceleration of rail for every member of the population

    return objective_maintenance_costs(np.array(force), np.array(acc))


def single_objective_travel_comfort(variables):
    """Function for single objective optimization of the travel comfort"""
    var1 = variables[:, 0]  # sleeper distance
    var2 = variables[:, 1]  # number of sleepers

    acc = list()
    for ix in range(len(var1)):
        acc.append(acc_inter(var1[ix], var2[ix])[0])  # get acceleration of rail for every member of the population

    return np.multiply(1 - (np.array(acc) - min_acc) / (max_acc - min_acc), -1)


def single_objective_investment_costs(variables):
    """Function for single objective optimization of the investment costs"""
    var1 = variables[:, 0]  # sleeper distance
    var2 = variables[:, 1]  # number of sleepers

    costs = var2 * 1000 - np.multiply(var1, var2) * 350
    return costs


# set the bounds for the 2 variables
b1 = [0.3, 0.7]  # x1
b2 = [4, 15]  # x2
bounds = [b1, b2]


def print_results(x):
    """Function that prints the results of the optimizations"""
    print(f'Optimal result for x1 = {round(x[0], 2)}m and x2 = {round(x[1], 2)} sleepers')


# make dictionary with parameter settings for the GA
options = {
    'aggregation': None,
    'var_type_mixed': ['real', 'int']
}

save_array = list()
methods = list()

# maintenance costs
ga = GeneticAlgorithm(objective=single_objective_maintenance_costs, constraints=[], bounds=bounds,
                      options=options)
res, design_variables_save_array, _ = ga.run()
print_results(design_variables_save_array)
print(f'SODO Maintenance Costs = €{round(res, 2)}')
save_array.append(design_variables_save_array)
methods.append('SODO Maintenance Costs')

# travel comfort
ga = GeneticAlgorithm(objective=single_objective_travel_comfort, constraints=[], bounds=bounds,
                      options=options)
res, design_variables_travel_comfort, _ = ga.run()

print_results(design_variables_travel_comfort)
print(f'SODO Travel Comfort = {round(-1 * res, 2)}')
save_array.append(design_variables_travel_comfort)
methods.append('SODO Travel Comfort')

# investment costs
ga = GeneticAlgorithm(objective=single_objective_investment_costs, constraints=[], bounds=bounds,
                      options=options)
res, design_variables_investment_costs, _ = ga.run()

print_results(design_variables_investment_costs)
print(f'SODO Investment Costs = €{round(res, 2)}')
save_array.append(design_variables_investment_costs)
methods.append('SODO Investment Costs')

# create figure that shows the results in the design space
markers = ['x', 'v', '+']
variable = np.array(save_array)  # make ndarray
fig, ax = plt.subplots()
ax.set_xlim((0.2, 0.8))
ax.set_ylim((3, 16))
ax.set_xlabel('x1: Sleeper spacing [m]')
ax.set_ylabel('x2: Number of sleepers')
ax.set_title('Design space')

# define corner points of solution space
x_fill = [0.3, 0.7, 0.7, 0.3]
y_fill = [4, 4, 15, 15]

ax.fill_between(x_fill, y_fill, color='#539ecd', label='Design space')
for i in range(len(save_array)):
    ax.scatter(variable[i, 0], variable[i, 1], label=methods[i], marker=markers[i], s=100)

ax.grid()  # show grid
ax.legend()  # show legend
plt.show()
