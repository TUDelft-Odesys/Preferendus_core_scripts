"""
Python code for the rail level crossing operational problem exemplar
"""

raise BaseException('Example is not working properly atm!')

import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import pchip_interpolate, interp2d

from genetic_algorithm_pfm import GeneticAlgorithm
from tetra_pfm import TetraSolver

"""
This script contains the code to run the  a priori optimization of the rail level crossing operational problem, 
described in chapter 6 of the reader.
"""


def interpolate_data(data):
    """
    Function to get an interpolated function from the discrete input.

    :param data: numpy array with data points [x, y, z]
    :return: interp2d
    """

    x, y, z = data[:, 0], data[:, 1], data[:, 2]
    x = x.reshape((10, 10))  # sleeper distance
    y = y.reshape((10, 10))  # nr of sleepers
    z = z.reshape((10, 10))

    return interp2d(x, y, z, kind='cubic')


# weights setting, weights can be adjusted between runs to check for outcome changes
w1 = 0.3
w2 = 0.2
w3 = 0.5

# x_points: the outcomes of the objective for which a preference score is defined by the stakeholders
# p_points: the corresponding preference scores
x_points_1, p_points_1 = [[0, 1000, 8000], [100, 50, 0]]
x_points_2, p_points_2 = [[0, 0.4, 1], [0, 40, 100]]
x_points_3, p_points_3 = [[10000, 25000, 37500], [100, 40, 0]]

# import Tetra solver
solver = TetraSolver()

# import force and acceleration data
data_force = np.unique(np.loadtxt('data/Data_force.txt', delimiter=','), axis=0)
data_acc = np.unique(np.loadtxt('data/Data_acceleration.txt', delimiter=','), axis=0)

data_force[:, 0] *= 0.05
data_acc[:, 0] *= 0.05

max_acc = 0.1

# get interpolated functions for force and acceleration
force_inter = interpolate_data(data_force)
acc_inter = interpolate_data(data_acc)


def objective_maintenance_costs(force, acc):
    """
    Function to calculate the maintenance costs.

    :param force: force on rail
    :param acc: acceleration of rail
    :return: maintenance costs
    """
    return np.multiply(2, force) + np.multiply(20000, acc)


def objective(variables):
    """
    Objective function that is fed to the GA. Calles the separate preference functions that are declared above.

    :param variables: array with design variable values per member of the population. Can be split by using array
    slicing
    :return: 1D-array with aggregated preference scores for the members of the population.
    """
    x1 = variables[:, 0]  # sleeper distance
    x2 = variables[:, 1]  # number of sleepers

    force_array = force_inter(x1, x2)  # get force on rail for every member of the population
    acc_array = acc_inter(x1, x2)  # get acceleration of rail for every member of the population

    # it might be that the force and acceleration arrays are 2d instead of 1d. Hence, the following is needed to
    # translate them to 1d
    force = list()
    acc = list()

    for it in range(len(x1)):
        try:  # if array is indeed 2d
            force.append(force_array[it][it])
            acc.append(acc_array[it][it])
        except IndexError:  # if array is 1d
            force.append(force_array[it])
            acc.append(acc_array[it])

    # calculate objectives
    maintenance_costs = objective_maintenance_costs(force, acc)
    # riding_comfort = 1 - np.multiply(acc, 12)
    riding_comfort = 1 - np.multiply(acc, 1 / max_acc)
    investment_costs = x2 * 1000

    # calculate the preference scores
    p_1 = pchip_interpolate(x_points_1, p_points_1, maintenance_costs)
    p_2 = pchip_interpolate(x_points_2, p_points_2, riding_comfort)
    p_3 = pchip_interpolate(x_points_3, p_points_3, investment_costs)

    # check if any preference scores are > 100 or < 100. If so, they are set to 100 and 0 resp.
    mask_higher = p_1 > 100
    mask_lower = p_1 < 0
    p_1[mask_higher] = 100
    p_1[mask_lower] = 0
    if mask_lower.any() or mask_higher.any():
        print('p1')
        print(min(maintenance_costs), max(maintenance_costs))
        print(min(p_1), max(p_1))

    mask_higher = p_2 > 100
    mask_lower = p_2 < 0
    p_2[mask_higher] = 100
    p_2[mask_lower] = 0
    if mask_lower.any() or mask_higher.any():
        print('p2')
        print(min(riding_comfort), max(riding_comfort))
        print(min(p_2), max(p_2))

    mask_higher = p_3 > 100
    mask_lower = p_3 < 0
    p_3[mask_higher] = 100
    p_3[mask_lower] = 0
    if mask_lower.any() or mask_higher.any():
        print('p3')
        print(min(investment_costs), max(investment_costs))
        print(min(p_3), max(p_3))

    # aggregate preference scores and return this to the GA
    return solver.request([w1, w2, w3], [p_1, p_2, p_3])


"""
Below, the a priori optimization is performed. For this, we first need to define the bounds of the design variables. 
There are no constraints.

The optimization can be ran multiple times, so you can check the consistency between the runs. The outcomes might differ
a bit, since the GA is stochastic from nature, but the differences should be limited.
"""

# bounds setting for the 2 variables
b1 = [0.3, 0.7]  # x1
b2 = [4, 15]  # x2
bounds = [b1, b2]

n_runs = 1

# make dictionary with parameter settings for the GA run with the Tetra solver
options = {
    'n_bits': 12,
    'n_iter': 400,
    'n_pop': 250,
    'r_cross': 0.9,
    'max_stall': 1,
    'tetra': True,
    'var_type_mixed': ['real', 'int']
}

save_array = list()  # list to save the results from every run to
ga = GeneticAlgorithm(objective=objective, constraints=[], bounds=bounds, options=options)

# run the GA and print its result
for i in range(n_runs):
    print(f'Initialize run {i + 1}')
    score, design_variables, _ = ga.run()

    print(f'Optimal result for a sleeper distance of {round(design_variables[0], 2)}m and '
          f'{design_variables[1]} sleepers')

    save_array.append(design_variables)
    print(f'Finished run {i + 1}')

"""
Now we have the results, we can plot the preference functions together with the results of the optimizations.
"""

# create figure that shows the results in the solution space
fig, ax = plt.subplots(figsize=(7, 7))
ax.set_xlim((0, 1))
ax.set_ylim((0, 17))
ax.set_xlabel('x1: Sleeper spacing [m]')
ax.set_ylabel('x2: Number of sleepers')
ax.set_title('Solution space')

# define corner points of solution space
x_fill = [0.3, 0.7, 0.7, 0.3]
y_fill = [4, 4, 15, 15]

ax.fill_between(x_fill, y_fill, color='#539ecd', label='Solution space')
ax.scatter(np.array(save_array)[:, 0], np.array(save_array)[:, 1], label='Optimal solutions Tetra', color='tab:purple')

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

# make numpy array of results, to allow for array splicing
variable = np.array(save_array)

f = force_inter(variable[:, 0], variable[:, 1])  # get the forces on the rail
a = acc_inter(variable[:, 0], variable[:, 1])  # get the accelerations of the rail

# it might be that the force and acceleration arrays are 2d instead of 1d. Hence, the following is needed to
# translate them to 1d
f_res = list()
a_res = list()

for i in range(len(variable[:, 0])):
    try:  # if array is indeed 2d
        f_res.append(f[i][i])
        a_res.append(a[i][i])
    except IndexError:  # if array is 1d
        f_res.append(f[i])
        a_res.append(a[i])

# calculate individual preference scores for the results of the GA, to plot them on the preference curves
c1_res = objective_maintenance_costs(f, a)
# c2_res = 1 - np.multiply(a_res, 12)
c2_res = 1 - np.multiply(a_res, 1 / max_acc)
c3_res = variable[:, 1] * 2500

p1_res = pchip_interpolate(x_points_1, p_points_1, c1_res)
p2_res = pchip_interpolate(x_points_2, p_points_2, c2_res)
p3_res = pchip_interpolate(x_points_3, p_points_3, c3_res)

# create figure that plots all preference curves and the preference scores of the returned results of the GA
fig = plt.figure()
ax1 = fig.add_subplot(1, 1, 1)
ax1.plot(c1, p1, label='Preference curve')
ax1.scatter(c1_res, p1_res, label='Optimal solution Tetra', color='tab:purple')
ax1.set_ylim((0, 100))
ax1.set_title('Maintenance Costs')
ax1.set_xlabel('Maintenance Costs [€/y]')
ax1.set_ylabel('Preference')
ax1.grid()
ax1.legend()

fig = plt.figure()
ax2 = fig.add_subplot(1, 1, 1)
ax2.plot(c2, p2, label='Preference curve')
ax2.scatter(c2_res, p2_res, label='Optimal solution Tetra', color='tab:purple')
ax2.set_ylim((0, 100))
ax2.set_title('Travel Comfort')
ax2.set_xlabel(r'Travel Comfort [$m/s^2$]')
ax2.set_ylabel('Preference')
ax2.grid()
ax2.legend()

fig = plt.figure()
ax3 = fig.add_subplot(1, 1, 1)
ax3.plot(c3 / 1000, p3, label='Preference curve')
ax3.scatter(c3_res / 1000, p3_res, label='Optimal solution Tetra', color='tab:purple')
ax3.set_ylim((0, 100))
ax3.set_title('Investment Costs')
ax3.set_xlabel(r'Investment Costs [€$*10^3$]')
ax3.set_ylabel('Preference')
ax3.grid()
ax3.legend()

# create figures and plot the force and acceleration data
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.scatter(data_force[:, 0], data_force[:, 1], data_force[:, 2])
ax.set(xlabel='Sleeper Spacing', ylabel='Number of Sleepers', zlabel='RMS Force [kN]')

fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.scatter(data_acc[:, 0], data_acc[:, 1], data_acc[:, 2])
ax.set(xlabel='Sleeper Spacing', ylabel='Number of Sleepers', zlabel=r'RMS Acceleration [$m/s^2$]')

plt.show()
