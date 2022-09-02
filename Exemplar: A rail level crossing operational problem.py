"""
Python code for the rail level crossing operational problem exemplar
"""

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
    z = data.transpose()
    x = np.arange(0.3, 0.75, 0.05)
    y = np.arange(0, 16, 1)  # nr of sleepers
    return interp2d(x, y, z, kind='cubic')


# weights setting, weights can be adjusted between runs to check for outcome changes
w1 = 1 / 3
w2 = 1 / 3
w3 = 1 / 3

# x_points: the outcomes of the objective for which a preference score is defined by the stakeholders
# p_points: the corresponding preference scores
x_points_1, p_points_1 = [[0, 6000, 30000], [100, 50, 0]]
x_points_2, p_points_2 = [[0, 0.3, 1], [0, 40, 100]]
x_points_3, p_points_3 = [[0, 9000, 15000], [100, 70, 0]]

# import Tetra solver
solver = TetraSolver()

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
    Function to calculate the maintenance costs.

    :param force: force on rail
    :param acc: acceleration of rail
    :return: maintenance costs
    """
    norm_force = 1 - (force - min_force) / (max_force - min_force)
    norm_acc = 1 - (acc - min_acc) / (max_acc - min_acc)
    agg = np.sqrt(norm_force ** 2 + norm_acc ** 2)
    return agg * 15000


def objective(variables):
    """
    Objective function that is fed to the GA. Calles the separate preference functions that are declared above.

    :param variables: array with design variable values per member of the population. Can be split by using array
    slicing
    :return: 1D-array with aggregated preference scores for the members of the population.
    """
    var1 = variables[:, 0]  # sleeper distance
    var2 = variables[:, 1]  # number of sleepers

    force_array = force_inter(var1, var2)  # get force on rail for every member of the population
    acc_array = acc_inter(var1, var2)  # get acceleration of rail for every member of the population

    # it might be that the force and acceleration arrays are 2d instead of 1d. Hence, the following is needed to
    # translate them to 1d
    force = list()
    acc = list()

    for it in range(len(var1)):
        try:  # if array is indeed 2d
            force.append(force_array[it][it])
            acc.append(acc_array[it][it])
        except IndexError:  # if array is 1d
            force.append(force_array[it])
            acc.append(acc_array[it])

    # calculate objectives
    maintenance_costs = objective_maintenance_costs(np.array(force), np.array(acc))
    riding_comfort = 1 - (np.array(acc) - min_acc) / (max_acc - min_acc)
    investment_costs = var2 * 1000 - np.multiply(var1, var2) * 350

    # calculate the preference scores
    p_1 = pchip_interpolate(x_points_1, p_points_1, maintenance_costs)
    p_2 = pchip_interpolate(x_points_2, p_points_2, riding_comfort)
    p_3 = pchip_interpolate(x_points_3, p_points_3, investment_costs)

    # check if any preference scores are > 100 or < 100. If so, they are set to 100 and 0 resp.
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

n_runs = 2

# make dictionary with parameter settings for the GA run with the Tetra solver
options = {
    'n_bits': 9,
    'n_iter': 400,
    'n_pop': 150,
    'r_cross': 0.9,
    'max_stall': 10,
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

# make numpy array of results, to allow for array splicing
variable = np.round_(np.array(save_array), 2)

# create figure that shows the results in the solution space
fig, ax = plt.subplots(figsize=(7, 7))
ax.set_xlim((0, 1))
ax.set_ylim((0, 20))
ax.set_xlabel('x1: Sleeper spacing [m]')
ax.set_ylabel('x2: Number of sleepers')
ax.set_title('Solution space')

# define corner points of solution space
x_fill = [0.3, 0.7, 0.7, 0.3]
y_fill = [4, 4, 15, 15]

ax.fill_between(x_fill, y_fill, color='#539ecd', label='Solution space')
ax.scatter(variable[:, 0], variable[:, 1], label='Optimal solutions Tetra', color='tab:purple')

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
c1_res = objective_maintenance_costs(np.array(f), np.array(a))
c2_res = 1 - (np.array(a_res) - min_acc) / (max_acc - min_acc)
c3_res = variable[:, 1] * 1000 - np.multiply(variable[:, 0], variable[:, 1]) * 350

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

df = np.loadtxt('data/data_force.txt', delimiter=',')
da = np.loadtxt('data/data_acceleration.txt', delimiter=',')

# create figures and plot the force and acceleration data
fig = plt.figure()
plt.rc('font', size=15)
ax = fig.add_subplot(projection='3d')
ax.scatter(df[:, 0], df[:, 1], df[:, 2] * 1e-3)
ax.set_xlabel('Sleeper Spacing', labelpad=10)
ax.set_ylabel('Number of Sleepers', labelpad=10)
ax.set_zlabel('RMS Force [kN]', labelpad=10)

fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.scatter(da[:, 0], da[:, 1], da[:, 2])
ax.set_xlabel('Sleeper Spacing', labelpad=10)
ax.set_ylabel('Number of Sleepers', labelpad=10)
ax.set_zlabel(r'RMS Acceleration [$m/s^2$]', labelpad=10)

plt.show()
