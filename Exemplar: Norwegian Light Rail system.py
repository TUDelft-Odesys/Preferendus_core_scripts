"""
Python code for the Norwegian Light Rail system exemplar
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as ss
from scipy.interpolate import pchip_interpolate

from genetic_algorithm_pfm import GeneticAlgorithm
from tetra_pfm import TetraSolver
from weighted_minmax import aggregate_max

"""
This script contains the code to run both the a posteriori evaluation as the a priori optimization of the Bergen Light 
Rail example, described in chapter 5 of the reader. 

Let's begin with the definition of the solver, weights, and objectives.
"""

# Initialize TetraSolver
solver = TetraSolver()

# set weights for the different objectives
w1 = 0.2  # weight of municipality
w2 = 0.4  # weight of users and inhabitants
w3 = 0.3  # weight of light rail operator
w4 = 0.1  # weight of project organisation


def objective_municipality(x1, x2):
    """
    Objective for the municipality, describes the development potential as a function of x1 and x2.

    :param x1: number of stations
    :param x2: number of trains per hour
    :return: development potential
    """
    return 500000 * x1 - 25000 * (-1 * x2 + 10)


def objective_inhabitants(x1, x2):
    """
    Objective for the inhabitants, describes the travel time as a function of x1 and x2.

    :param x1: number of stations
    :param x2: number of trains per hour
    :return: travel time
    """
    return 60 * 1 / x2 + 1.5 * x1


def objective_operator(x1, x2):
    """
    Objective for the light rail operator, describes the maintenance costs as a function of x1 and x2.

    :param x1: number of stations
    :param x2: number of trains per hour
    :return: maintenance costs
    """
    return 120000 * x1 + 0.1 * 120000 * (x2 - 10) / 10


def objective_project_team(x1):
    """
    Objective for the project organisation, describes the project duration as a function of x1.

    :param x1: number of stations
    :return: project duration
    """
    return 0.5 * x1


def objective(variables):
    """
    Objective function that is fed to the GA. Calles the separate preference functions that are declared above.

    :param variables: array with design variable values per member of the population. Can be split by using array
    slicing
    :return: 1D-array with aggregated preference scores for the members of the population.
    """
    x1 = variables[:, 0]  # number of stops
    x2 = variables[:, 1]  # number of trains

    # calculate the preference scores
    p_1 = pchip_interpolate([p1_min, p1_mid, p1_max], [0, 100, 60], objective_municipality(x1, x2))
    p_2 = pchip_interpolate([p2_min, p2_mid1, p2_mid2, p2_max], [100, 80, 10, 0], objective_inhabitants(x1, x2))
    p_3 = pchip_interpolate([p3_min, p3_mid, p3_max], [60, 100, 0], objective_operator(x1, x2))
    p_4 = pchip_interpolate([p4_min, p4_mid, p4_max], [100, 95, 0], objective_project_team(x1))

    # aggregate preference scores and return this to the GA
    return solver.request([w1, w2, w3, w4], [p_1, p_2, p_3, p_4])


"""
Below, first the evaluation is done. Since we need the preference curves already for this, we first define the 
coordinates for different preference scores. These can then be used in the pchip interpolation functions.

The variable x_array below contains the coordinates for the corner points of the solution space, ie. the points which 
are being evaluated. This array is then fed to the preference functions to calculate the preference scores of the 
different alternatives per criteria/objective. These are then printed as a pandas DataFrame.

Next, we can aggregate the different preference score per alternative, and print this too to the console. The printed 
DataFrames should contain the same data as table 9 and 10 of the reader resp.
"""

# define coordinates for preference points, to be used in the interpolation function (see also table 7 of the reader)
p1_min, p1_mid, p1_max = [1300000, 4000000, 5250000]
p2_min, p2_mid1, p2_mid2, p2_max = [7.5, 20, 35, 45]
p3_min, p3_mid, p3_max = [350400, 750000, 1212000]
p4_min, p4_mid, p4_max = [1.5, 2, 5]

# define array with coordinates of the corner points
x_array = np.array([
    [3, 2],
    [3, 20],
    [10, 2],
    [10, 20]
])

# calculate the preference scores per alternative per preference function
results_p1 = pchip_interpolate([p1_min, p1_mid, p1_max], [0, 100, 60],
                               objective_municipality(x_array[:, 0], x_array[:, 1]))

results_p2 = pchip_interpolate([p2_min, p2_mid1, p2_mid2, p2_max], [100, 80, 10, 0],
                               objective_inhabitants(x_array[:, 0], x_array[:, 1]))

results_p3 = pchip_interpolate([p3_min, p3_mid, p3_max], [60, 100, 0], objective_operator(x_array[:, 0], x_array[:, 1]))

results_p4 = pchip_interpolate([p4_min, p4_mid, p4_max], [100, 95, 0], objective_project_team(x_array[:, 0]))

# print preference scores as pandas DataFrame (see also table 9 of the reader)
alternatives = ['X1 = 3; X2= 2',
                'X1 = 3; X2= 20',
                'X1 = 10; X2= 2',
                'X1 = 10; X2= 20'
                ]
data = {'Alternatives': alternatives, 'P1': np.round_(results_p1), 'P2': np.round_(results_p2),
        'P3': np.round_(results_p3), 'P4': np.round_(results_p3)}
df = pd.DataFrame(data)
print(df)
print()

# aggregate the preference scores and print it (see also table 10 of the reader)
# For getting the scores, we just call the objective function instead of using the data calculated above (lines 123-131)
aggregation_results = objective(x_array)
data = {'Alternatives': alternatives, 'rank': ss.rankdata(aggregation_results, method='min'),
        'Aggregated scores': np.round_(np.multiply(aggregation_results, -1), 2)}

df = pd.DataFrame(data)
print(df)
print()

"""
Below, the a priori optimization is performed. For this, we first need to define the bounds of the design variables. 
There are no constraints.

The optimization can be ran multiple times, so you can check the consistency between the runs. The outcomes might differ
a bit, since the GA is stochastic from nature, but the differences should be limited.
"""

# bounds for the 2 variables
b1 = [3, 10]  # x1
b2 = [2, 20]  # x2
bounds = [b1, b2]

# specify the number of runs of the optimization
n_runs = 2

# make dictionary with parameter settings for the GA
print('Run Tetra')
options = {
    'n_bits': 20,
    'n_iter': 400,
    'n_pop': 350,
    'r_cross': 0.8,
    'max_stall': 10,
    'tetra': True,
    'var_type_mixed': ['int', 'real']
}

save_array = list()  # list to save the results from every run to
ga = GeneticAlgorithm(objective=objective, constraints=[], bounds=bounds, options=options)  # initialize GA

# run the GA and print its result
for i in range(n_runs):
    print(f'Initialize run {i + 1}')
    score, design_variables, plot_array = ga.run()

    print(f'Optimal result for x1 = {design_variables[0]} stations and x2 = {round(design_variables[1], 2)} trains')
    save_array.append([design_variables[0], design_variables[1]])

    print(f'Finished run {i + 1}')

"""
Now we have the results, we can make some figures. First, the resulting design variables are plotted into the solution 
space. Secondly, we can plot the preference functions together with the results of the optimizations.
"""

# create figure that shows the results in the solution space
fig, ax = plt.subplots(figsize=(7, 7))
ax.set_xlim((0, 15))
ax.set_ylim((0, 25))
ax.set_xlabel('x1 [Number of stations]')
ax.set_ylabel('x2 [Number of trains per hour]')
ax.set_title('Solution space')

# define corner points of solution space
x_fill = [3, 10, 10, 3]
y_fill = [20, 20, 2, 2]

ax.fill_between(x_fill, y_fill, color='#539ecd', label='Solution space')
ax.scatter(np.array(save_array)[:, 0], np.array(save_array)[:, 1], label='Optimal solution Tetra')
ax.scatter([9], [12], label='As-built', marker='p', color='r')

ax.grid()  # show grid
fig.legend()  # show legend

# arrays for plotting continuous preference curves
c1 = np.linspace(1300000, 5250000)
c2 = np.linspace(7.5, 45)
c3 = np.linspace(350400, 1212000)
c4 = np.linspace(1.5, 5)

# calculate the preference functions
p1 = pchip_interpolate([p1_min, p1_mid, p1_max], [0, 100, 60], c1)
p2 = pchip_interpolate([p2_min, p2_mid1, p2_mid2, p2_max], [100, 80, 10, 0], c2)
p3 = pchip_interpolate([p3_min, p3_mid, p3_max], [60, 100, 0], c3)
p4 = pchip_interpolate([p4_min, p4_mid, p4_max], [100, 95, 0], c4)

# make numpy array of results, to allow for array splicing
variable = np.array(save_array)

# calculate individual preference scores for the results of the GA, to plot them on the preference curves
# first for the optimization results
c1_res = objective_municipality(variable[:, 0], variable[:, 1])
c2_res = objective_inhabitants(variable[:, 0], variable[:, 1])
c3_res = objective_operator(variable[:, 0], variable[:, 1])
c4_res = objective_project_team(variable[:, 0])

p1_res = pchip_interpolate([p1_min, p1_mid, p1_max], [0, 100, 60], c1_res)
p2_res = pchip_interpolate([p2_min, p2_mid1, p2_mid2, p2_max], [100, 80, 10, 0], c2_res)
p3_res = pchip_interpolate([p3_min, p3_mid, p3_max], [60, 100, 0], c3_res)
p4_res = pchip_interpolate([p4_min, p4_mid, p4_max], [100, 95, 0], c4_res)

# and secondly, for the as-built solution, so we can compare the two.
c1_res_actual = objective_municipality(9, 12)
c2_res_actual = objective_inhabitants(9, 12)
c3_res_actual = objective_operator(9, 12)
c4_res_actual = objective_project_team(9)

p1_res_actual = pchip_interpolate([p1_min, p1_mid, p1_max], [0, 100, 60], c1_res_actual)
p2_res_actual = pchip_interpolate([p2_min, p2_mid1, p2_mid2, p2_max], [100, 80, 10, 0], c2_res_actual)
p3_res_actual = pchip_interpolate([p3_min, p3_mid, p3_max], [60, 100, 0], c3_res_actual)
p4_res_actual = pchip_interpolate([p4_min, p4_mid, p4_max], [100, 95, 0], c4_res_actual)

# create figure that plots all preference curves and the preference scores of the returned results of the GA
fig = plt.figure()
ax1 = fig.add_subplot(1, 1, 1)
ax1.plot(c1, p1, zorder=3, label='Preference curve')
ax1.scatter(c1_res, p1_res, label='Optimal solution Tetra', color='tab:purple', zorder=1)
ax1.scatter(c1_res_actual, p1_res_actual, label='As built', marker='p', color='r', zorder=1)
ax1.set_xlim((0, 5500000))
ax1.set_ylim((0, 100))
ax1.set_title('Municipality')
ax1.set_xlabel('Developer potential [€]')
ax1.set_ylabel('Preference score')
ax1.legend()
ax1.grid()

fig = plt.figure()
ax2 = fig.add_subplot(1, 1, 1)
ax2.plot(c2, p2, zorder=3, label='Preference curve')
ax2.scatter(c2_res, p2_res, label='Optimal solution Tetra', color='tab:purple', zorder=1)
ax2.scatter(c2_res_actual, p2_res_actual, label='As built', marker='p', color='tab:red', zorder=1)
ax2.set_xlim((0, 60))
ax2.set_ylim((0, 100))
ax2.set_title('Users and inhabitants')
ax2.set_xlabel('Travel time [min]')
ax2.set_ylabel('Preference score')
ax2.legend()
ax2.grid()

fig = plt.figure()
ax3 = fig.add_subplot(1, 1, 1)
ax3.plot(c3, p3, zorder=3, label='Preference curve')
ax3.scatter(c3_res, p3_res, label='Optimal solution Tetra', color='tab:purple', zorder=1)
ax3.scatter(c3_res_actual, p3_res_actual, label='As built', marker='p', color='tab:red', zorder=1)
ax3.set_xlim((0, 1500000))
ax3.set_ylim((0, 100))
ax3.set_title('Light rail operator')
ax3.set_xlabel('Operational costs [€]')
ax3.set_ylabel('Preference score')
ax3.legend()
ax3.grid()

fig = plt.figure()
ax4 = fig.add_subplot(1, 1, 1)
ax4.plot(c4, p4, zorder=3, label='Preference curve')
ax4.scatter(c4_res, p4_res, label='Optimal solution Tetra', color='tab:purple')
ax4.scatter(c4_res_actual, p4_res_actual, label='As built', marker='p', color='tab:red')
ax4.set_xlim((0, 7))
ax4.set_ylim((0, 100))
ax4.set_title('Project organisation')
ax4.set_xlabel('Building time [years]')
ax4.set_ylabel('Preference score')
ax4.legend()
ax4.grid()

plt.show()
