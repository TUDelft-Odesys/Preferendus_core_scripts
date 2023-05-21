"""
Python code for the shopping mall design problem revisited non-linear example (Chapter 7.2)
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import pchip_interpolate

from genetic_algorithm_pfm import GeneticAlgorithm

"""
This script is fairly similar to the non-linear shopping mall example. Only the preference functions for objective 1 and
 2 are changed, together with the weights.

Note that the non-linear preference curves are created by using an interpolation function called pchip_interpolation.
"""

# set weights for the different objectives
w1 = 0.25
w2 = 0.50
w3 = 0.25


def objective_p1(x1, x2):
    """
    Objective to maximize the profit preference.

    :param x1: 1st design variable
    :param x2: 2nd design variable
    """
    return pchip_interpolate([240000, 450000, 1400000], [0, 65, 100], (160 * x1 + 80 * x2))


def objective_p2(x1, x2):
    """
    Objective to maximize the CO2 emission preference.

    :param x1: 1st design variable
    :param x2: 2nd design variable
    """
    return pchip_interpolate([90000, 200000, 750000], [100, 30, 0], (120 * x1 + 30 * x2))


def objective_p3(x1, x2):
    """
    Objective to maximize the shopping potential preference.

    :param x1: 1st design variable
    :param x2: 2nd design variable
    """
    return pchip_interpolate([45000, 200000, 360000], [0, 80, 100], (15 * x1 + 45 * x2))


def objective(variables):
    """
    Objective function that is fed to the GA. Calles the separate preference functions that are declared above.

    :param variables: array with design variable values per member of the population. Can be split by using array
    slicing
    :return: 1D-array with aggregated preference scores for the members of the population.
    """
    # extract 1D design variable arrays from full 'variables' array
    x1 = variables[:, 0]
    x2 = variables[:, 1]

    # calculate the preference scores
    p_1 = objective_p1(x1, x2)
    p_2 = objective_p2(x1, x2)
    p_3 = objective_p3(x1, x2)

    # aggregate preference scores and return this to the GA
    return [w1, w2, w3], [p_1, p_2, p_3]

"""
Before we can run the optimization, we finally need to define the constraints and bounds
"""

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

# set bounds for all variables
b1 = [0, 5000]  # x1
b2 = [0, 7000]  # x2
bounds = [b1, b2]

"""
Now we have everything for the optimization, we can run it. For more information about the different options to 
configure the GA, see the docstring of GeneticAlgorithm (via help()) or chapter 4 of the reader.
"""

# make dictionary with parameter settings for the GA run with the IMAP solver
options = {
    'n_bits': 8,
    'n_iter': 400,
    'n_pop': 500,
    'r_cross': 0.8,
    'max_stall': 8,
    'aggregation': 'tetra',
    'var_type': 'real'
}

# run the GA and print its result
print(f'Run GA with IMAP')
ga = GeneticAlgorithm(objective=objective, constraints=cons, bounds=bounds, options=options)
score_IMAP, design_variables_IMAP, _ = ga.run()

print(f'Optimal result for x1 = {round(design_variables_IMAP[0], 2)}m2 and '
      f'x2 = {round(design_variables_IMAP[1], 2)}m2 (sum = {round(sum(design_variables_IMAP))}m2)')

"""
Now we have the results, we can make some figures. First, the resulting design variables are plotted into the solution 
space. Secondly, we can plot the preference functions together with the results of the optimizations.
"""

# create figure that shows the results in the design space
fig, ax = plt.subplots(figsize=(7, 7))
ax.set_xlim((-200, 9000))
ax.set_ylim((0, 9000))
ax.set_xlabel('x1 [m2]')
ax.set_ylabel('x2 [m2]')
ax.set_title('Design space')

# define corner points of design space
x_fill = [0, 3000, 5000, 5000, 3000, 0]
y_fill = [7000, 7000, 5000, 0, 0, 3000]

ax.fill_between(x_fill, y_fill, color='#539ecd', label='Design space')
ax.scatter(design_variables_IMAP[0], design_variables_IMAP[1], color='tab:purple', label='Optimal solution IMAP')

ax.grid()  # show grid
fig.legend()  # show legend

# create arrays for plotting continuous preference curves
c1 = np.linspace(240000, 1200000)
c2 = np.linspace(90000, 750000)
c3 = np.linspace(45000, 360000)

# calculate the preference functions
p1 = pchip_interpolate([240000, 450000, 1400000], [0, 65, 100], c1)
p2 = pchip_interpolate([90000, 200000, 750000], [100, 30, 0], c2)
p3 = pchip_interpolate([45000, 200000, 360000], [0, 80, 100], c3)

# calculate individual preference scores for the results of the GA, to plot them on the preference curves
c1_res = (160 * design_variables_IMAP[0] + 80 * design_variables_IMAP[1])
p1_res = pchip_interpolate([240000, 450000, 1400000], [0, 65, 100], c1_res)

c2_res = (120 * design_variables_IMAP[0] + 30 * design_variables_IMAP[1])
p2_res = pchip_interpolate([90000, 200000, 750000], [100, 30, 0], c2_res)

c3_res = (15 * design_variables_IMAP[0] + 45 * design_variables_IMAP[1])
p3_res = pchip_interpolate([45000, 200000, 360000], [0, 80, 100], c3_res)

# create figure that plots all preference curves and the preference scores of the returned results of the GA
fig = plt.figure()
ax1 = fig.add_subplot(1, 1, 1)
ax1.plot(c1, p1, label='Preference curve')
ax1.scatter(c1_res, p1_res, label='Optimal solution IMAP', color='tab:purple')
ax1.set_xlim((0, 1200000))
ax1.set_ylim((0, 102))
ax1.set_title('Profit')
ax1.set_xlabel('Profit [€]')
ax1.set_ylabel('Preference score')
ax1.grid()
ax1.legend()

fig = plt.figure()
ax2 = fig.add_subplot(1, 1, 1)
ax2.plot(c2, p2, label='Preference curve')
ax2.scatter(c2_res, p2_res, label='Optimal solution IMAP', color='tab:purple')
ax2.set_xlim((0, 750000))
ax2.set_ylim((0, 102))
ax2.set_title('CO2 Emission')
ax2.set_xlabel('Emissions [kg]')
ax2.set_ylabel('Preference score')
ax2.grid()
ax2.legend()

fig = plt.figure()
ax3 = fig.add_subplot(1, 1, 1)
ax3.plot(c3, p3, label='Preference curve')
ax3.scatter(c3_res, p3_res, label='Optimal solution IMAP', color='tab:purple')
ax3.set_xlim((0, 365000))
ax3.set_ylim((0, 102))
ax3.set_title('Shopping potential')
ax3.set_xlabel('Shopping potential [people]')
ax3.set_ylabel('Preference score')
ax3.grid()
ax3.legend()

plt.show()
