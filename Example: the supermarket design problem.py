"""
Python code for the supermarket design problem example
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import pchip_interpolate

from genetic_algorithm_pfm import GeneticAlgorithm
from tetra_pfm import TetraSolver
from weighted_minmax.aggregation_algorithm import aggregate_max

"""
This script is fairly similar to the shopping mall examples. Since the objective functions are a bit more complex, the 
code is this too. However, in the basis it is no different from what you have seen before.

In this script, mask arrays are used. This is a way to efficiently check if members of an array fulfill a statement. For
example:

    x = [5, 6, 8, 9]
    mask = x > 7

will result in the same as:

    x = [5, 6, 8, 9]
    mask = []
    for item in x:
        mask.append(item > 7)

namely: [False, False, True, True]. This can then be used to set a part of the array to a different value:

    x[mask] = 10

This will result in x = [10, 10, 8, 9]. 

The fist method is more efficient and thus faster, the latter is easier to understand. Remember: the result is the same,
so use the method you are most comfortable with!

Note that for the mask = x > 7 to work, x must be a numpy array. To learn more about this topic, see: 
https://jakevdp.github.io/PythonDataScienceHandbook/02.06-boolean-arrays-and-masks.html
"""

# Initialize TetraSolver
solver = TetraSolver()

# set weights for the different objectives
w1 = 0.65
w2 = 0.35


def objective_p1(x1, x2):
    """
    Objective to maximize the shopping added value

    :param x1: 1st design variable
    :param x2: 2nd design variable
    """
    normalized_x1 = 1 - (x1 - 100) / (1000 - 100)
    normalized_x2 = (x2 - 800) / (30_000 - 800)

    ret = 100 * (np.sqrt(normalized_x1 ** 2 + normalized_x2) ** 2)

    mask = ret > 100
    ret[mask] = 0

    return ret


def objective_p2(x1, x2):
    """
    Objective that describes the relative sustainability.

    :param x1: 1st design variable
    :param x2: 2nd design variable
    """
    return pchip_interpolate([0, 1.2, 2], [0, 100, 60], x2 / (20_000 * x1 / 400))


def objective(variables, method='tetra'):
    """
    Objective function that is fed to the GA. Calles the separate preference functions that are declared above.
    Objective can be used both with Tetra as with the minmax aggregation method. Declare which to use by the method
    argument.

    :param variables: array with design variable values per member of the population. Can be split by using array
    slicing
    :param method: which aggregation method to use: 'tetra' or 'minmax'. Defaults to 'tetra'
    :return: 1D-array with aggregated preference scores for the members of the population.
    """
    # extract 1D design variable arrays from full 'variables' array
    x1 = variables[:, 0]
    x2 = variables[:, 1]

    # calculate the preference scores
    p_1 = objective_p1(x1, x2)
    p_2 = objective_p2(x1, x2)

    # check if there are no preference scores < 0 or > 100
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


# there are no constraints
cons = []

# set bounds for all variables
b1 = [100, 1000]  # x1
b2 = [800, 30_000]  # x2
bounds = [b1, b2]

"""
Now we have everything for the optimization, we can run it. Two runs are made with the GA: the first with the Tetra 
solver, the second with the minmax solver. Both require a different configuration of the GA, so you will see two 
different dictionaries called 'options', one for each run. For more information about the different options, see the 
docstring of GeneticAlgorithm (via help()) or chapter 4 of the reader.
"""

# make dictionary with parameter settings for the GA run with the Tetra solver
options = {
    'n_bits': 16,
    'n_iter': 400,
    'n_pop': 1100,
    'r_cross': 0.80,
    'max_stall': 10,
    'tetra': True,
    'var_type_mixed': ['real', 'int']
}

# run the GA and print its result
print(f'Run GA with Tetra')
ga = GeneticAlgorithm(objective=objective, constraints=cons, bounds=bounds, options=options)
score_tetra, design_variables_tetra, _ = ga.run()

print(f'Optimal result for a distance of {round(design_variables_tetra[0], 2)} meters and '
      f'{design_variables_tetra[1]} products')

# make dictionary with parameter settings for the GA run with the minmax solver
options = {
    'n_bits': 24,
    'n_iter': 400,
    'n_pop': 1000,
    'r_cross': 0.75,
    'max_stall': 15,
    'tetra': False,
    'var_type_mixed': ['real', 'int']
}

# run the GA and print its result
print(f'Initialize run with MinMax')
ga = GeneticAlgorithm(objective=objective, constraints=cons, bounds=bounds, options=options, args=('minmax',))
score_minmax, design_variables_minmax, _ = ga.run()

print(f'Optimal result for a distance of {round(design_variables_minmax[0], 2)} meters and '
      f'{design_variables_minmax[1]} products')

"""
Now we have the results, we can make some figures. First, the resulting design variables are plotted into the solution 
space. Secondly, we can plot the preference functions together with the results of the optimizations.
"""

# create figure that shows the results in the solution space
fig, ax = plt.subplots(figsize=(7, 7))
ax.set_xlim((0, 1200))
ax.set_ylim((600, 32_000))
ax.set_xlabel('x1 [m]')
ax.set_ylabel('x2 [-]')
ax.set_title('Solution space')

# define corner points of solution space
x_fill = [100, 1000, 1000, 100]
y_fill = [30_000, 30_000, 800, 800]

ax.fill_between(x_fill, y_fill, color='#539ecd', label='Solution space')
ax.scatter(design_variables_tetra[0], design_variables_tetra[1], label='Optimal solution Tetra', color='tab:purple')
ax.scatter(design_variables_minmax[0], design_variables_minmax[1], label='Optimal solution MinMax', marker='*',
           color='tab:orange')

ax.grid()  # show grid
fig.legend()  # show legend

# create arrays for plotting continuous preference curves
c1 = np.linspace(0, 2, 100)
c2 = np.linspace(0, 2, 100)

# calculate the preference functions
norm_1 = np.linspace(0, 1, 100)
norm_2 = np.linspace(0, 1, 100)
N1, N2 = np.meshgrid(norm_1, norm_2)

p1 = 100 * (np.sqrt(N1 ** 2 + N2) ** 2)
m = p1 > 100
p1[m] = 0
p2 = pchip_interpolate([0, 1.2, 2], [0, 100, 60], c2)

# calculate individual preference scores for the results of the GA, to plot them on the preference curves
temp_tetra = np.zeros((2, 2))  # workaround to fix problem with masking a float
temp_tetra[0] = design_variables_tetra

normalized_c1x1 = 1 - (design_variables_tetra[0] - 100) / (1000 - 100)
normalized_c1x2 = (design_variables_tetra[1] - 800) / (30_000 - 800)

c2_res = design_variables_tetra[1] / (20_000 * design_variables_tetra[0] / 400)

p1_res = objective_p1(temp_tetra[:, 0], temp_tetra[:, 1])[0]
p2_res = objective_p2(design_variables_tetra[0], design_variables_tetra[1])

temp_minmax = np.zeros((2, 2))  # workaround to fix problem with masking a float
temp_minmax[0] = design_variables_minmax

normalized_c1x1_mm = 1 - (design_variables_minmax[0] - 100) / (1000 - 100)
normalized_c1x2_mm = (design_variables_minmax[1] - 800) / (30_000 - 800)

c2_res_mm = design_variables_minmax[1] / (20_000 * design_variables_minmax[0] / 400)

p1_res_mm = objective_p1(temp_minmax[:, 0], temp_minmax[:, 1])[0]
p2_res_mm = objective_p2(design_variables_minmax[0], design_variables_minmax[1])

# create figure that plots all preference curves and the preference scores of the returned results of the GA
fig = plt.figure()
ax2 = fig.add_subplot(1, 1, 1)
ax2.plot(c2, p2, label='Preference curve')
ax2.scatter(c2_res, p2_res, label='Optimal solution Tetra', color='tab:purple')
ax2.scatter(c2_res_mm, p2_res_mm, label='Optimal solution MinMax', marker='*', color='tab:orange')
ax2.set(xlabel='Sustainability index', ylabel='Preference')
ax2.set_title('Preference Curve Transport Sustainability & Wasted')
ax2.legend()
ax2.grid()

fig = plt.figure()
ax1 = fig.add_subplot(1, 1, 1, projection='3d')
surf = ax1.plot_surface(N1, N2, p1, label='Preference curve')
ax1.scatter(normalized_c1x1, normalized_c1x2, p1_res, label='Optimal solution Tetra', color='tab:purple')
ax1.scatter(normalized_c1x1_mm, normalized_c1x2_mm, p1_res_mm, label='Optimal solution MinMax', marker='*',
            color='tab:orange')
ax1.set(xlabel='Normalized travel distance', ylabel='Normalized items in assortiment', zlabel='Preference')
ax1.set_title('Preference Curves Shopping Added Value')
ax1.view_init(elev=15, azim=160)
surf._facecolors2d = surf._facecolor3d
surf._edgecolors2d = surf._edgecolor3d
ax1.legend()
ax1.grid()

fig = plt.figure()
ax3 = fig.add_subplot(1, 1, 1)
fig = ax3.imshow(p1, cmap='GnBu', interpolation='nearest')
ax3.scatter(normalized_c1x1 * 100, normalized_c1x2 * 100, label='Optimal solution Tetra', color='tab:purple')
ax3.scatter(normalized_c1x1_mm * 100, normalized_c1x2_mm * 100, label='Optimal solution MinMax', marker='*',
            color='tab:orange')
ax3.set(xlabel='Normalized travel distance', ylabel='Normalized items in assortiment')
ax3.set_title('Preference Curves Shopping Added Value')
ax3.legend()
ax3.grid()

plt.colorbar(fig, ax=ax3, location='left')

plt.show()
