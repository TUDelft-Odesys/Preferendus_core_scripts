"""
Python code for the German high voltage power system line exemplar
"""

import numpy as np
import pandas as pd
import scipy.stats as ss
import matplotlib.pyplot as plt
from scipy.interpolate import pchip_interpolate
from scipy.optimize import fsolve

from genetic_algorithm_pfm import GeneticAlgorithm
from genetic_algorithm_pfm.tetra_pfm import TetraSolver

"""
This script contains the code to run both the a posteriori evaluation as the a priori optimization of the German high 
voltage power system ine example, described in chapter 5 of the reader.
"""

# Initialize TetraSolver
solver = TetraSolver()

# define overall system length
LOA = 700

# x_points: the outcomes of the objective for which a preference score is defined by the stakeholders
# p_points: the corresponding preference scores
x_points_1, p_points_1 = [[500, 600, 800], [100, 50, 0]]
x_points_2, p_points_2 = [[20, 130, 200], [100, 50, 0]]
x_points_3, p_points_3 = [[1300, 1450, 1700], [100, 40, 0]]

# set weights for the different objectives
w1 = 0.40  # weight of the costs objective
w2 = 0.20  # weight of the area objective
w3 = 0.40  # weight of the project duration objective

"""
Part of the area objective is the noise of the power line. Below, the required distance from the powerline to an 
observer is calculated by using a solver from scipy, called fsolve. This solver will find the input variables for which 
the returned value of the function is equal to 0.
"""


def objective_noise(d, dc: bool):
    """
    Function to calculate the noise emissions of a high voltage power line, to be used by the fsolve algorithm of scipy.

    Calculation and parameters are based on "FORMULAS FOR PREDICTING AUDIBLE NOISE FROM OVERHEAD HIGH VOLTAGE AC AND DC
    LINES" by Fellow & Member, 1981

    :param d: distance over ground to the nearest power line
    :param dc: ac or dc system (DC = True)
    :return: difference between target value 'db' and the calculated noise level
    """
    height = 12  # height of power lines above the ground
    distance_per_line = 20  # distance between power lines, relevant for AC systems

    db = 45  # target noise level

    if dc:  # DC system
        e = 22  # kV/cm
        deq = 109.71  # mm
        noise_level = -133.4 + 86 * np.log10(e) + 40 * np.log10(deq) - 11.4 * np.log10(np.sqrt(d ** 2 + height ** 2))
    else:  # AC system
        e_inner_phase = 14.58  # kV/cm
        e_outer_phase = 13.66  # kV/cm
        deq = 69.4  # mm
        pwl_inner = -164.6 + 120 * np.log10(e_inner_phase) + 55 * np.log10(deq)
        pwl_outer = -164.6 + 120 * np.log10(e_outer_phase) + 55 * np.log10(deq)

        temp_outer_close = (pwl_outer - 11.4 * np.log10(np.sqrt(d ** 2 + height ** 2)) - 5.8) / 10
        temp_inner = (pwl_inner - 11.4 * np.log10(np.sqrt((d + distance_per_line) ** 2 + height ** 2)) - 5.8) / 10
        temp_outer_far = (pwl_outer - 11.4 * np.log10(
            np.sqrt((d + 2 * distance_per_line) ** 2 + height ** 2)) - 5.8) / 10

        noise_level = 10 * np.log10(10 ** temp_inner + 10 ** temp_outer_close + 10 ** temp_outer_far)
    return noise_level - db


# find minimal needed distance over ground to stay under 45 dB(A) noise emission
d_ac = fsolve(objective_noise, np.array([10]), (False,))
d_dc = fsolve(objective_noise, np.array([10]), (True,))


def objective_costs(x1, x2):
    """
    Objective for the costs.

    :param x1: type of power system (True is AC)
    :param x2: length underground
    :return: development potential
    """
    # initialize array wit results
    costs = np.zeros(len(x1))

    # set results for AC system
    mask_ac = x1 == 1  # identify members in system containing AC system parameters
    acu = x2[mask_ac]
    aco = LOA - acu
    costs[mask_ac] = 0.475 * aco + 0.580 * acu + 375

    # set results for DC system
    mask_dc = np.invert(mask_ac)  # identify members in system containing DC system parameters
    dcu = x2[mask_dc]
    dco = LOA - dcu
    costs[mask_dc] = 0.120 * dco + 0.190 * dcu + 430

    return costs


def objective_area(x1, x2):
    """
    Objective for the area.

    :param x1: type of power system (True is AC)
    :param x2: length underground
    :return: development potential
    """
    # initialize array wit results
    area = np.zeros(len(x1))

    # set results for AC system
    mask_ac = x1 == 1  # identify members in system containing AC system parameters
    acu = x2[mask_ac]
    aco = LOA - acu
    area[mask_ac] = (0.170 + d_ac[0] / 1000) * aco + 0.018 * acu

    # set results for DC system
    mask_dc = np.invert(mask_ac)  # identify members in system containing DC system parameters
    dcu = x2[mask_dc]
    dco = LOA - dcu
    area[mask_dc] = (0.120 + d_dc[0] / 1000) * dco + 0.015 * dcu

    return area


def objective_duration(x1, x2):
    """
    Objective for the project duration.

    :param x1: type of power system (True is AC)
    :param x2: length underground
    :return: development potential
    """
    # initialize array wit results
    duration = np.zeros(len(x1))

    # set results for AC system
    mask_ac = x1 == 1  # identify members in system containing AC system parameters
    acu = x2[mask_ac]
    aco = LOA - acu
    duration[mask_ac] = 2.5 * aco + 2.6 * acu

    # set results for DC system
    mask_dc = np.invert(mask_ac)  # identify members in system containing DC system parameters
    dcu = x2[mask_dc]
    dco = LOA - dcu
    duration[mask_dc] = 1.8 * dco + 2.3 * dcu

    return duration


def objective(variabels):
    """
    Objective function that is fed to the GA. Calles the separate preference functions that are declared above.
    Objective can be used both with IMAP as with the minmax aggregation method. Declare which to use by the method
    argument.

    :param variabels: array with design variable values per member of the population. Can be split by using array
    slicing
    :return: 1D-array with aggregated preference scores for the members of the population.
    """
    x1 = variabels[:, 0]  # which power system, bool (True = AC)
    x2 = variabels[:, 1]  # kilometers of underground cable in system

    # calculate objectives
    costs = objective_costs(x1, x2)
    area = objective_area(x1, x2)
    duration = objective_duration(x1, x2)

    # calculate preference scores based on objective values
    p_costs = pchip_interpolate(x_points_1, p_points_1, costs)
    p_area = pchip_interpolate(x_points_2, p_points_2, area)
    p_time = pchip_interpolate(x_points_3, p_points_3, duration)

    return [w1, w2, w3], [p_costs, p_area, p_time]


"""
Below, first the evaluation is done. The variable x_array below contains the coordinates for the corner points of the 
design space, ie. the points which are being evaluated. This array is then fed to the preference functions to 
calculate the preference scores of the different alternatives per criteria/objective. These are then printed as a pandas
DataFrame.

Next, we can aggregate the different preference score per alternative, and print this too to the console. The printed 
DataFrames should contain the same data as table 13 and 14 of the reader resp.
"""

# define array with coordinates of the corner points. Note that this only contains a boolean value for the used power
# system and the underground length. The overground length is calculated by using the overall length of the system
x_array = np.array([
    [1, 300],
    [1, 600],
    [0, 300],
    [0, 600]
])

# calculate the preference scores per alternative per preference function
results_p1 = pchip_interpolate(x_points_1, p_points_1, objective_costs(x_array[:, 0], x_array[:, 1]))
results_p2 = pchip_interpolate(x_points_2, p_points_2, objective_area(x_array[:, 0], x_array[:, 1]))
results_p3 = pchip_interpolate(x_points_3, p_points_3, objective_duration(x_array[:, 0], x_array[:, 1]))

# print preference scores as pandas DataFrame (see also table 13 of the reader)
alternatives = ['AC – 400 ACO – 300 ACU',
                'AC – 100 ACO – 600 ACU',
                'DC – 400 DCO – 300 DCU',
                'DC – 100 DCO – 600 DCU'
                ]
data = {'Alternatives': alternatives, 'P1': np.round_(results_p1), 'P2': np.round_(results_p2),
        'P3': np.round_(results_p3)}
df = pd.DataFrame(data)
print(df)
print()

# aggregate the preference scores and print it (see also table 10 of the reader)
# For getting the scores, we just call the objective function instead of using the data calculated above (lines 217-219)
w, p = objective(x_array)
aggregation_results = solver.request(w, p)
data = {'Alternatives': alternatives, 'rank': ss.rankdata(aggregation_results),
        'Aggregated scores': np.round_(np.multiply(aggregation_results, -1), 2)}

df = pd.DataFrame(data)
print(df)
print()

"""
Below, the a priori optimization is performed. For this, we first need to define the bounds of the design variables. 
There are no constraints.

Two runs are made with the GA: the first with the IMAP solver, the second with the minmax solver. Both require a 
different configuration of the GA, so you will see two different dictionaries called 'options', one for each run. For 
more information about the different options, see the docstring of GeneticAlgorithm (via help()) or chapter 4 of the 
reader.

The optimization can be ran multiple times, so you can check the consistency between the runs. The outcomes might differ
a bit, since the GA is stochastic from nature, but the differences should be limited.
"""

# bounds for the 2 variables
bounds = [[0, 1], [300, 600]]

# specify the number of runs of the optimization
n_runs = 2

# run IMAP version
print('Run IMAP')
options = {  # make dictionary with parameter settings for the GA
    'n_bits': 20,
    'n_iter': 400,
    'n_pop': 150,
    'r_cross': 0.85,
    'max_stall': 10,
    'aggregation': 'tetra',
    'var_type_mixed': ['bool', 'real']
}

save_array_IMAP = list()  # list to save the results from every run to
ga = GeneticAlgorithm(objective=objective, constraints=[], bounds=bounds, options=options)

# run the GA and print its result
for i in range(n_runs):
    print(f'Initialize run {i + 1}')
    score, design_variables_IMAP, _ = ga.run()

    print(
        f"Optimal result for type = {'AC' if design_variables_IMAP[0] else 'DC'}, length underground = "
        f"{round(design_variables_IMAP[1], 2)}km, and length overground = "
        f"{LOA - round(design_variables_IMAP[1], 2)}km")
    print()

    save_array_IMAP.append([design_variables_IMAP[0], design_variables_IMAP[1]])
    print(f'Finished run {i + 1}')

# run MinMax version
print('Run MinMax')
options = {  # make dictionary with parameter settings for the GA
    'n_bits': 24,
    'n_iter': 400,
    'n_pop': 250,
    'r_cross': 0.9,
    'max_stall': 10,
    'aggregation': 'minmax',
    'var_type_mixed': ['bool', 'real']
}

save_array_minmax = list()
ga = GeneticAlgorithm(objective=objective, constraints=[], bounds=bounds, options=options)

# run the GA and print its result
for i in range(n_runs):
    print(f'Initialize run {i + 1}')
    score, design_variables_minmax, _ = ga.run()

    print(f"Optimal result for type = {'AC' if design_variables_minmax[0] else 'DC'}, length underground = "
          f"{round(design_variables_minmax[1], 2)}km, and length overground = "
          f"{LOA - round(design_variables_minmax[1], 2)}km")
    print()

    save_array_minmax.append([design_variables_minmax[0], design_variables_minmax[1]])
    print(f'Finished run {i + 1}')

"""
Now we have the results, we can make some figures. First, the resulting design variables are plotted into the solution 
space. Secondly, we can plot the preference functions together with the results of the optimizations.
"""

# create figure that shows the results in the design space
fig, ax = plt.subplots()
ax.set_xlim((250, 650))
ax.set_ylim((-1, 2))
ax.set_yticks([0, 1])
ax.set_ylabel('AC / DC')
ax.set_xlabel('Length underground cable [km]')
ax.set_title('Design space')

ax.scatter(np.array(save_array_IMAP)[:, 1], np.array(save_array_IMAP)[:, 0], label='Optimal solution IMAP',
           color='tab:purple')
ax.scatter(np.array(save_array_minmax)[:, 1], np.array(save_array_minmax)[:, 0], label='Optimal solution MinMax',
           marker='^', color='tab:orange')
plt.hlines(y=1, xmin=300, xmax=600, label='Design space')
plt.hlines(0, 300, 600)

ax.grid()  # show grid
ax.legend()  # show legend

# arrays for plotting continuous preference curves
c1 = np.linspace(500, 800)  # costs
c2 = np.linspace(20, 200)  # area
c3 = np.linspace(1300, 1700)  # time

# calculate the preference functions
p1 = pchip_interpolate(x_points_1, p_points_1, c1)
p2 = pchip_interpolate(x_points_2, p_points_2, c2)
p3 = pchip_interpolate(x_points_3, p_points_3, c3)

# make numpy array of results, to allow for array splicing
variable_t = np.array(save_array_IMAP)
variable_mm = np.array(save_array_minmax)

# calculate individual preference scores for the results of the GA, to plot them on the preference curves
# first for the results with IMAP
c1_res_t = objective_costs(variable_t[:, 0], variable_t[:, 1])
c2_res_t = objective_area(variable_t[:, 0], variable_t[:, 1])
c3_res_t = objective_duration(variable_t[:, 0], variable_t[:, 1])

p1_res_t = pchip_interpolate(x_points_1, p_points_1, c1_res_t)
p2_res_t = pchip_interpolate(x_points_2, p_points_2, c2_res_t)
p3_res_t = pchip_interpolate(x_points_3, p_points_3, c3_res_t)

# and secondly, for the results with MinMax
c1_res_mm = objective_costs(variable_t[:, 0], variable_t[:, 1])
c2_res_mm = objective_area(variable_t[:, 0], variable_t[:, 1])
c3_res_mm = objective_duration(variable_t[:, 0], variable_t[:, 1])

p1_res_mm = pchip_interpolate(x_points_1, p_points_1, c1_res_mm)
p2_res_mm = pchip_interpolate(x_points_2, p_points_2, c2_res_mm)
p3_res_mm = pchip_interpolate(x_points_3, p_points_3, c3_res_mm)

# create figure that plots all preference curves and the preference scores of the returned results of the GA
fig = plt.figure()
ax1 = fig.add_subplot(1, 1, 1)
ax1.plot(c1, p1, label='Preference curve')
ax1.scatter(c1_res_t, p1_res_t, label='Optimal solution IMAP', color='tab:purple')
ax1.scatter(c1_res_mm, p1_res_mm, label='Optimal solution MinMax', marker='^', color='tab:orange')
ax1.set_ylim((0, 100))
ax1.set_title('Costs')
ax1.set_xlabel('Costs of installation [€*1e6]')
ax1.set_ylabel('Preference score')
ax1.grid()
ax1.legend()

fig = plt.figure()
ax2 = fig.add_subplot(1, 1, 1)
ax2.plot(c2, p2, label='Preference curve')
ax2.scatter(c2_res_t, p2_res_t, label='Optimal solution IMAP', color='tab:purple')
ax2.scatter(c2_res_mm, p2_res_mm, label='Optimal solution MinMax', marker='^', color='tab:orange')
ax2.set_ylim((0, 100))
ax2.set_title('Area')
ax2.set_xlabel(r'Required area [km$^2$]')
ax2.set_ylabel('Preference score')
ax2.grid()
ax2.legend()

fig = plt.figure()
ax3 = fig.add_subplot(1, 1, 1)
ax3.plot(c3, p3, label='Preference curve')
ax3.scatter(c3_res_t, p3_res_t, label='Optimal solution IMAP', color='tab:purple')
ax3.scatter(c3_res_mm, p3_res_mm, label='Optimal solution MinMax', marker='^', color='tab:orange')
ax3.set_ylim((0, 100))
ax3.set_title('Project duration')
ax3.set_xlabel('Duration [days]')
ax3.set_ylabel('Preference score')
ax3.grid()
ax3.legend()

plt.show()
