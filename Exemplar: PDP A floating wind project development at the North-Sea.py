"""
Python code for the floating wind project development at the North-Sea exemplar
"""

import pathlib
from math import ceil

import matplotlib.pyplot as plt
import numpy as np
from numpy import pi
from scipy.interpolate import pchip_interpolate

from genetic_algorithm_pfm import GeneticAlgorithm
from tetra_pfm import TetraSolver

"""
This script contains the code to run the  a priori optimization of the Offshore floating wind exemplar, described in 
chapter 5 of the reader.
"""

# tell the script where it is
HERE = pathlib.Path(__file__).parent

# set some variables
density_seawater = 1025  # kg/m3
max_t = 3800  # tension on anchor in kN
n_anchors = 108  # number of anchors to install

Su = 70  # undrained shear strength of clat in kPa
Np = 10  # lateral bearing capacity factor

# set some characteristics of the vessels
ship_options = {
    'OCV small': {
        'day_rate': 47000,  # €/day
        'deck_space': 8,  # max anchors on deck
        'max_available': 3,  # how many are available for the project
        'CO2_emission': 30  # tonnes per day
    },
    'OCV big': {
        'day_rate': 55000,  # €/day
        'deck_space': 12,  # max anchors on deck
        'max_available': 2,  # how many are available for the project
        'CO2_emission': 40  # tonnes per day
    },
    'Barge': {
        'day_rate': 35000,  # €/day
        'deck_space': 16,  # max anchors on deck
        'max_available': 2,  # how many are available for the project
        'CO2_emission': 35  # tonnes per day
    }
}

time_installation = 1  # time it takes to install one anchor
time_bunkering = 2  # time it takes to get a new set of anchors onboard

# set weights for the different objectives
w1 = 0.25  # project duration
w2 = 0.35  # costs
w3 = 0.15  # fleet utilization
w4 = 0.25  # CO2 emissions

# x_points: the outcomes of the objective for which a preference score is defined by the stakeholders
# p_points: the corresponding preference scores
x_points_1, p_points_1 = [[44, 80, 200], [100, 40, 0]]
x_points_2, p_points_2 = [[9_000_000, 12_000_000, 17_000_000], [100, 30, 0]]
x_points_3, p_points_3 = [[1, 3, 7], [0, 50, 100]]
x_points_4, p_points_4 = [[1400, 20000, 55000], [100, 40, 0]]

# Initialize TetraSolver
solver = TetraSolver()


def objective_time(ocv_s, ocv_l, barge):
    """
    This function contains a simpel discrete event simulation to determine the time it takes to execute the project.
    It depends on the number of vessels, anchors to install, and the time it takes to install an anchor and to reload
    them.

    The simulation is done in a while loop. An infinite loop is prevented by the so-called inf_loop_prevent.

    If the fleet consists of no vessels, the time is taken as 1e6 for all vessel types. A constraint should take care of
    the entry further (i.e. a fleet of zero vessels should be considered as infeasible).

    :param ocv_s: number of small offshore construction vessels
    :param ocv_l: number of large offshore construction vessels
    :param barge: number of barges
    :return: overall project duration, ocv_s time, ocv_l time, barge time
    """
    t_array = list()
    t_ocv_s = list()
    t_ocv_l = list()
    t_barge = list()

    for ip in range(len(ocv_s)):
        inf_loop_prevent = 0
        time_ocv_s = 0
        time_ocv_l = 0
        time_barge = 0
        anchor_counter = 0

        ds_ocv_s = ship_options['OCV small']['deck_space']
        ds_ocv_l = ship_options['OCV big']['deck_space']
        ds_barge = ship_options['Barge']['deck_space']

        if ocv_s[ip] + ocv_l[ip] + barge[ip] == 0.:
            time_ocv_s = 1e6
            time_ocv_l = 1e6
            time_barge = 1e6
        else:
            while n_anchors - anchor_counter > 0:
                if n_anchors - anchor_counter < ocv_s[ip] * ds_ocv_s + ocv_l[ip] * ds_ocv_l + barge[ip] * ds_barge:
                    n = ocv_s[ip] + ocv_l[ip] + barge[ip]
                    anchors_left_per_vessel = ceil((n_anchors - anchor_counter) / n)
                    diff_1 = 0
                    diff_2 = 0
                    ds_ocv_s = anchors_left_per_vessel
                    ds_ocv_l = anchors_left_per_vessel
                    ds_barge = anchors_left_per_vessel

                    if ds_ocv_s > ship_options['OCV small']['deck_space']:
                        diff_1 = ocv_s[ip] * (anchors_left_per_vessel - ship_options['OCV small']['deck_space'])
                        ds_ocv_s = ship_options['OCV small']['deck_space']

                        if ocv_l[ip] != 0:
                            if ds_ocv_l + diff_1 / ocv_l[ip] > ship_options['OCV big']['deck_space']:
                                diff_2 = ocv_l[ip] * (
                                        anchors_left_per_vessel + round(diff_1 / ocv_l[ip]) - ship_options['OCV big'][
                                    'deck_space'])
                                ds_ocv_l = ship_options['OCV big']['deck_space']
                                ds_barge += diff_2 / barge[ip]
                            else:
                                ds_ocv_l = anchors_left_per_vessel + ceil(diff_1 / (ocv_l[ip] + barge[ip]))
                                ds_barge = anchors_left_per_vessel + ceil(diff_1 / (ocv_l[ip] + barge[ip]))
                        else:
                            ds_barge = anchors_left_per_vessel + ceil(diff_1 / barge[ip])

                    try:
                        assert ocv_s[ip] * ds_ocv_s + ocv_l[ip] * ds_ocv_l + barge[ip] * ds_barge >= \
                               (n_anchors - anchor_counter)
                    except AssertionError as err:
                        print(ocv_s[ip], ocv_l[ip], barge[ip])
                        print(n_anchors - anchor_counter)
                        print(anchors_left_per_vessel)
                        print(diff_1)
                        print(diff_2)
                        print(ds_ocv_s)
                        print(ds_ocv_l)
                        print(ds_barge)
                        raise err

                time_ocv_s += ocv_s[ip] * ds_ocv_s * time_installation
                time_ocv_l += ocv_l[ip] * ds_ocv_l * time_installation
                time_barge += barge[ip] * ds_barge * time_installation

                anchor_counter += ocv_s[ip] * ship_options['OCV small']['deck_space'] + ocv_l[ip] * \
                                  ship_options['OCV big']['deck_space'] + barge[ip] * ship_options['Barge'][
                                      'deck_space']

                if n_anchors - anchor_counter <= 0:  # check if it is still the case after installation of last anchors
                    time_ocv_s += ocv_s[ip] * time_bunkering
                    time_ocv_l += ocv_l[ip] * time_bunkering
                    time_barge += barge[ip] * time_bunkering
                inf_loop_prevent += 1
                if inf_loop_prevent > 20:
                    break

        t_ocv_s.append(time_ocv_s)
        t_ocv_l.append(time_ocv_l)
        t_barge.append(time_barge)
        t_array.append(max(time_ocv_s, time_ocv_l, time_barge))

    return t_array, t_ocv_s, t_ocv_l, t_barge


def objective_costs(diameter, length, t_ocv_s, t_ocv_l, t_barge):
    """
    Function to calculate the costs of the project. This will depend on the procurement costs of the anchors and the
    time a vessel is needed.

    :param diameter: diameter of the anchors
    :param length: length of the anchors
    :param t_ocv_s: overall time small offshore construction vessels are needed
    :param t_ocv_l: overall time large offshore construction vessels are needed
    :param t_barge: overall time barges are needed
    :return: costs of the project
    """

    t = 0.02 * diameter
    mass_steel = (pi * length * diameter * t + pi / 4 * diameter ** 2 * t) * 7.85  # mT
    production_costs_anchor = (mass_steel * 815 + 40000) * n_anchors

    costs_ocv_s = np.array(t_ocv_s) * ship_options['OCV small']['day_rate']
    costs_ocv_l = np.array(t_ocv_l) * ship_options['OCV big']['day_rate']
    costs_barge = np.array(t_barge) * ship_options['Barge']['day_rate']
    return production_costs_anchor + costs_ocv_s + costs_ocv_l + costs_barge


def objective_fleet_utilization(ocv_s, ocv_l, barge):
    """
    Function to calculate the utilization of the fleet. In this exemplar this is simply the sum of the vessels.

    :param ocv_s: number of small offshore construction vessels
    :param ocv_l: number of large offshore construction vessels
    :param barge: number of barges
    :return: number of vessels used on the project
    """
    return ocv_s + ocv_l + barge


def objective_co2(ocv_s, ocv_l, barge, t_ocv_s, t_ocv_l, t_barge):
    """
    Function to calculate the overall tonnes of CO2 emitted on the project. This is calculated by the number of vessels,
     their time on the project, and the average emission rate per day.

    :param ocv_s: number of small offshore construction vessels
    :param ocv_l: number of large offshore construction vessels
    :param barge: number of barges
    :param t_ocv_s: overall time small offshore construction vessels are needed
    :param t_ocv_l: overall time large offshore construction vessels are needed
    :param t_barge: overall time barges are needed
    :return: tonnes of CO2 emitted on the project
    """
    co2_emission_ocv_s = np.array(t_ocv_s) * ship_options['OCV small']['CO2_emission'] * ocv_s
    co2_emission_ocv_l = np.array(t_ocv_l) * ship_options['OCV big']['CO2_emission'] * ocv_l
    co2_emission_barge = np.array(t_barge) * ship_options['Barge']['CO2_emission'] * barge
    return co2_emission_ocv_s + co2_emission_ocv_l + co2_emission_barge


def objective(variables):
    """
    Objective function that is fed to the GA. Calles the separate preference functions that are declared above.

    :param variables: array with design variable values per member of the population. Can be split by using array
    slicing
    :return: 1D-array with aggregated preference scores for the members of the population.
    """
    n_ocv_s = variables[:, 0]  # number of small offshore construction vessels
    n_ocv_l = variables[:, 1]  # number of large offshore construction vessels
    n_barge = variables[:, 2]  # number of barges
    length = variables[:, 3]  # length of anchors
    diameter = variables[:, 4]  # diameter of anchors

    # calculate objectives
    project_time, time_ocv_s, time_ocv_l, time_barge = objective_time(n_ocv_s, n_ocv_l, n_barge)
    costs = objective_costs(diameter, length, time_ocv_s, time_ocv_l, time_barge)
    fleet_util = objective_fleet_utilization(n_ocv_s, n_ocv_l, n_barge)
    co2_emission = objective_co2(n_ocv_s, n_ocv_l, n_barge, time_ocv_s, time_ocv_l, time_barge)

    # calculate the preference scores
    p_1 = pchip_interpolate(x_points_1, p_points_1, project_time)
    p_2 = pchip_interpolate(x_points_2, p_points_2, costs)
    p_3 = pchip_interpolate(x_points_3, p_points_3, fleet_util)
    p_4 = pchip_interpolate(x_points_4, p_points_4, co2_emission)

    # aggregate preference scores and return this to the GA
    return solver.request([w1, w2, w3, w4], [p_1, p_2, p_3, p_4])


def constraint_1(variables):
    """
    Inequality constraint that checks of the sum of vessels on the project > 0

    :param variables: array with design variable values per member of the population. Can be split by using array
    slicing
    :return: 1D-array with the results from the constraint
    """
    n_ocv_s = variables[:, 0]  # number of small offshore construction vessels
    n_ocv_l = variables[:, 1]  # number of large offshore construction vessels
    n_barge = variables[:, 2]  # number of barges

    return -1 * (n_ocv_s + n_ocv_l + n_barge) + 1  # < 0


def constraint_2(variables):  # V_ult * n_anchor > F_pull
    """
    Inequality constraint that checks if the pull force on the anchors is lower than the resistance of the anchors to
    this force. Assumed is an equal distribution of the force over the anchors and a pure horizontal translation of the
    anchor.

    :param variables: array with design variable values per member of the population. Can be split by using array
    slicing
    :return: 1D-array with the results from the constraint
    """
    length = variables[:, 3]  # length of anchors
    diameter = variables[:, 4]  # diameter of anchors

    t = 0.02 * diameter  # wall thickness
    d_e = diameter + t  # outer diameter

    fh_max = length * d_e * Np * Su  # resistance of the anchor under pure horizontal translation

    return max_t - fh_max  # < 0


"""
Below, the a priori optimization is performed. For this, we first need to define the bounds of the design variables, and
the list with constraints and their type.

The optimization can be ran multiple times, so you can check the consistency between the runs. The outcomes might differ
a bit, since the GA is stochastic from nature, but the differences should be limited.
"""

bounds = [
    [0, ship_options['OCV small']['max_available']],
    [0, ship_options['OCV big']['max_available']],
    [0, ship_options['Barge']['max_available']],
    [2, 8],
    [1.5, 4]
]
cons = [['ineq', constraint_1], ['ineq', constraint_2]]

n_runs = 2

# make dictionary with parameter settings for the GA run with the Tetra solver
options = {
    'n_bits': 16,
    'n_iter': 400,
    'n_pop': 250,
    'r_cross': 0.9,
    'max_stall': 10,
    'tetra': True,
    'var_type_mixed': ['int', 'int', 'int', 'real', 'real']
}

save_array = list()  # list to save the results from every run to
ga = GeneticAlgorithm(objective=objective, constraints=cons, cons_handler='CND', bounds=bounds, options=options)

# run the GA and print its result
for i in range(n_runs):
    print(f'Initialize run {i + 1}')
    score, design_variables, _ = ga.run()

    print(f'Optimal result for:\n'
          f'\t {design_variables[0]} small Offshore Construction Vessels\n'
          f'\t {design_variables[1]} large Offshore Construction Vessels\n'
          f'\t {design_variables[2]} Barges\n'
          f'\tAn anchor length of {round(design_variables[3], 2)}m\n'
          f'\tAn anchor diameter of {round(design_variables[4], 2)}m\n'
          )

    save_array.append(design_variables)
    print(f'Finished run {i + 1}')

"""
Now we have the results, we can plot the preference functions together with the results of the optimizations.
"""

# arrays for plotting continuous preference curves
c1 = np.linspace(44, 200)
c2 = np.linspace(9000000, 17000000)
c3 = np.linspace(1, 7)
c4 = np.linspace(1400, 55000)

# calculate the preference functions
p1 = pchip_interpolate(x_points_1, p_points_1, c1)
p2 = pchip_interpolate(x_points_2, p_points_2, c2)
p3 = pchip_interpolate(x_points_3, p_points_3, c3)
p4 = pchip_interpolate(x_points_4, p_points_4, c4)

# make numpy array of results, to allow for array splicing
variable = np.array(save_array)

# calculate individual preference scores for the results of the GA, to plot them on the preference curves
c1_res, t_res_1, t_res_2, t_res_3 = objective_time(variable[:, 0], variable[:, 1], variable[:, 2])
c2_res = objective_costs(variable[:, 4], variable[:, 3], t_res_1, t_res_2, t_res_3)
c3_res = objective_fleet_utilization(variable[:, 0], variable[:, 1], variable[:, 2])
c4_res = objective_co2(variable[:, 0], variable[:, 1], variable[:, 2], t_res_1, t_res_2, t_res_3)

p1_res = pchip_interpolate(x_points_1, p_points_1, c1_res)
p2_res = pchip_interpolate(x_points_2, p_points_2, c2_res)
p3_res = pchip_interpolate(x_points_3, p_points_3, c3_res)
p4_res = pchip_interpolate(x_points_4, p_points_4, c4_res)

# create figure that plots all preference curves and the preference scores of the returned results of the GA
fig = plt.figure()
ax1 = fig.add_subplot(1, 1, 1)
ax1.plot(c1, p1, label='Preference curve')
ax1.scatter(c1_res, p1_res, label='Optimal solution Tetra', color='tab:purple')
ax1.set_ylim((0, 100))
ax1.set_title('Project Duration')
ax1.set_xlabel('Duration [days]')
ax1.set_ylabel('Preference score')
ax1.grid()
ax1.legend()

fig = plt.figure()
ax2 = fig.add_subplot(1, 1, 1)
ax2.plot(c2, p2, label='Preference curve')
ax2.scatter(c2_res, p2_res, label='Optimal solution Tetra', color='tab:purple')
ax2.set_ylim((0, 100))
ax2.set_title('Costs')
ax2.set_xlabel('Costs [€]')
ax2.set_ylabel('Preference score')
ax2.grid()
ax2.legend()

fig = plt.figure()
ax3 = fig.add_subplot(1, 1, 1)
ax3.plot(c3, p3, label='Preference curve')
ax3.scatter(c3_res, p3_res, label='Optimal solution Tetra', color='tab:purple')
ax3.set_ylim((0, 100))
ax3.set_title('Fleet Utilization')
ax3.set_xlabel('Number of vessels [-]')
ax3.set_ylabel('Preference score')
ax3.grid()
ax3.legend()

fig = plt.figure()
ax4 = fig.add_subplot(1, 1, 1)
ax4.plot(c4, p4, label='Preference curve')
ax4.scatter(c4_res, p4_res, label='Optimal solution Tetra', color='tab:purple')
ax4.set_ylim((0, 100))
ax4.set_title(r'$CO_2$ emissions')
ax4.set_xlabel(r'$CO_2$ emission [tonnes]')
ax4.set_ylabel('Preference score')
ax4.grid()
ax4.legend()

plt.show()
