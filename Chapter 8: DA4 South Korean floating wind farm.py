"""Copyright (c) 2022. Harold Van Heukelum"""

"""
Python code for the floating wind project development exemplar (Chapter 8.5)
"""

from math import ceil

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numpy import pi
from scipy.interpolate import pchip_interpolate
from scipy.optimize import fsolve

from genetic_algorithm_pfm import GeneticAlgorithm

"""
Before starting the SODO and MODO some general variables and characteristics are set.
Reference for the ship characteristics:
https://www.oilandgasiq.com/drilling-and-development/articles/offshore-support-vessels-leading-emissions-reducti
"""

# Set constants for objective functions
max_t = 3800
n_anchors = 108

constants = {
    'NC': 9,
    'NQ': 1,
    'W_steel': 78.5,  # kN/m3
    'W_water': 10.25,  # kN/m3
    'W_concrete': 25,  # kN/m3
}

# Set charachteristics of the different ship types
ship_options = {
    'OCV small': {
        'day_rate': 47000,  # €/day
        'deck_space': 8,  # max anchors on deck
        'max_available': 3, # how many of this type are available for the project
        'CO2_emission': 30,  # tonnes per day
        'chance': 0.7
    },
    'OCV big': {
        'day_rate': 55000,
        'deck_space': 12,
        'max_available': 2,
        'CO2_emission': 40,  # tonnes per day
        'chance': 0.8
    },
    'Barge': {
        'day_rate': 35000,
        'deck_space': 16,
        'max_available': 2,
        'CO2_emission': 35,  # tonnes per day
        'chance': 0.5
    }
}

soil_data = {
    'type': 'clay',
    'su': 60,  # kPa (Undrained shear strength)
    'a_i': 0.64, # - (Coefficient of shaft friction)
    'a_o': 0.64,
    'sat_weight': 9,  # kN/m3 (Submerged unit weight)
}

mooring_data = {
    'type': 'catenary',
    'line type': 'chain',
    'd': 0.24,  # m (Nominal chain diameter)
    'mu': 0.25,  # - (Coefficient of seabed friction)
    'AWB': 2.5  # - (Active bearing area coefficient)
}

# set variables for
time_installation = 1  # time it takes to install one anchor
time_bunkering = [1.5, 2, 2.5]  # time it takes to get a new set of anchors onboard

"""
Note the time objective function resulting in the overall project duration is implemented using a
Discrete Event Simulation (DES). You are not required to understand the code of the objective_time function.
It is annotated for further interest.
"""

def objective_time(ocv_s, ocv_l, barge):
    """
    Function to calculate the project duration

    :param ocv_s: number of small offshore construction vessels
    :param ocv_l: number of large offshore construction vessels
    :param barge: number of barges
    :return: overall project duration, ocv_s time, ocv_l time, barge time
    """
    # set empty list for respective vessel time
    t_array = list()
    t_ocv_s = list()
    t_ocv_l = list()
    t_barge = list()

    # loop through all diffrent combinations of ships
    for ip in range(len(ocv_s)):

        # initialize timers and counter
        inf_loop_prevent = 0
        time_ocv_s = 0
        time_ocv_l = 0
        time_barge = 0
        anchor_counter = 0

        # define ship deck space option
        ds_ocv_s = ship_options['OCV small']['deck_space']
        ds_ocv_l = ship_options['OCV big']['deck_space']
        ds_barge = ship_options['Barge']['deck_space']

        # iteration through installation process as long as number of anchor to install is smaller than
        # the number of anchors available
        while n_anchors - anchor_counter > 0:

            # check if ships are fully or only partially loaded
            if n_anchors - anchor_counter < ocv_s[ip] * ds_ocv_s + ocv_l[ip] * ds_ocv_l + barge[ip] * ds_barge:
                n = ocv_s[ip] + ocv_l[ip] + barge[ip]  # Number of vessels
                # number of anchors left is number of remaining anchors divided by number of ships (oversimplification)
                anchors_left_per_vessel = ceil((n_anchors - anchor_counter) / n)
                diff_1 = 0
                diff_2 = 0
                ds_ocv_s = anchors_left_per_vessel
                ds_ocv_l = anchors_left_per_vessel
                ds_barge = anchors_left_per_vessel

                # distribute remaining number of anchors on ship starting with the OCV small
                # if number of remaining anchors to store is smaller then the deck space of the OCV small
                # are distributed on to the OCV big
                if ds_ocv_s > ship_options['OCV small']['deck_space']:
                    diff_1 = ocv_s[ip] * (anchors_left_per_vessel - ship_options['OCV small']['deck_space'])
                    ds_ocv_s = ship_options['OCV small']['deck_space']

                    # if the deck space of the OCV big is exceede the remaining anchors are stored on the barge
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

                # check that the number of anchors to be installed is equal to or greater than the number of anchors
                # remaining to be installed
                try:
                    assert ocv_s[ip] * ds_ocv_s + ocv_l[ip] * ds_ocv_l + barge[ip] * ds_barge >= \
                           (n_anchors - anchor_counter)
                # if not the code will be stopped
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

            #  increase respective ship time
            time_ocv_s += ocv_s[ip] * ds_ocv_s * time_installation
            time_ocv_l += ocv_l[ip] * ds_ocv_l * time_installation
            time_barge += barge[ip] * ds_barge * time_installation

            # increase anchor counter to number of installed anchors
            anchor_counter += ocv_s[ip] * ship_options['OCV small']['deck_space'] + ocv_l[ip] * ship_options['OCV big'][
                'deck_space'] + barge[ip] * ship_options['Barge']['deck_space']

            if n_anchors - anchor_counter <= 0:  # check if it is still the case after installation of last anchors
                time_ocv_s += ocv_s[ip] * time_bunkering[0]  # add time to load new anchor
                time_ocv_l += ocv_l[ip] * time_bunkering[1]
                time_barge += barge[ip] * time_bunkering[2]

            inf_loop_prevent += 1  # preventing an infinite loop (if sum of ships is zero)

            # if no anchors are installed, the while loop returns a high value for the timers and breaks the while loop
            if inf_loop_prevent > 20:
                time_ocv_s += 1e4
                time_ocv_l += 1e4
                time_barge += 1e4
                break

        # time is added to overall list of alternatives
        t_ocv_s.append(time_ocv_s)
        t_ocv_l.append(time_ocv_l)
        t_barge.append(time_barge)
        t_array.append(max(time_ocv_s, time_ocv_l, time_barge))

    return t_array, t_ocv_s, t_ocv_l, t_barge


def objective_costs(diameter, length, t_ocv_s, t_ocv_l, t_barge):
    """Function to calculate the installation costs"""

    t = 0.02 * diameter
    mass_steel = (pi * length * diameter * t + pi / 4 * diameter ** 2 * t) * 7.85  # mT
    production_costs_anchor = (mass_steel * 815 + 40000) * n_anchors  # Calculate material cost

    costs_ocv_s = np.array(t_ocv_s) * ship_options['OCV small']['day_rate']
    costs_ocv_l = np.array(t_ocv_l) * ship_options['OCV big']['day_rate']
    costs_barge = np.array(t_barge) * ship_options['Barge']['day_rate']
    return production_costs_anchor + costs_ocv_s + costs_ocv_l + costs_barge


def objective_fleet_utilization(ocv_s, ocv_l, barge):
    """Function to calculate the fleet utilization"""
    chance_ocv_s = ship_options['OCV small']['chance'] ** ocv_s
    chance_ocv_l = ship_options['OCV big']['chance'] ** ocv_l
    chance_barge = ship_options['Barge']['chance'] ** barge
    return np.prod([np.power(chance_ocv_s, ocv_s), np.power(chance_ocv_l, ocv_l), np.power(chance_barge, barge)],
                   axis=0)


def objective_co2(ocv_s, ocv_l, barge, t_ocv_s, t_ocv_l, t_barge):
    """Function to calculate the CO2 emissions"""
    co2_emission_ocv_s = np.array(t_ocv_s) * ship_options['OCV small']['CO2_emission'] * ocv_s
    co2_emission_ocv_l = np.array(t_ocv_l) * ship_options['OCV big']['CO2_emission'] * ocv_l
    co2_emission_barge = np.array(t_barge) * ship_options['Barge']['CO2_emission'] * barge
    return co2_emission_ocv_s + co2_emission_ocv_l + co2_emission_barge


def single_objective_time(variables):
    """Function for single objective optimization of the project duration"""
    n_ocv_s = variables[:, 0]
    n_ocv_l = variables[:, 1]
    n_barge = variables[:, 2]
    t_array, t_ocv_s, t_ocv_l, t_barge = objective_time(n_ocv_s, n_ocv_l, n_barge)
    return t_array


def single_objective_costs(variables):
    """Function for single objective optimization of the installation costs"""
    n_ocv_s = variables[:, 0]
    n_ocv_l = variables[:, 1]
    n_barge = variables[:, 2]
    diameter = variables[:, 3]
    length = variables[:, 4]
    _, t_ocv_s, t_ocv_l, t_barge = objective_time(n_ocv_s, n_ocv_l, n_barge)
    return objective_costs(diameter, length, t_ocv_s, t_ocv_l, t_barge)


def single_objective_fleet(variables):
    """Function for single objective optimization of the fleet utilization"""
    n_ocv_s = variables[:, 0]
    n_ocv_l = variables[:, 1]
    n_barge = variables[:, 2]
    return objective_fleet_utilization(n_ocv_s, n_ocv_l, n_barge)


def single_objective_co2(variables):
    """Function for single objective optimization of the CO2 emissions"""
    n_ocv_s = variables[:, 0]
    n_ocv_l = variables[:, 1]
    n_barge = variables[:, 2]
    _, t_ocv_s, t_ocv_l, t_barge = objective_time(n_ocv_s, n_ocv_l, n_barge)
    return objective_co2(n_ocv_s, n_ocv_l, n_barge, t_ocv_s, t_ocv_l, t_barge)


# arrays for plotting continuous preference curves
c1 = np.linspace(45, 113)
c2 = np.linspace(9_500_000, 17_000_000)
c3 = np.linspace(0, 1)
c4 = np.linspace(3_200, 10_200)

# x_points: the outcomes of the objective for which a preference score is defined by the stakeholders
# p_points: the corresponding preference scores
x_points_1, p_points_1 = [[45, 80, 113], [100, 60, 0]]
x_points_2, p_points_2 = [[9_500_000, 11_000_000, 17_000_000], [100, 20, 0]]
x_points_3, p_points_3 = [[0, 0.6, 1], [100, 50, 0]]
x_points_4, p_points_4 = [[3_200, 5_000, 10_200], [100, 40, 0]]

# calculate the preference scores
p1 = pchip_interpolate(x_points_1, p_points_1, c1)
p2 = pchip_interpolate(x_points_2, p_points_2, c2)
p3 = pchip_interpolate(x_points_3, p_points_3, c3)
p4 = pchip_interpolate(x_points_4, p_points_4, c4)

# set weights for the different objectives
w1 = 0.30  # project duration
w2 = 0.35  # cost
w3 = 0.15  # fleet utilization
w4 = 0.20  # sustainability


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
    n_ocv_s = variables[:, 0]  # number of small offshore construction vessels
    n_ocv_l = variables[:, 1]  # number of large offshore construction vessels
    n_barge = variables[:, 2]  # number of barges
    diameter = variables[:, 3]  # anchor diameter
    length = variables[:, 4]  # anchor length

    project_time, time_ocv_s, time_ocv_l, time_barge = objective_time(n_ocv_s, n_ocv_l, n_barge)
    costs = objective_costs(diameter, length, time_ocv_s, time_ocv_l, time_barge)
    fleet_util = objective_fleet_utilization(n_ocv_s, n_ocv_l, n_barge)
    co2_emission = objective_co2(n_ocv_s, n_ocv_l, n_barge, time_ocv_s, time_ocv_l, time_barge)

    # calculate the preference scores including the check of the preference score
    p_1 = check_p_score(pchip_interpolate(x_points_1, p_points_1, project_time))
    p_2 = check_p_score(pchip_interpolate(x_points_2, p_points_2, costs))
    p_3 = check_p_score(pchip_interpolate(x_points_3, p_points_3, fleet_util))
    p_4 = check_p_score(pchip_interpolate(x_points_4, p_points_4, co2_emission))

    # aggregate preference scores and return this to the GA
    return [w1, w2, w3, w4], [p_1, p_2, p_3, p_4]

"""
Before we can run the optimization, we finally need to define the constraints and bounds.
You are not required to understand the code, so it is not annotated further.
"""

def constraint_1(variables):
    """Constraint that ensures there is at least one vessel on the project"""
    n_ocv_s = variables[:, 0]
    n_ocv_l = variables[:, 1]
    n_barge = variables[:, 2]

    return -1 * (n_ocv_s + n_ocv_l + n_barge) + 1  # < 0


def _solve_ta_ta(p, tension_mudline, theta_m, za, d, mu, su, nc=7.6):
    """Solve the force o the anchor and its angle, based on the tension and angle of the mooring line at the seabed"""
    tension_a, theta = p
    za_q = mooring_data['AWB'] * d * nc * su * za

    return (2 * za_q / tension_a) - (theta ** 2 - theta_m ** 2), np.exp(
        mu * (theta - theta_m)) - tension_mudline / tension_a


def constraint_2(variables):
    """Constraint that checks if the pull force on the anchors is lower than the resistance of the anchors to this
    force.

    The calculations are based on:
        - Houlsby, G. T. and Byrne, B. W. (2005). “Design procedures for installation of suction caissons in clay and
        other materials.” Proceedings of the Institution of Civil Engineers-Geotechnical Engineering, 158(2), 75–82.
        - Randolph, M. and Gourvenec, S. (2017). Offshore geotechnical engineering. CRC press.
        - Arany, L. and Bhattacharya, S. (2018). “Simplified load estimation and sizing of suction anchors for spar buoy
         type floating offshore wind turbines.” Ocean Engineering, 159, 348–357.
    """
    diameter = variables[:, 3]
    length = variables[:, 4]

    t = 0.02 * diameter
    d_i = diameter - t
    d_e = diameter + t
    mean_diameter = diameter
    weight_anchor = (np.pi * length * mean_diameter * t + np.pi * mean_diameter ** 2 * t / 4) * (
            constants['W_steel'] - constants['W_water'])

    weight_plug = np.pi / 4 * d_i ** 2 * length * soil_data['sat_weight']

    external_shaft_fric = np.pi * d_e * length * soil_data['a_o'] * soil_data['su']
    internal_shaft_fric = np.pi * d_i * length * soil_data['a_i'] * soil_data['su']
    reverse_end_bearing = 6.7 * soil_data['su'] * d_e ** 2 * np.pi / 4

    v_mode_1 = weight_anchor + external_shaft_fric + reverse_end_bearing
    v_mode_2 = weight_anchor + external_shaft_fric + internal_shaft_fric
    v_mode_3 = weight_anchor + external_shaft_fric + weight_plug

    v_max = np.amin([v_mode_1, v_mode_2, v_mode_3], axis=0)
    h_max = length * d_e * 10 * soil_data['su']

    rel_pos_pad_eye = 0.5

    tension_pad_eye = np.zeros(len(length))
    angle_pad_eye = np.zeros(len(length))
    for lng in np.unique(length):
        x = fsolve(_solve_ta_ta, np.array([10000, 1]),
                   (max_t, 0, rel_pos_pad_eye * lng, mooring_data['d'], mooring_data['mu'], soil_data['su'], 12))
        mask = length == lng
        tension_pad_eye[mask] = x[0]
        angle_pad_eye[mask] = x[1]

    h = np.cos(angle_pad_eye) * tension_pad_eye
    v = np.sin(angle_pad_eye) * tension_pad_eye

    a = length / mean_diameter + 0.5
    b = length / (3 * mean_diameter) + 4.5

    hor_util = h / h_max
    ver_util = v / v_max

    return (hor_util ** a + ver_util ** b) - 1


# define bounds and set constraints list
bounds = [
    [0, ship_options['OCV small']['max_available']],
    [0, ship_options['OCV big']['max_available']],
    [0, ship_options['Barge']['max_available']],
    [1.5, 4],
    [2, 8]
]
cons = [['ineq', constraint_1], ['ineq', constraint_2]]

"""
Below, the a priori optimization is performed. The optimization can be ran multiple times, so you can check the
consistency between the runs. The outcomes might differ a bit, since the GA is stochastic from nature, but the
differences should be limited. Besides the tetra solver both the SODO of the objective installation costs and the
MODO min-max optimization are performed.

Note that the other SODO on project time, fleet utilisation and co2 emissions cannot be included because in this
example, these do not depend on the variables anchor length and anchor diameter.
"""

def print_results(res):
    """Function that prints the results of the optimizations"""
    print(f'Optimal result for:\n')
    print(f'\t {res[0]} small Offshore Construction Vessels\n')
    print(f'\t {res[1]} large Offshore Construction Vessels\n')
    print(f'\t {res[2]} Barges\n')
    print(f'\tAn anchor diameter of {round(res[3], 2)}m\n')
    print(f'\tAn anchor length of {round(res[4], 2)}m\n')


if __name__ == '__main__':
    ####################################################################################
    # run single objectives and save to save_array
    save_array = list()
    methods = list()

    # make dictionary with parameter settings for the GA
    options = {
        'n_bits': 16,
        'n_iter': 400,
        'n_pop': 500,
        'r_cross': 0.8,
        'max_stall': 10,
        'var_type_mixed': ['int', 'int', 'int', 'real', 'real'],
    }

    # time
    ga = GeneticAlgorithm(objective=single_objective_time, constraints=cons, cons_handler='CND', bounds=bounds,
                          options=options)
    res_time, design_variables_SO_time, _ = ga.run()
    print_results(design_variables_SO_time)
    print(f'SODO project duration: {round(res_time, 2)} days')

    # fleet utilization
    ga = GeneticAlgorithm(objective=single_objective_fleet, constraints=cons, cons_handler='CND', bounds=bounds,
                          options=options)
    res_fleet, design_variables_SO_fleet, _ = ga.run()
    print_results(design_variables_SO_fleet)
    print(f'SODO fleet utilization: {round(res_fleet, 2)}')

    # CO2
    ga = GeneticAlgorithm(objective=single_objective_co2, constraints=cons, cons_handler='CND', bounds=bounds,
                          options=options)
    res_co2, design_variables_SO_co2, _ = ga.run()
    print_results(design_variables_SO_co2)
    print(f'SODO CO2 emissions: {round(res_co2, 2)} [mT]')

    # costs
    options['n_bits'] = 20
    options['n_pop'] = 1500
    options['r_cross'] = 0.85
    options['mutation_rate_order'] = 4
    options['elitism percentage'] = 10

    ga = GeneticAlgorithm(objective=single_objective_costs, constraints=cons, bounds=bounds,
                          options=options)
    res_costs, design_variables_SO_costs, _ = ga.run()
    print_results(design_variables_SO_costs)
    print(f'SODO installation costs: €{round(res_costs, 2)}')
    save_array.append(design_variables_SO_costs)
    methods.append('SODO Costs')

    ####################################################################################
    # run multi-objective with minmax solver

    # change some entries in the options dictionary
    options['n_bits'] = 24
    options['r_cross'] = 0.85
    options['aggregation'] = 'minmax'

    ga = GeneticAlgorithm(objective=objective, constraints=cons, cons_handler='CND', bounds=bounds, options=options)
    _, design_variables_minmax, best_mm = ga.run()
    print_results(design_variables_minmax)
    save_array.append(design_variables_minmax)
    methods.append('Min-max')

    ####################################################################################
    # run multi-objective with tetra solver

    # change some entries in the options dictionary
    options['n_bits'] = 20
    options['n_pop'] = 500
    options['r_cross'] = 0.85
    options['tetra'] = True
    options['aggregation'] = 'tetra'
    options['mutation_rate_order'] = 2

    ga = GeneticAlgorithm(objective=objective, constraints=cons, cons_handler='CND', bounds=bounds, options=options,
                          start_points_population=[design_variables_minmax])
    _, design_variables_tetra, best_t = ga.run()
    print_results(design_variables_tetra)
    save_array.append(design_variables_tetra)
    methods.append('IMAP')

    ###################################################################################
    # evaluate all runs

    variable = np.array(save_array)  # make ndarray

    w, p = objective(variable)  # evaluate objective
    r = ga.solver.request(w, p)  # get aggregated scores to rank them

    # create pandas DataFrame and print it to console
    d = {'Method': methods,
         'Results': r,
         'Variable 1': np.round_(variable[:, 0]),
         'Variable 2': np.round_(variable[:, 1]),
         'Variable 3': np.round_(variable[:, 2]),
         'Variable 4': np.round_(variable[:, 3], 2),
         'Variable 5': np.round_(variable[:, 4], 2),
         }
    print()
    print(pd.DataFrame(data=d).to_string())
    print()

    c1_res, t_res_1, t_res_2, t_res_3 = objective_time(variable[:, 0], variable[:, 1], variable[:, 2])
    c2_res = objective_costs(variable[:, 3], variable[:, 4], t_res_1, t_res_2, t_res_3)
    c3_res = objective_fleet_utilization(variable[:, 0], variable[:, 1], variable[:, 2])
    c4_res = objective_co2(variable[:, 0], variable[:, 1], variable[:, 2], t_res_1, t_res_2, t_res_3)

    p1_res = pchip_interpolate(x_points_1, p_points_1, c1_res)
    p2_res = pchip_interpolate(x_points_2, p_points_2, c2_res)
    p3_res = pchip_interpolate(x_points_3, p_points_3, c3_res)
    p4_res = pchip_interpolate(x_points_4, p_points_4, c4_res)

    d = {'Method': methods,
         'Project duration': np.round_(c1_res, 2),
         'Costs [1e6]': np.round_(c2_res * 1e-6, 2),
         'Fleet util': np.round_(c3_res, 2),
         'Emissions': np.round_(c4_res),
         }
    print()
    print(pd.DataFrame(data=d).to_string())
    print()

    d = {'Method': methods,
         'Project duration': np.round_(p1_res),
         'Costs': np.round_(p2_res),
         'Fleet util': np.round_(p3_res),
         'Emissions': np.round_(p4_res),
         }
    print()
    print(pd.DataFrame(data=d).to_string())
    print()

    # create figure that plots all preference curves and the preference scores of the returned results of the GA
    markers = ['o', 's', '+']
    fig2, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(15, 5))

    ax1.plot(c1, p1, label='Preference Function')
    for i in range(len(c1_res)):
        ax1.scatter(c1_res[i], p1_res[i], label=methods[i], marker=markers[i])
    ax1.set_ylim((0, 100))
    ax1.set_title('Project Duration')
    ax1.set_xlabel('Time [days]')
    ax1.set_ylabel('Preference function outcome')
    ax1.grid()
    fig2.legend()

    ax2.plot(c2, p2)
    for i in range(len(c2_res)):
        ax2.scatter(c2_res[i], p2_res[i], marker=markers[i])
    ax2.set_ylim((0, 100))
    ax2.set_title('Installation Costs')
    ax2.set_xlabel('Costs [€]')
    ax2.set_ylabel('Preference function outcome')
    ax2.grid()

    ax3.plot(c3, p3)
    for i in range(len(c3_res)):
        ax3.scatter(c3_res[i], p3_res[i], marker=markers[i])
    ax3.set_ylim((0, 100))
    ax3.set_title('Fleet Utilization')
    ax3.set_xlabel('Number of vessels [-]')
    ax3.set_ylabel('Preference function outcome')
    ax3.grid()

    ax4.plot(c4 * 1e-3, p4)
    for i in range(len(c4_res)):
        ax4.scatter(c4_res[i] * 1e-3, p4_res[i], marker=markers[i])
    ax4.set_ylim((0, 100))
    ax4.set_title(r'$CO_2$ emissions')
    ax4.set_xlabel(r'$CO_2$ emission [$10^3$ tonnes]')
    ax4.set_ylabel('Preference function outcome')
    ax4.grid()

    plt.show()
