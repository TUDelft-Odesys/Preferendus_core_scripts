"""Copyright (c) 2022. Harold Van Heukelum"""

import pathlib
from math import ceil

import matplotlib.pyplot as plt
import numpy as np
from numpy import pi
from scipy.interpolate import pchip_interpolate

from genetic_algorithm_pfm import GeneticAlgorithm
from tetra_pfm import TetraSolver

HERE = pathlib.Path(__file__).parent

density_seawater = 1025
max_t = 3800
n_anchors = 108

# https://www.oilandgasiq.com/drilling-and-development/articles/offshore-support-vessels-leading-emissions-reducti

ship_options = {
    'OCV small': {
        'day_rate': 47000,
        'deck_space': 8,
        'max_available': 3,
        'CO2_emission': 30  # tonnes per day
    },
    'OCV big': {
        'day_rate': 55000,
        'deck_space': 12,
        'max_available': 2,
        'CO2_emission': 40  # tonnes per day
    },
    'Barge': {
        'day_rate': 35000,
        'deck_space': 16,
        'max_available': 2,
        'CO2_emission': 35  # tonnes per day
    }
}

w1 = 0.25
w2 = 0.35
w3 = 0.15
w4 = 0.25

Su = 70
Np = 10

solver = TetraSolver()

time_installation = 1
time_bunkering = 2


def objective_time(ocv_s, ocv_l, barge):
    """Function to calculate the installation time"""
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

            anchor_counter += ocv_s[ip] * ship_options['OCV small']['deck_space'] + ocv_l[ip] * ship_options['OCV big'][
                'deck_space'] + barge[ip] * ship_options['Barge']['deck_space']

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
    """Function to calculate the installation costs"""

    t = 0.02 * diameter
    mass_steel = (pi * length * diameter * t + pi / 4 * diameter ** 2 * t) * 7.85  # mT
    production_costs_anchor = (mass_steel * 815 + 40000) * n_anchors

    costs_ocv_s = np.array(t_ocv_s) * ship_options['OCV small']['day_rate']
    costs_ocv_l = np.array(t_ocv_l) * ship_options['OCV big']['day_rate']
    costs_barge = np.array(t_barge) * ship_options['Barge']['day_rate']
    return production_costs_anchor + costs_ocv_s + costs_ocv_l + costs_barge


def objective_fleet_utilization(ocv_s, ocv_l, barge):
    """Function to calculate the L/D ratio"""
    return ocv_s + ocv_l + barge


def objective_co2(ocv_s, ocv_l, barge, t_ocv_s, t_ocv_l, t_barge):
    """

    :return:
    """
    co2_emission_ocv_s = np.array(t_ocv_s) * ship_options['OCV small']['CO2_emission'] * ocv_s
    co2_emission_ocv_l = np.array(t_ocv_l) * ship_options['OCV big']['CO2_emission'] * ocv_l
    co2_emission_barge = np.array(t_barge) * ship_options['Barge']['CO2_emission'] * barge
    return co2_emission_ocv_s + co2_emission_ocv_l + co2_emission_barge


# arrays for plotting continuous preference curves
c1 = np.linspace(44, 200)
c2 = np.linspace(9000000, 17000000)
c3 = np.linspace(1, 7)
c4 = np.linspace(1400, 55000)

p1_min, p1_mid, p1_max = [44, 80, 200]
p2_min, p2_mid, p2_max = [9_000_000, 12_000_000, 17_000_000]
p3_min, p3_mid, p3_max = [1, 3, 7]
p4_min, p4_mid, p4_max = [1400, 20000, 55000]

p1 = pchip_interpolate([p1_min, p1_mid, p1_max], [100, 40, 0], c1)
p2 = pchip_interpolate([p2_min, p2_mid, p2_max], [100, 30, 0], c2)
p3 = pchip_interpolate([p3_min, p3_mid, p3_max], [0, 50, 100], c3)
p4 = pchip_interpolate([p4_min, p4_mid, p4_max], [100, 40, 0], c4)


def objective(variables):
    """Objective function for the GA. Calculates all sub-objectives and transforms it from their scale to the preference
    scale."""
    n_ocv_s = variables[:, 0]
    n_ocv_l = variables[:, 1]
    n_barge = variables[:, 2]
    length = variables[:, 3]
    diameter = variables[:, 4]

    project_time, time_ocv_s, time_ocv_l, time_barge = objective_time(n_ocv_s, n_ocv_l, n_barge)
    costs = objective_costs(diameter, length, time_ocv_s, time_ocv_l, time_barge)
    fleet_util = objective_fleet_utilization(n_ocv_s, n_ocv_l, n_barge)
    co2_emission = objective_co2(n_ocv_s, n_ocv_l, n_barge, time_ocv_s, time_ocv_l, time_barge)

    p_1 = pchip_interpolate([p1_min, p1_mid, p1_max], [100, 40, 0], project_time)
    p_2 = pchip_interpolate([p2_min, p2_mid, p2_max], [100, 30, 0], costs)
    p_3 = pchip_interpolate([p3_min, p3_mid, p3_max], [0, 50, 100], fleet_util)
    p_4 = pchip_interpolate([p4_min, p4_mid, p4_max], [100, 40, 0], co2_emission)

    return solver.request([w1, w2, w3, w4], [p_1, p_2, p_3, p_4])


def constraint_1(variables):
    """Constraint that checks if the pull force on the anchors is lower than the resistance of the anchors to this
    force. Assumed is an equal distribution of the force over the anchors."""
    n_ocv_s = variables[:, 0]
    n_ocv_l = variables[:, 1]
    n_barge = variables[:, 2]

    return -1 * (n_ocv_s + n_ocv_l + n_barge) + 1  # < 0


def constraint_2(variables):  # V_ult * n_anchor > F_pull
    """Constraint that checks if the pull force on the anchors is lower than the resistance of the anchors to this
    force. Assumed is an equal distribution of the force over the anchors."""
    length = variables[:, 3]
    diameter = variables[:, 4]

    t = 0.02 * diameter
    d_e = diameter + t

    fh_max = length * d_e * Np * Su

    return max_t - fh_max  # < 0


bounds = [
    [0, ship_options['OCV small']['max_available']],
    [0, ship_options['OCV big']['max_available']],
    [0, ship_options['Barge']['max_available']],
    [2, 8],
    [1.5, 4]
]
cons = [['ineq', constraint_1], ['ineq', constraint_2]]

if __name__ == '__main__':
    n_runs = 1

    # make dictionary with parameter settings for the GA, no changes have been made here.
    options = {
        'n_bits': 16,
        'n_iter': 400,
        'n_pop': 250,
        'r_cross': 0.9,
        'max_stall': 10,
        'tetra': True,
        'method_tetra': 1,
        'var_type_mixed': ['int', 'int', 'int', 'real', 'real']
    }

    save_array = list()
    ga = GeneticAlgorithm(objective=objective, constraints=cons, cons_handler='CND', bounds=bounds, options=options)
    for i in range(n_runs):
        print(f'Initialize run {i + 1}')
        score, decoded, _ = ga.run()
        print(f'Optimal result for:\n')
        print(f'\t {decoded[0]} small Offshore Construction Vessels\n')
        print(f'\t {decoded[1]} large Offshore Construction Vessels\n')
        print(f'\t {decoded[2]} Barges\n')
        print(f'\tAn anchor length of {round(decoded[3], 2)}m\n')
        print(f'\tAn anchor diameter of {round(decoded[4], 2)}m\n')
        save_array.append(decoded)
        print(f'Finished run {i + 1}')

    variable = np.array(save_array)

    c1_res, t_res_1, t_res_2, t_res_3 = objective_time(variable[:, 0], variable[:, 1], variable[:, 2])
    c2_res = objective_costs(variable[:, 4], variable[:, 3], t_res_1, t_res_2, t_res_3)
    c3_res = objective_fleet_utilization(variable[:, 0], variable[:, 1], variable[:, 2])
    c4_res = objective_co2(variable[:, 0], variable[:, 1], variable[:, 2], t_res_1, t_res_2, t_res_3)

    p1_res = pchip_interpolate([p1_min, p1_mid, p1_max], [100, 40, 0], c1_res)
    p2_res = pchip_interpolate([p2_min, p2_mid, p2_max], [100, 30, 0], c2_res)
    p3_res = pchip_interpolate([p3_min, p3_mid, p3_max], [0, 50, 100], c3_res)
    p4_res = pchip_interpolate([p4_min, p4_mid, p4_max], [100, 40, 0], c4_res)

    print(c1_res, p1_res)
    print(c2_res, p2_res)
    print(c3_res, p3_res)
    print(c4_res, p4_res)

    # create figure that plots all preference curves and the preference scores of the returned results of the GA
    fig2, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(15, 5))

    ax1.plot(c1, p1, label='Preference curve')
    ax1.scatter(c1_res, p1_res, label='Optimal solutions Tetra')
    ax1.set_ylim((0, 100))
    ax1.set_title('Project Duration')
    ax1.set_xlabel('Time [days]')
    ax1.set_ylabel('Preference score')
    ax1.grid()
    fig2.legend()

    ax2.plot(c2, p2)
    ax2.scatter(c2_res, p2_res)
    ax2.set_ylim((0, 100))
    ax2.set_title('Costs')
    ax2.set_xlabel('Costs [€]')
    ax2.set_ylabel('Preference score')
    ax2.grid()

    ax3.plot(c3, p3)
    ax3.scatter(c3_res, p3_res)
    ax3.set_ylim((0, 100))
    ax3.set_title('Fleet Utilization')
    ax3.set_xlabel('Number of vessels [-]')
    ax3.set_ylabel('Preference score')
    ax3.grid()

    ax4.plot(c4, p4)
    ax4.scatter(c4_res, p4_res)
    ax4.set_ylim((0, 100))
    ax4.set_title(r'$CO_2$ emissions')
    ax4.set_xlabel(r'$CO_2$ emission [tonnes]')
    ax4.set_ylabel('Preference score')
    ax4.grid()

    plt.show()
