"""
Copyright (c) 2022. Harold Van Heukelum
"""

import matplotlib.pyplot as plt
import numpy as np
from numpy import log10, zeros, invert, array, linspace, count_nonzero, sqrt
from scipy.interpolate import pchip_interpolate
from scipy.optimize import fsolve

from genetic_algorithm_pfm import GeneticAlgorithm
from tetra_pfm import TetraSolver
from weighted_minmax import aggregate_max

# arrays for plotting continuous preference curves
c1 = linspace(500, 800)  # costs
c2 = linspace(20, 200)  # area
c3 = linspace(1300, 1700)  # time

p1_min, p1_mid, p1_max = [500, 600, 800]
p2_min, p2_mid, p2_max = [20, 130, 200]
p3_min, p3_mid, p3_max = [1300, 1450, 1700]

p1 = pchip_interpolate([p1_min, p1_mid, p1_max], [100, 50, 0], c1)
# the preference function for costs

p2 = pchip_interpolate([p2_min, p2_mid, p2_max], [100, 50, 0], c2)
# the preference function for area

p3 = pchip_interpolate([p3_min, p3_mid, p3_max], [100, 40, 0], c3)
# the preference function for the building time


solver = TetraSolver()

LOA = 700


def objective_noise(r, dc):
    """
    Based on "FORMULAS FOR PREDICTING AUDIBLE NOISE FROM OVERHEAD HIGH VOLTAGE AC AND DC LINES" by Fellow & Member, 1981
    :return:
    """
    height = 12
    distance_per_line = 20
    db = 45
    if dc:
        e = 22  # kV/cm
        deq = 109.71  # mm
        ret = -133.4 + 86 * log10(e) + 40 * log10(deq) - 11.4 * log10(sqrt(r ** 2 + height ** 2))
    else:
        e_inner_phase = 14.58  # kV/cm
        e_outer_phase = 13.66  # kV/cm
        deq = 69.4  # mm
        pwl_inner = -164.6 + 120 * log10(e_inner_phase) + 55 * log10(deq)
        pwl_outer = -164.6 + 120 * log10(e_outer_phase) + 55 * log10(deq)

        temp_outer_close = (pwl_outer - 11.4 * log10(sqrt(r ** 2 + height ** 2)) - 5.8) / 10
        temp_inner = (pwl_inner - 11.4 * log10(sqrt((r + distance_per_line) ** 2 + height ** 2)) - 5.8) / 10
        temp_outer_far = (pwl_outer - 11.4 * log10(sqrt((r + 2 * distance_per_line) ** 2 + height ** 2)) - 5.8) / 10

        ret = 10 * log10(10 ** temp_inner + 10 ** temp_outer_close + 10 ** temp_outer_far)
    return ret - db


r_ac = fsolve(objective_noise, np.array([10]), (False,))
r_dc = fsolve(objective_noise, np.array([10]), (True,))


def objective(variabels, length_overall=LOA, method='tetra'):
    """

    :param variabels:
    :param length_overall
    :param method
    :return:
    """
    ac_dc = variabels[:, 0]  # bool
    length_underground = variabels[:, 1]  # real, km

    length_overground = length_overall - length_underground

    costs = zeros(len(length_underground))
    area = zeros(len(length_underground))
    time = zeros(len(length_underground))

    mask = ac_dc == 1
    costs[mask] = 0.475 * length_overground[mask] + 0.580 * length_underground[mask] + 375
    area[mask] = (0.170 + r_ac[0] / 1000) * length_overground[mask] + 0.018 * length_underground[mask]
    time[mask] = 2.5 * length_overground[mask] + 2.6 * length_underground[mask]

    mask = invert(mask)
    costs[mask] = 0.120 * length_overground[mask] + 0.190 * length_underground[mask] + 430
    area[mask] = (0.120 + r_dc[0] / 1000) * length_overground[mask] + 0.015 * length_underground[mask]
    time[mask] = 1.8 * length_overground[mask] + 2.3 * length_underground[mask]

    assert count_nonzero(costs) == len(costs), 'Problem with masks in objective function'

    p_costs = pchip_interpolate([p1_min, p1_mid, p1_max], [100, 50, 0], costs)
    p_area = pchip_interpolate([p2_min, p2_mid, p2_max], [100, 50, 0], area)
    p_time = pchip_interpolate([p3_min, p3_mid, p3_max], [100, 40, 0], time)

    w = [0.40, 0.20, .40]
    if method == 'minmax':
        ret = aggregate_max(w, [p_costs, p_area, p_time], 100)
    else:
        ret = solver.request(w, [p_costs, p_area, p_time])

    return ret


bounds = [[0, 1], [300, 600]]

# ar = np.array([[1, 300], [1, 600], [0, 300], [0, 600]])
# print(objective(ar))

if __name__ == '__main__':
    n_runs = 1
    # run Tetra version
    print('Run Tetra')
    options = {
        'n_bits': 20,
        'n_iter': 400,
        'n_pop': 150,
        'r_cross': 0.85,
        'max_stall': 15,
        'tetra': True,
        'method_tetra': 1,
        'var_type_mixed': ['bool', 'real']
    }

    save_array = list()
    ga = GeneticAlgorithm(objective=objective, constraints=[], bounds=bounds, options=options, args=(LOA, 'tetra'))
    for i in range(n_runs):
        print(f'Initialize run {i + 1}')
        score, decoded, _ = ga.run(verbose=True)
        if decoded[0]:
            type_cable = 'AC'
        else:
            type_cable = 'DC'
        print(
            f"Optimal result for type = {'AC' if decoded[0] else 'DC'}, length underground = {round(decoded[1], 2)}km"
            f", and length overground = {LOA - round(decoded[1], 2)}km")

        print()

        LOG = LOA - decoded[1]

        if decoded[0]:  # AC
            c = 0.476 * LOG + 0.578 * decoded[1] + 376.9
            a = (0.170 + r_ac[0] / 1000) * LOG + 0.018 * decoded[1]
            t = 2.22 * LOG + 2.35 * decoded[1]
        else:
            c = 0.120 * LOG + 0.189 * decoded[1] + 427.6
            a = (0.120 + r_dc[0] / 1000) * LOG + 0.015 * decoded[1]
            t = 1.75 * LOG + 2.33 * decoded[1]

        p_c = pchip_interpolate([p1_min, p1_mid, p1_max], [100, 50, 0], c)
        p_a = pchip_interpolate([p2_min, p2_mid, p2_max], [100, 50, 0], a)
        p_t = pchip_interpolate([p3_min, p3_mid, p3_max], [100, 40, 0], t)

        print(f'Costs = €{round(c, 4)}E6')
        print(f'Area = {round(a, 2)} km^2')
        print(f'Time = {round(t, 2)} days')

        print()
        print(f'Preference costs = {p_c.round(2)}')
        print(f'Preference area = {p_a.round(2)}')
        print(f'Preference time = {p_t.round(2)}')

        save_array.append([decoded[0], decoded[1], LOA - decoded[1], c, a, t, p_c, p_a, p_t])

        print(f'Finished run {i + 1}')

    # run minmax version
    print('Run MinMax')
    options = {
        'n_bits': 24,
        'n_iter': 400,
        'n_pop': 250,
        'r_cross': 0.9,
        'max_stall': 15,
        'tetra': False,
        'var_type_mixed': ['bool', 'real']
    }

    save_array_minmax = list()
    ga = GeneticAlgorithm(objective=objective, constraints=[], bounds=bounds, options=options, args=(LOA, 'minmax'))
    for i in range(n_runs):
        print(f'Initialize run {i + 1}')
        score, decoded, _ = ga.run(verbose=True)

        if decoded[0]:
            type_cable = 'AC'
        else:
            type_cable = 'DC'
        print(
            f"Optimal result for type = {'AC' if decoded[0] else 'DC'}, length underground = {round(decoded[1], 2)}km"
            f", and length overground = {LOA - round(decoded[1], 2)}km")

        print()

        LOG = LOA - decoded[1]

        if decoded[0]:  # AC
            c = 0.476 * LOG + 0.578 * decoded[1] + 376.9
            a = (0.170 + r_ac[0] / 1000) * LOG + 0.018 * decoded[1]
            t = 2.22 * LOG + 2.35 * decoded[1]
        else:
            c = 0.120 * LOG + 0.189 * decoded[1] + 427.6
            a = (0.120 + r_dc[0] / 1000) * LOG + 0.015 * decoded[1]
            t = 1.75 * LOG + 2.33 * decoded[1]

        p_c = pchip_interpolate([p1_min, p1_mid, p1_max], [100, 50, 0], c)
        p_a = pchip_interpolate([p2_min, p2_mid, p2_max], [100, 50, 0], a)
        p_t = pchip_interpolate([p3_min, p3_mid, p3_max], [100, 40, 0], t)

        print(f'Costs = €{round(c, 4)}E6')
        print(f'Area = {round(a, 2)} km^2')
        print(f'Time = {round(t, 2)} days')

        print()
        print(f'Preference costs = {p_c.round(2)}')
        print(f'Preference area = {p_a.round(2)}')
        print(f'Preference time = {p_t.round(2)}')

        save_array_minmax.append([decoded[0], decoded[1], LOA - decoded[1], c, a, t, p_c, p_a, p_t])

        print(f'Finished run {i + 1}')

    # Create figure that shows the results in the solution space, the solution space is also
    # shown in figure 3 in the report. The optimal results are determined in this model.
    x_fill = [300, 600, 600, 300]
    y_fill = [400, 400, 100, 100]

    variable = array(save_array)
    variable_mm = array(save_array_minmax)
    c1_res = variable[:, 3]
    c2_res = variable[:, 4]
    c3_res = variable[:, 5]
    c1_res_mm = variable_mm[:, 3]
    c2_res_mm = variable_mm[:, 4]
    c3_res_mm = variable_mm[:, 5]

    p1_res = variable[:, 6]
    p2_res = variable[:, 7]
    p3_res = variable[:, 8]
    p1_res_mm = variable_mm[:, 6]
    p2_res_mm = variable_mm[:, 7]
    p3_res_mm = variable_mm[:, 8]

    fig, ax = plt.subplots()
    ax.set_xlim((0, 800))
    ax.set_ylim((0, 800))
    ax.set_ylabel('Length overground cable')
    ax.set_xlabel('Length underground cable')
    ax.set_title('Solution space')
    ax.fill_between(x_fill, y_fill, color='#539ecd', label='Solution space')
    ax.scatter(variable[:, 1], variable[:, 2], label='Optimal solutions Tetra')
    ax.scatter(variable_mm[:, 1], variable_mm[:, 2], label='Optimal solutions MinMax', marker='^')
    fig.legend()
    ax.grid()

    # create figure that plots all preference curves and the preference scores of the returned results of the GA
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))

    ax1.plot(c1, p1, label='Optimal solutions Tetra')
    ax1.scatter(c1_res, p1_res, label='Optimal solutions Tetra')
    ax1.scatter(c1_res_mm, p1_res_mm, label='Optimal solutions MinMax', marker='^')
    ax1.set_ylim((0, 100))
    ax1.set_title('Costs')
    ax1.set_xlabel('Costs [€*1e6]')
    ax1.set_ylabel('Preference score')
    ax1.grid()
    ax1.legend()

    ax2.plot(c2, p2)
    ax2.scatter(c2_res, p2_res)
    ax2.scatter(c2_res_mm, p2_res_mm, marker='^')
    ax2.set_ylim((0, 100))
    ax2.set_title('Area')
    ax2.set_xlabel(r'Area [km$^2$]')
    ax2.set_ylabel('Preference score')
    ax2.grid()

    ax3.plot(c3, p3)
    ax3.scatter(c3_res, p3_res)
    ax3.scatter(c3_res_mm, p3_res_mm, marker='^')
    ax3.set_ylim((0, 100))
    ax3.set_title('Time')
    ax3.set_xlabel('Duration [days]')
    ax3.set_ylabel('Preference score')
    ax3.grid()

    plt.show()
