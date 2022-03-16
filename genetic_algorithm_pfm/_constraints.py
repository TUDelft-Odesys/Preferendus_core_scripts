"""Scripts for constraints handling. See functions for reference to papers that describe the different methods.

Copyright (c) Harold van Heukelum, 2021
"""

from numpy import array, zeros, multiply

k = 1e9


def _static_penalty(scores, result_array):
    """
    Source:
    Morales, A. K., & Quezada, C. V. (1998, September). A universal eclectic genetic algorithm for constrained
    optimization. In Proceedings of the 6th European congress on intelligent techniques and soft computing
    (Vol. 1, pp. 518-522).

    :param scores: List with the scores from the objective function
    :param result_array: List with the results from the constraints calculations in function _cons_handler
    :return: penalized scores, number of non-feasible results
     """

    scores = array(scores)
    mask_array = list()
    mask = [False, ] * len(scores)

    for eq, result in result_array:
        if eq == 'ineq':
            mask_array.append(array(result) > 0)
        elif eq == 'eq':
            mask_array.append(abs(array(result)) - 1e-4 > 0)

    mask_array = array(mask_array)

    for item in mask_array:
        mask = mask | item

    for i in range(len(scores)):
        if mask_array[:, i].any():
            scores[i] = k - (len(mask_array) - mask_array[:, i].sum()) * k / len(mask_array)

    num_non_feasible = len(scores[mask]) if mask.any() else 0
    return scores, num_non_feasible


def _coello_non_dominance(scores, variables, result_array):
    """
    Source:
    Coello, C. A. C. (2002). Theoretical and numerical constraint-handling techniques used with evolutionary algorithms:
    a survey of the state of the art. Computer methods in applied mechanics and engineering, 191(11-12), 1245-1287.

    :param scores: List with the scores from the objective function
    :param variables: array with all decoded members of the population
    :param result_array: List with the results from the constraints calculations in function _cons_handler
    :return: rank of all members in the population (see paper), number of non-feasible results
    """
    rank = zeros(len(variables))
    mask_array = list()
    mask = [False, ] * len(variables)
    exceedance_array = list()

    for eq, result in result_array:
        exceedance = zeros(len(result))
        if eq == 'ineq':
            mask = result > 0
            exceedance[mask] = result[mask]
            mask_array.append(mask)
            exceedance_array.append(exceedance)
        elif eq == 'eq':
            mask = abs(result) - 1e-4 > 0
            exceedance[mask] = abs(result[mask])
            mask_array.append(mask)
            exceedance_array.append(exceedance)

    mask_array = array(mask_array)
    violation_array = multiply(mask_array, 1).sum(axis=0)  # sum number of constraints violated
    coefficient_array = array(exceedance_array).sum(axis=0)  # sum of violation

    for item in mask_array:
        mask = mask | item

    for xi in range(len(variables)):
        feasibility_xi = list()
        for item in mask_array:
            if item[xi]:
                feasibility_xi.append(1)
            else:
                feasibility_xi.append(0)
        if sum(feasibility_xi) == 0:
            rank[xi] = scores[xi]
            continue
        else:
            for xj in range(len(variables)):
                if xi == xj:
                    continue
                elif mask[xj] is False:
                    rank[xi] += 1
                    continue
                elif violation_array[xi] > violation_array[xj]:
                    rank[xi] += 1
                    continue
                elif coefficient_array[xi] > coefficient_array[xj]:
                    rank[xi] += 1
                    continue
                else:
                    continue
            rank[xi] = 1 / rank[xi]

    rank[mask] += max(rank)
    num_non_feasible = mask.sum()
    assert len(scores) == len(rank), 'Fitness is not calculated correctly, size does not match scores'

    return rank, num_non_feasible


def _const_handler(handler, constraints, decoded, scores):
    """
    Initial function to handle the constraints. Calculates the results from all constraints and appends them to one
    list: result_array. Next, the use specified constraint handler is called in which the result_array is an argument.

    :param handler: simple or CND; the constraint handler the user wants to use
    :param constraints: list with type of constraint and the constraint itself as function ([[type, function], etc.])
    :param decoded: list with decoded members of the population
    :param scores: list with the result from the objective function
    :return: penalized scores, number of non-feasible results
    """
    if len(constraints) == 0:
        return scores, 0

    result_array = list()
    for eq, func in constraints:
        assert eq == 'eq' or eq == 'ineq', "Type of constraint should be 'eq' or 'ineq'"
        result_array.append([eq, func(decoded)])

    if handler == 'CND':
        scores, non_feasible_counter = _coello_non_dominance(scores, decoded, result_array)
    else:
        scores, non_feasible_counter = _static_penalty(scores, result_array)

    return scores, non_feasible_counter
