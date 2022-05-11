"""
Constraint handlers for the GA

(c) Harold van Heukelum, 2022
"""

from numpy import array, zeros, multiply, ones

k = 1e9


def _static_penalty(scores, result_array):
    """
    Static penalty handler for GA.

    References:
    Morales, A. K., & Quezada, C. V. (1998, September). A universal eclectic genetic algorithm for constrained
    optimization. In Proceedings of the 6th European congress on intelligent techniques and soft computing
    (Vol. 1, pp. 518-522).

    :param scores: list with scores of objective evaluation
    :param result_array: list with result from calculating the constraints
    :return: penalized scores; number of non-feasible solutions
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
    Non-dominance constraint handler.

    References:
    Coello, C. A. C. (2000). Use of a self-adaptive penalty approach for engineering optimization problems.
    Computers in Industry, 41(2), 113-127.

    :param scores: list with scores of objective evaluation
    :param variables: list with decoded variables
    :param result_array: list with result from calculating the constraints
    :return: final scores (rank); number of non-feasible solutions
    """
    rank = ones(len(variables))
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
    Initial function for constraint handling. Calculate results from constraints function and call the correct handler.

    :param handler: type of handler to use ('simple' or 'CND')
    :param constraints: list with constraints and their type (format: [[type, func]])
    :param decoded: list with decoded variables
    :param scores: list with scores of objective evaluation
    :return: final scores; number of non-feasible solutions
    """
    if len(constraints) == 0:
        return scores, 0

    result_array = list()
    for eq, func in constraints:
        assert eq == 'eq' or eq == 'ineq', "Type of constraint should be 'eq' or 'ineq'"
        assert (callable(func)), 'Constraint function must be callable'
        result_array.append([eq, func(decoded)])

    if handler == 'CND':
        scores, non_feasible_counter = _coello_non_dominance(scores, decoded, result_array)
    else:
        scores, non_feasible_counter = _static_penalty(scores, result_array)

    return scores, non_feasible_counter
