"""
Constraint handlers for the GA. For references to the used constraint handlers, see the docstrings of the functions
itself.

(c) Harold van Heukelum, 2022
"""

from numpy import array, zeros, multiply, ones

k = 1e9  # factor k of static penalty function


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
    scores = array(scores)  # make np.ndarray of list with scores
    mask_array = list()  # create empty list to store all masks into
    mask = [False, ] * len(scores)  # create default mask-list, to prevent referenced before assignment error.

    """
    Here we check if the result of the constraints are feasible or not. the result is an array of the same length as 
    the result array, but now only with boolean values: true when non-feasible, and false otherwise. This boolean array 
    is than added to the mask_array list.
    """
    for eq, result in result_array:
        if eq == 'ineq':
            mask_array.append(array(result) > 0)
        elif eq == 'eq':
            mask_array.append(abs(array(result)) - 1e-4 > 0)

    mask_array = array(mask_array)  # make from the list and np.ndarray

    """
    Combine all separate mask arrays into one big one. This will make it clear which members in the population violate 
    a constraint, but not the number of violated constraints.
    """
    for item in mask_array:
        mask = mask | item

    # set penalty for non-feasible members
    for i in range(len(scores)):
        if mask_array[:, i].any():
            scores[i] = k - (len(mask_array) - mask_array[:, i].sum()) * k / len(mask_array)

    # count non-feasible solutions
    num_non_feasible = mask.sum()

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
    rank = ones(len(variables))  # default rank is 1
    mask_array = list()  # empty list to store masks
    mask = [False, ] * len(variables)  # create default mask-list, to prevent referenced before assignment error.
    exceedance_array = list()  # create empty list to store exceedence values into

    """
    For this constraint handler, we do not only need to know which members are non-feasible, but also what their 
    exceedence is and the number constraints they violate. hence, in the checks below, we store already the level of 
    exceedance. The number of violated constraints can be determined by using the mask arrays (see below)  
    """
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

    """
    Now we know the exceedance, we can check the number of violated constraints. This is done by summation of all 
    True-valued entries. Secondly, the exceedance is summarized per member of the population.
    """
    mask_array = array(mask_array)
    violation_array = multiply(mask_array, 1).sum(axis=0)  # sum number of constraints violated
    coefficient_array = array(exceedance_array).sum(axis=0)  # sum of violation

    """
    Combine all separate mask arrays into one big one. This will make it clear which members in the population violate 
    a constraint, but not the number of violated constraints.
    """
    for item in mask_array:
        mask = mask | item

    """
    Below, the sequence to determine the rank is performed for all members in the population. The interested reader 
    is referred to the reference paper to read about the sequence
    """
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

    # the rank-scores need to be heightened by the maximal rank. Otherwise, the scores of non-feasible members
    # might be lower than for feasible members.
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
    if len(constraints) == 0:  # there are no constraint
        return scores, 0

    """
    Below, the results of the constraints are saved into an array, together with the information about the type of 
    constraint. This can than be used by the specific handlers.
    """
    result_array = list()
    for eq, func in constraints:
        assert eq == 'eq' or eq == 'ineq', "Type of constraint should be 'eq' or 'ineq'"
        assert (callable(func)), 'Constraint function must be callable'
        result_array.append([eq, func(decoded)])

    # call correct constraint handler
    if handler == 'CND':
        scores, non_feasible_counter = _coello_non_dominance(scores, decoded, result_array)
    else:
        scores, non_feasible_counter = _static_penalty(scores, result_array)

    return scores, non_feasible_counter
