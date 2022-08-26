"""Copyright (c) 2022. Harold Van Heukelum"""
from numpy import ndarray, array, amax


def aggregate_max(w, p, goal):
    """
    Simple function to aggregate preference scores via the minmax method.

    :param w: weights of criteria
    :param p: population
    :param goal: goal value
    :return: per alternative the score of the criteria that is furthest from the goal. Can then be minimized with GA
    """
    assert type(w) is list or type(w) is ndarray, 'Weights should be of type list'
    assert round(sum(w), 4) == 1, 'Sum of weights should be equal to 1'
    assert type(p) is list or type(p) is ndarray, 'Function values p should be of type list or ndarray'

    # goal can be either a single value or a list with different values for all criteria
    if type(goal) is int or type(goal) is str:
        goal_array = [int(goal)] * len(w)
    elif type(goal) is list or type(goal) is ndarray:
        assert len(goal) == len(w), f'List with goal values should be of equal size as the weights ' \
                                    f'({len(goal)}, {len(w)})'
        goal_array = goal
    else:
        raise TypeError('Goal value(s) should either be an integer, string, list or ndarray')

    assert round(sum(w), 4) == 1, 'Sum of weights should be equal to 1'
    p = array(p)  # if p is given as list

    distance_array = list()  # list that will contain the distances from the goal per criteria
    for i in range(len(w)):
        weight = w[i]
        p_i = p[i, :]
        goal_i = goal_array[i]
        distance_array.append(weight * (goal_i - p_i))

    return amax(distance_array, axis=0).tolist()  # see docstring
