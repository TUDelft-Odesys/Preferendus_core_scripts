"""Copyright (c) 2022. Harold Van Heukelum"""
from numpy import ndarray, array, amax


def aggregate_max(w, p, goal):
    """

    :param w:
    :param p:
    :param goal:
    """
    assert type(w) is list or type(w) is ndarray, 'Weights should be of type list'
    assert round(sum(w), 4) == 1, 'Sum of weights should be equal to 1'
    assert type(p) is list or type(p) is ndarray, 'Function values p should be of type list or ndarray'
    p = array(p)

    if type(goal) is int or type(goal) is str:
        goal_array = [int(goal)] * len(w)
    elif type(goal) is list or type(goal) is ndarray:
        assert len(goal) == len(w), f'List with goal values should be of equal size as the weights ' \
                                    f'({len(goal)}, {len(w)})'
        goal_array = goal
    else:
        raise TypeError('Goal value(s) should either be an integer, string, list or ndarray')

    distance_array = list()
    for i in range(len(w)):
        weight = w[i]
        p_i = p[i, :]
        goal_i = goal_array[i]
        distance_array.append(weight * (goal_i - p_i))

    return amax(distance_array, axis=0).tolist()
