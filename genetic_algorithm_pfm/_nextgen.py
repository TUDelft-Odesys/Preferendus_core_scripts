"""Functions that handle the selection, creation and mutation of the next generation in a genetic algorithm.

Copyright (c) Harold van Heukelum, 2021
"""

from numpy.random import randint, rand


def _selection(pop, scores, k=3):
    """
    Tournament selection

    Function that selects parents for next generation. The score of a random parent is taken as reference value. Next,
    a tournament is held to check if from a random list of scores, a better score can be obtained. If so, this becomes
    the parent to beat. The parent with the lowest score is returned in the end.

    The random selection means that a single member of the population can be selected zero, one, or multiple times!

    k defines the length of the tournament. A higher k means the tournament is taking more time, but the convergence
    rate might be higher.

    :param pop: array with all bitstrings (members of the population)
    :param scores: array with the results of the current population against the objective function
    :return: random bitstring in the population that is the winner of the tournament
    """

    # first random selection
    selection_ix = randint(len(pop))
    for ix in randint(0, len(pop), k - 1):
        # check if better (e.g. perform a tournament)
        if scores[ix] < scores[selection_ix]:
            selection_ix = ix
    return pop[selection_ix]


def _crossover(p1, p2, r_cross):
    """
    Crossover two parents to create two children.

    Function takes two parents as basis and creates two children. The cross ratio r_cross determines if the children
    are a direct copy of the parent or not. If this is not the case, then the first child is the inverse of the second
    child (ie. a 1 in child 1 is a 0 in child 2 and vise versa).

    :param p1: bitstring of parent 1
    :param p2: bitstring of parent 2
    :param r_cross: cross over ratio [0-1]
    :return: list of two children as bitstring
    """
    # children are copies of parents by default
    c1, c2 = p1.copy(), p2.copy()

    # check for recombination
    if rand() < r_cross:
        # select crossover point that is not on the end of the string (otherwise rand() would be similar to > r_cross)
        pt = randint(1, len(p1) - 2)

        # perform crossover
        c1 = p1[:pt] + p2[pt:]
        c2 = p2[:pt] + p1[pt:]
    return [c1, c2]


def _mutation(bitstring, r_mut):
    """
    Mutation operator

    :param bitstring: bitstring of child
    :param r_mut: mutation rate
    :return: None
    """
    for i in range(len(bitstring)):
        # check for a mutation

        if rand() < r_mut:
            # flip the bit
            bitstring[i] = 1 - bitstring[i]
