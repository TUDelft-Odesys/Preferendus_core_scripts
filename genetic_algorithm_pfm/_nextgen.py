"""
Function to create next generation of GA

(c) Harold van Heukelum, 2022
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
    rate look higher. A sensitivity study will be conducted to see the influence of k.

    :param pop: array with all bitstrings (members of the population)
    :param scores: array with the results of the current population against the objective function
    :param k: length of tournament (default = 3)
    :return: random bitstring in the population that is the winner of the tournament
    """

    # first random selection
    selection_ix = randint(len(pop))
    for ix in randint(0, len(pop), int(len(pop)/2)):  # k - 1):
        # check if better (e.g. perform a tournament)
        if scores[ix] < scores[selection_ix]:
            selection_ix = ix
    return pop[selection_ix]


def _crossover_real(p1, p2, r_cross):
    """
    Function to create two new lists of bits from two lists from the current generation. Note that a higher r_cross
    results in a lower change current members are copied one-on-one.
    :param p1: first parent
    :param p2: second parent
    :param r_cross: crossover range (0 < r_cross < 1)
    :return: child_1, child_2
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
    return c1, c2


def _crossover(p1, p2, r_cross, approach):
    """
    Crossover two parents to create two children. Function takes two parents as basis and creates two children.

    In the case of integer parents:
        If a uniform random generated value [0,1] < 0.5, parent 1 will become child 2 and vise versa.
        Else, parent 1 is child 2 and vise versa.

    In the case of a bitstring:
        If r_cross is lower than a random generated value [0,1], a part of the first parent is combined with a part of
        the second parent to create a child. The other parts of the parents are combined to create a second child.

    :param p1: bitstring of parent 1
    :param p2: bitstring of parent 2
    :param r_cross: cross over ratio [0-1]
    :return: list of two children as bitstring
    """
    c1 = list()
    c2 = list()

    for i in range(len(p1)):
        if approach[i] in ['int', 'bool']:
            if rand() < 0.5:
                c1.append(p2[i])
                c2.append(p1[i])
            else:
                c1.append(p1[i])
                c2.append(p2[i])
        else:
            sub_c1, sub_c2 = _crossover_real(p1[i], p2[i], r_cross)
            c1.append(sub_c1)
            c2.append(sub_c2)
    return [c1, c2]


def _mutation(member, r_mut, approach, bounds):
    """
    Mutation operator

    If the member of the population is an integer:
        If a random generated number [0,1] < r_mut, a random generated integer [-1,1] is added to the integer.

    If the member of the population is a bitstring; for every bit in the bitstring:
        If a random generated number [0,1] < r_mut, the bit is flipped.

    :param member: bitstring of child
    :param r_mut: mutation rate
    :param approach: list with type of variables
    :param bounds: list with bounds to check integer mutation against
    :return: None
    """
    for i in range(len(member)):
        if approach[i] == 'int':
            # check for a mutation
            if rand() < r_mut:
                member[i] = randint(bounds[i][0], bounds[i][1])
        elif approach[i] == 'bool':
            # check for a mutation
            if rand() < r_mut:
                member[i] = 1 - member[i]  # flip the boolean
        else:
            for j in range(len(member[i])):
                # check for a mutation
                if rand() < r_mut:
                    # flip the bit
                    member[i][j] = 1 - member[i][j]
