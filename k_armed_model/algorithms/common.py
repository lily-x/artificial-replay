import numpy as np
import random as rd

def conf_r(T, t, n_pulls):
    """ compute confidence radius """
    return np.sqrt(2*np.log(1+T) / n_pulls)

def rd_argmax(vector):
    """
    Compute random among eligible maximum indices
    :param vector: np.array
    :return: int, random index among eligible maximum indices
    """
    m = np.amax(vector)
    indices = np.nonzero(vector == m)[0]
    return rd.choice(indices)
