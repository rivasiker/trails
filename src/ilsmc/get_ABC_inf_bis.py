import sys
import numpy as np
import pandas as pd
from scipy.linalg import expm
from scipy.stats import truncexpon
from scipy.stats import expon
from scipy.special import comb
import ast
import multiprocessing as mp

from cutpoints_ABC import cutpoints_ABC
from get_ABC_inf_bis import get_ABC_inf_bis
from vanloan_3 import vanloan_3
from get_times import get_times
from get_tab_AB import get_tab_AB
from get_ordered import get_ordered
from vanloan_2 import vanloan_2
from cutpoints_AB import cutpoints_AB
from instant_mat import instant_mat
from vanloan_1 import vanloan_1
from get_ABC import get_ABC
from get_tab_ABC import get_tab_ABC
from get_joint_prob_mat import get_joint_prob_mat
from combine_states import combine_states
from load_trans_mat import load_trans_mat
from trans_mat_num import trans_mat_num

def get_ABC_inf_bis(trans_mat, times, omegas):
    """
    This function calculates the joint probabilities
    for the three-sequence CTMC. It is a wrapper function
    of get_ABC(), which outputs the same as get_ABC() if the
    last interval does not end in infinity. If instead the last
    interval contains infinity, then get_ABC() is run without
    the last interval. This is a trick because expm cannot 
    handle infinite values. However, the sum of the output will
    not change, because once we have reached the last interval,
    the coalescent within that time interval will equal to 1.
    
    Parameters
    ----------
    trans_mat : numpy array
        The transition rate matrix of the two-sequence CTMC
    times : list of numbers
        Time intervals for each matrix multiplication
    omegas : list of lists
        Sets of states for each matrix multiplication
    """
    if times[-1] == np.inf:
        g = get_ABC(trans_mat, times[:-1], omegas[:-1])
    else:
        g = get_ABC(trans_mat, times, omegas)
    return g