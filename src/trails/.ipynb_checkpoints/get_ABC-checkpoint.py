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

def get_ABC(trans_mat, times, omegas):
    """
    This function calculates the joint probabilities
    for the two-sequence CTMC.
    
    Parameters
    ----------
    trans_mat : numpy array
        The transition rate matrix of the two-sequence CTMC
    times : list of numbers
        Time intervals for each matrix multiplication
    omegas : list of lists
        Sets of states for each matrix multiplication
    """
    # Calculate first multiplication
    g = expm(trans_mat*times[0])[omegas[0]][:,omegas[1]]
    # For each of the remaining omegas
    for i in range(1, len(times)):
        # Perform multiplication
        g = g @ expm(trans_mat*times[i])[omegas[i]][:,omegas[i+1]]
    # Return a numpy array that contains the probabilities in the right order.
    return g