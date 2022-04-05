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

def get_ordered(p_ABC, omega_end, omega_tot):
    """
    This functions orders a list of probabilities given
    a list of its indices and a list with all possible
    indices. 
    
    Parameters
    ----------
    p_ABC : list of floats
        Probabilities to be ordered
    omega_end : list of integers
        Indices of the probabilities
    omega_tot : list of integers
        List of all possible indices
    """
    # Add probability to list if its index equals the position in
    #  omega_tot, otherwise add 0
    return np.array([p_ABC[omega_end.index(j)] if j in omega_end else 0 for j in omega_tot])