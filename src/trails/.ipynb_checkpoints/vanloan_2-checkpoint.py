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

def vanloan_2(trans_mat, tup, omega_start, omega_end, time):
    """
    This function performs the van Loan (1978) method for 
    finding the integral of a series of 3 multiplying matrix
    exponentials. 
    
    Parameters
    ----------
    trans_mat : numeric numpy matrix
        Transition rate matrix.
    tup : tupple
        Tupple of size 3, where the first and second entries
        are lists with the indices of the transition rate 
        matrix to go from and to in the first instantaneous
        transition. Similarly, the third entry is the indices
        to go to from the indices of the second entry.
    omega_start : list of integers
        List of starting states of the transition rate matrix.
    omega_end : list of integers
        List of ending states of the transition rate matrix.
    time : float
        Upper boundary of the definite integral. 
    """
    n = trans_mat.shape[0]
    A_01 = instant_mat(tup[0], tup[1], trans_mat)
    A_12 = instant_mat(tup[1], tup[2], trans_mat)
    C_mat_upper =  np.concatenate((trans_mat, A_01, np.zeros((n,n))), axis = 1)
    C_mat_middle = np.concatenate((np.zeros((n,n)), trans_mat, A_12), axis = 1)
    C_mat_lower = np.concatenate((np.zeros((n,n)), np.zeros((n,n)), trans_mat), axis = 1)
    C_mat = np.concatenate((C_mat_upper, C_mat_middle, C_mat_lower), axis = 0)
    res_test = (expm(C_mat*(time))[0:n,-n:])[omega_start][:,omega_end]
    return res_test