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

def instant_mat(tup_0, tup_1, trans_mat):
    """
    This function returns the same transition rate matrix
    as supplied, except only the entries from tup_0 to
    tup_1 are kept, and all of the others are zeroed. This
    function is useful to calculate the instantaneous 
    transition rate matrix for the van Loan method, as well
    as for the integrals when the time interval goes to 
    infinity. 
    
    Parameters
    ----------
    tup_0 : list of integers
        These are the indices of the transition rate matrix
        corresponding to the starting states.
    tup_1 : list of integers
        These are the indices of the transition rate matrix
        corresponding to the end states.
    trans_mat : numeric numpy matrix
        Transition rate matrix.
    """
    n = trans_mat.shape[0]
    A_mat = np.zeros((n,n))
    for a in tup_0:
        for b in tup_1:
            A_mat[a,b] = trans_mat[a,b]
    return A_mat