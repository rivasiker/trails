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

def cutpoints_ABC(n_int_ABC, coal_ABC):
    """
    This function returns a the cutpoints for the
    intervals for the three-sequence CTMC. The cutpoints
    will be defined by the quantiles of an exponential 
    distribution. 
    
    Parameters
    ----------
    n_int_ABC : integer
        Number of intervals in the three-sequence CTMC.
    coal_ABC : float
        coalescent rate of the three-sequence CTMC.
    """
    # Define probabilities for quantiles
    quantiles_AB = np.array(list(range(n_int_ABC+1)))/n_int_ABC
    # Get quantiles
    cut_ABC = expon.ppf(quantiles_AB, scale=1/coal_ABC)
    return cut_ABC