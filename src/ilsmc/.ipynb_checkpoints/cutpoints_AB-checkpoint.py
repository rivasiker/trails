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
def cutpoints_AB(n_int_AB, t_AB, coal_AB):
    """
    This function returns a the cutpoints for the
    intervals for the two-sequence CTMC. The cutpoints
    will be defined by the quantiles of a truncated
    exponential distribution. 
    
    Parameters
    ----------
    n_int_AB : integer
        Number of intervals in the two-sequence CTMC.
    t_AB : float
        Total time interval of the two-sequence CTMC
    coal_AB : float
        coalescent rate of the two-sequence CTMC.
    """
    # Define probabilities for quantiles
    quantiles_AB = np.array(list(range(n_int_AB+1)))/n_int_AB
    # Define truncexpon shape parameters
    lower, upper, scale = 0, t_AB, 1/coal_AB
    # Get quantiles
    cut_AB = truncexpon.ppf(quantiles_AB, b=(upper-lower)/scale, loc=lower, scale=scale)
    return cut_AB 