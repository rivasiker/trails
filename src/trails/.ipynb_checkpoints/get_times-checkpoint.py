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

def get_times(cut, intervals):
    """
    This functions returns a list of times representing
    the time within each of the specified intervals. It 
    does so by using a list of all possible cutpoints and
    a list of indices representing the interval cutpoints
    in order.
    
    Parameters
    ----------
    cut : list of floats
        List of ordered cutpoints
    intervals : list of integers
        Ordered indices of cutpoints
    """
    return [cut[intervals[i+1]]-cut[intervals[i]] for i in range(len(intervals)-1)]