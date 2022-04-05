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

def trans_mat_num(trans_mat, coal, rho):
    """
    This function returns a transition matrix given a 
    string matrix whose values are either '0', or 'R'
    or 'C' preceeded by a number, corresponding to a
    multiplication factor for the recombination and 
    the coalescence rate, respectively. The user can
    specify these two numerical rates. The function
    calculates the rates in the diagonals as  
    (-1)*rowSums
    
    Parameters
    ----------
    trans_mat : string numpy matrix
        Matrix containing coalescent and recombination
        rates in strings.
    coal : float
        Coalescent rate.
    rho : float
        Recombination rate
    """
    # Define the number of rows and columns from string matrix shape
    num_rows, num_cols = trans_mat.shape
    # Define numeric matrix of that shape
    trans_mat_num = np.full((num_rows, num_cols), 0.0)
    # For each row
    for i in range(num_rows):
        # For each column
        for j in range(num_cols):
            # If the string matrix is '0'
            if trans_mat[i,j] == '0':
                # Add numeric 0 to numeric matrix
                trans_mat_num[i,j] = 0.0
            # Otherwise
            else:
                # Multiply factor from string matrix to either coalescent or recombination rate
                trans_mat_num[i,j] = (coal if trans_mat[i,j]=='C' else rho)
    # Calculate diagonal as (-1)*rowSums
    for i in range(num_rows):
        trans_mat_num[i,i]=-sum(trans_mat_num[i])
    return trans_mat_num