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

def get_tab_AB(state_space_AB, trans_mat_AB, cut_AB, pi_AB):
    """
    This functions returns a table with joint probabilities of
    the end probabilities per state after running a two-sequence
    CTMC segregated by the fate of each pair of sites. 
    
    Parameters
    ----------
    state_space_AB : list of lists of tuples
        States of the whole state space two-sequence CTMC
    trans_mat_AB : numeric numpy matrix
        Transition rate matrix of the two-sequence CTMC
    cut_AB : list of floats
        Ordered cutpoints of the two-sequence CTMC
    pi_AB : list of floats
        Starting probabilities after merging two one-sequence CTMCs. 
    """
    
    ###############################
    ### State-space information ###
    ###############################
    
    # Get flatten list of states, where even-indexed numbers (0, 2, ...)
    # represent the left-side coalescence states and odd-indexed numbers
    # (1, 3, ...) represent right-side coalescence.
    flatten = [list(sum(i, ())) for i in state_space_AB]
    # Get the index of all states where there is not a 2 (no coalescent)
    omega_B = [i for i in range(15) if 3 not in flatten[i]]
    # Get the index of all states where there is a 2 on left but not on right
    omega_L = [i for i in range(15) if (3 in flatten[i][::2]) and (3 not in flatten[i][1::2])]
    # Get the index of all states where there is a 2 on right but not on left
    omega_R = [i for i in range(15) if (3 not in flatten[i][::2]) and (3 in flatten[i][1::2])]
    # Get the index of all states where there is a 2 on left and right
    omega_E = [i for i in range(15) if (3 in flatten[i][::2]) and (3 in flatten[i][1::2])]
    # Get the index of all states
    omega_tot_AB = [i for i in range(15)]
    
        
    # Number of intervals
    n_int_AB = len(cut_AB)-1
    # Create empty table for the joint probabilities
    tab = np.zeros((n_int_AB*n_int_AB+n_int_AB*2+1, 15))
    # Create empty vector for the names of the states
    tab_names = []
    # Create accumulator for keeping track of the indices for the table
    acc = 0
    
    ############################################
    ### Deep coalescence -> deep coalescence ###
    ############################################
    
    # A pair of sites whose fate is to be of deep coalescence is represented as (('D'), ('D')).
    p_ABC = pi_AB @ get_ABC(trans_mat_AB, [cut_AB[-1]-cut_AB[0]], [omega_tot_AB, omega_B])
    tab[acc] = get_ordered(p_ABC, omega_B, omega_tot_AB)
    tab_names.append((('D'), ('D')))
    acc += 1
    
    
    
    ##############################
    ### V0 -> deep coalescence ###
    ### Deep coalescence -> V0 ###
    ##############################
    
    # A pair of sites where the left site is in V0 and the right site is of deep coalescence
    # is represented as ((0, L), ('D')), where L is the index of the interval where the first
    # left coalescent happens. Remember that the probability of ((0, L) -> ('D')) is the same as
    # that of (('D'), (0, L)).
    for L in range(n_int_AB):
        times = get_times(cut_AB, [0, L, L+1, -1])
        omegas = [omega_tot_AB, omega_B, omega_L, omega_L]
        p_ABC = pi_AB @ get_ABC(trans_mat_AB, times, omegas)
        p_ABC = get_ordered(p_ABC, omega_L, omega_tot_AB)
        tab[acc] = p_ABC
        tab_names.append(((0, L), ('D')))
        tab[acc+1] = p_ABC
        tab_names.append((('D'), (0, L)))
        acc += 2
        
    
        
    ################
    ### V0 -> V0 ###
    ################    
    
    # A pair of sites whose fate is to be V0 states is represented as ((0, L), (0, R)), where
    # L is the index of the interval where the first left coalescent happens, and R is the same
    # for the first right coalescent. Remember that the probability of ((0, L) -> (0, R)) equals
    # that of ((0, R), (0, L)).
    for L in range(n_int_AB):
        for R in range(L, n_int_AB):
            if R == L:
                times = get_times(cut_AB, [0, L, L+1, -1])
                omegas = [omega_tot_AB, omega_B, omega_E, omega_E]
                p_ABC = pi_AB @ get_ABC(trans_mat_AB, times, omegas)
                tab[acc] = get_ordered(p_ABC, omega_E, omega_tot_AB)
                tab_names.append(((0, L), (0, R)))
                acc += 1
            elif L < R:
                times = get_times(cut_AB, [0, L, L+1, R, R+1, -1])
                omegas = [omega_tot_AB, omega_B, omega_L, omega_L, omega_E, omega_E]
                p_ABC = pi_AB @ get_ABC(trans_mat_AB, times, omegas)
                p_ABC = get_ordered(p_ABC, omega_E, omega_tot_AB)
                tab[acc] = p_ABC
                tab_names.append(((0, L), (0, R)))
                tab[acc+1] = p_ABC
                tab_names.append(((0, R), (0, L)))
                acc += 2
                
    return tab_names, tab