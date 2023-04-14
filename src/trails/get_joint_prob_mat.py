import numpy as np
from scipy.linalg import expm
from ast import literal_eval
from trails.cutpoints import cutpoints_AB, cutpoints_ABC
from trails.load_trans_mat import load_trans_mat, trans_mat_num
from trails.combine_states import combine_states
from trails.get_tab import get_tab_AB, get_tab_ABC

def get_joint_prob_mat(t_A,    t_B,    t_AB,    t_C, 
                      rho_A,  rho_B,  rho_AB,  rho_C,  rho_ABC, 
                      coal_A, coal_B, coal_AB, coal_C, coal_ABC,
                      n_int_AB, n_int_ABC):
    
    """
    This is a wrapper function that unifies all the CTMCs to 
    get a matrix of joint probabilities for the HMM. 
    
    Parameters
    ----------
    t_A : float
        Time for the one-sequence CTMC of the first sequence (A)
    t_B : float
        Time for the one-sequence CTMC of the second sequence (B)
    t_AB : float
        Time for the two-sequence CTMC (AB)
    t_C : float
        Time for the one-sequence CTMC of the third sequence (C)
    rho_A : float
        Recombination rate for the one-sequence CTMC of the first sequence (A)  
    rho_B : float
        Recombination rate for the one-sequence CTMC of the second sequence (B)  
    rho_AB : float
        Recombination rate for the two-sequence CTMC (AB)  
    rho_C : float
        Recombination rate for the one-sequence CTMC of the third sequence (C)    
    rho_ABC : float
        Recombination rate for the three-sequence CTMC (ABC)  
    coal_A : float
        Coalescent rate for the one-sequence CTMC of the first sequence (A)  
    coal_B : float
        Coalescent rate for the one-sequence CTMC of the second sequence (B)  
    coal_AB : float
        Coalescent rate for the two-sequence CTMC (AB)  
    coal_C : float
        Coalescent rate for the one-sequence CTMC of the third sequence (C)    
    coal_ABC : float
        Coalescent rate for the three-sequence CTMC (ABC)  
    n_int_AB : integer
        Number of intervals of the two-sequence CTMC (AB)
    n_int_ABC : integer
        Number of intervals of the three-sequence CTMC (ABC)
    """
    
    ####################################
    ### Load state-space information ###
    ####################################
    
    # Load string transition rate matrix and convert string names to actual lists
    (trans_mat_1, state_space_1) = load_trans_mat(1)
    state_space_A = [literal_eval(i) for i in state_space_1]
    (trans_mat_2, state_space_2) = load_trans_mat(2)
    state_space_AB = [literal_eval(i) for i in state_space_2]
    (trans_mat_3, state_space_3) = load_trans_mat(3)
    state_space_ABC = [literal_eval(i) for i in state_space_3]
    
    
    ##########################
    ### One-sequence CTMCs ###
    ##########################
    
    # Convert string transition rate matrices to numeric matrices
    # These are (2x2) matrices
    trans_mat_A = trans_mat_num(trans_mat_1, coal_A, rho_A)
    trans_mat_B = trans_mat_num(trans_mat_1, coal_B, rho_B)
    trans_mat_C = trans_mat_num(trans_mat_1, coal_C, rho_C)
    
    # Run the one-sequence CTMCs
    # These are (1x2) vectors
    # with objmode(final_A='float64[:,:]'):
    #     final_A = expm(trans_mat_A*t_A)[0]
    # with objmode(final_B='float64[:,:]'):
    #     final_B = expm(trans_mat_B*t_B)[0]
    # with objmode(final_C='float64[:,:]'):
    #     final_C = expm(trans_mat_C*t_C)[0]
    final_A = expm(trans_mat_A*t_A)[0]
    final_B = expm(trans_mat_B*t_B)[0]
    final_C = expm(trans_mat_C*t_C)[0]
    
    state_space_B = []
    for j in state_space_A:
        lst = []
        for i in j:
            one = 2 if i[0] == 1 else i[0]
            two = 2 if i[1] == 1 else i[1]
            lst.append((one, two))
        state_space_B.append(lst)
    
    # Combine A and B CTMCs
    (comb_AB_name, comb_AB_value) = combine_states(
        state_space_A, 
        state_space_B, 
        final_A, 
        final_B)
    # Order states
    pi_AB = [comb_AB_value[comb_AB_name.index(i)] if i in comb_AB_name else 0 for i in state_space_2]
    
        
    #########################
    ### Two-sequence CTMC ###
    #########################
    
    # Calculate cutpoints
    if cut_AB == 'standard':
        cut_AB = cutpoints_AB(n_int_AB, t_AB, coal_AB)
    # Convert string transition rate matrix to numeric matrix
    # This is a (15x15) matrix
    trans_mat_AB = trans_mat_num(trans_mat_2, coal_AB, rho_AB)
    
    
    # Run the two-sequence CTMC
    (names_tab_AB, tab_AB) = get_tab_AB(state_space_AB, trans_mat_AB, cut_AB, pi_AB)
    
    

    state_space_C = []
    for j in state_space_A:
        lst = []
        for i in j:
            one = 4 if i[0] == 1 else i[0]
            two = 4 if i[1] == 1 else i[1]
            lst.append((one, two))
        state_space_C.append(lst)
        
    # Define function wrapper for ordering each row in the joint probability matrix
    def comb_wrapper(x):
        # Combine AB and C CTMCs
        (comb_ABC_name, comb_ABC_value) = combine_states(state_space_AB, state_space_C, x, final_C)
        # Order states
        pi_ABC = [comb_ABC_value[comb_ABC_name.index(i)] if i in comb_ABC_name else 0 for i in state_space_3]
        return pi_ABC
    # Apply function row-wise
    pi_ABC = np.apply_along_axis(comb_wrapper, axis=1, arr=tab_AB)
    
        
    ###########################
    ### Three-sequence CTMC ###
    ###########################
    
    # Calculate cutpoints
    if cut_ABC == 'standard':
        cut_ABC = cutpoints_ABC(n_int_ABC, coal_ABC)
    # Convert string transition rate matrix to numeric matrix
    # This is a (31x31) matrix
    trans_mat_ABC = trans_mat_num(trans_mat_3, coal_ABC, rho_ABC)
    
    
    # Run the three-sequence CTMC
    tab_2 = get_tab_ABC(state_space_ABC, trans_mat_ABC, cut_ABC, pi_ABC, names_tab_AB, n_int_AB)
    
    return tab_2
