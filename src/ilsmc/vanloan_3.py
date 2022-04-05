def vanloan_3(trans_mat, tup, omega_start, omega_end, time):
    """
    This function performs the van Loan (1978) method for 
    finding the integral of a series of 4 multiplying matrix
    exponentials. 
    
    Parameters
    ----------
    trans_mat : numeric numpy matrix
        Transition rate matrix.
    tup : tupple
        Tupple of size 4, where the first and second entries
        are lists with the indices of the transition rate 
        matrix to go from and to in the first instantaneous
        transition. Similarly, the third entry is the indices
        to go to from the indices of the second entry and so on.
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
    A_23 = instant_mat(tup[2], tup[3], trans_mat)
    C_mat_upper =  np.concatenate((trans_mat, A_01, np.zeros((n,n)),np.zeros((n,n))), axis = 1)
    C_mat_middle = np.concatenate((np.zeros((n,n)), trans_mat, A_12, np.zeros((n,n))), axis = 1)
    C_mat_lower = np.concatenate((np.zeros((n,n)), np.zeros((n,n)), trans_mat, A_23), axis = 1)
    C_mat_lowest = np.concatenate((np.zeros((n,n)), np.zeros((n,n)), np.zeros((n,n)), trans_mat), axis = 1)
    C_mat = np.concatenate((C_mat_upper, C_mat_middle, C_mat_lower, C_mat_lowest), axis = 0)
    res_test = (expm(C_mat*(time))[0:n,-n:])[omega_start][:,omega_end]
    return res_test