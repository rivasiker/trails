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