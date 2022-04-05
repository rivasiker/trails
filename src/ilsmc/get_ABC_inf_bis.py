def get_ABC_inf_bis(trans_mat, times, omegas):
    """
    This function calculates the joint probabilities
    for the three-sequence CTMC. It is a wrapper function
    of get_ABC(), which outputs the same as get_ABC() if the
    last interval does not end in infinity. If instead the last
    interval contains infinity, then get_ABC() is run without
    the last interval. This is a trick because expm cannot 
    handle infinite values. However, the sum of the output will
    not change, because once we have reached the last interval,
    the coalescent within that time interval will equal to 1.
    
    Parameters
    ----------
    trans_mat : numpy array
        The transition rate matrix of the two-sequence CTMC
    times : list of numbers
        Time intervals for each matrix multiplication
    omegas : list of lists
        Sets of states for each matrix multiplication
    """
    if times[-1] == np.inf:
        g = get_ABC(trans_mat, times[:-1], omegas[:-1])
    else:
        g = get_ABC(trans_mat, times, omegas)
    return g