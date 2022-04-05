def get_ABC(trans_mat, times, omegas):
    """
    This function calculates the joint probabilities
    for the two-sequence CTMC.
    
    Parameters
    ----------
    trans_mat : numpy array
        The transition rate matrix of the two-sequence CTMC
    times : list of numbers
        Time intervals for each matrix multiplication
    omegas : list of lists
        Sets of states for each matrix multiplication
    """
    # Calculate first multiplication
    g = expm(trans_mat*times[0])[omegas[0]][:,omegas[1]]
    # For each of the remaining omegas
    for i in range(1, len(times)):
        # Perform multiplication
        g = g @ expm(trans_mat*times[i])[omegas[i]][:,omegas[i+1]]
    # Return a numpy array that contains the probabilities in the right order.
    return g