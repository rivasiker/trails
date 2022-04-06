import numpy as np

def get_ordered(p_ABC, omega_end, omega_tot):
    """
    This functions orders a list of probabilities given
    a list of its indices and a list with all possible
    indices. 
    
    Parameters
    ----------
    p_ABC : list of floats
        Probabilities to be ordered
    omega_end : list of integers
        Indices of the probabilities
    omega_tot : list of integers
        List of all possible indices
    """
    # Add probability to list if its index equals the position in
    #  omega_tot, otherwise add 0
    return np.array([p_ABC[omega_end.index(j)] if j in omega_end else 0 for j in omega_tot])