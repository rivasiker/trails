def cutpoints_ABC(n_int_ABC, coal_ABC):
    """
    This function returns a the cutpoints for the
    intervals for the three-sequence CTMC. The cutpoints
    will be defined by the quantiles of an exponential 
    distribution. 
    
    Parameters
    ----------
    n_int_ABC : integer
        Number of intervals in the three-sequence CTMC.
    coal_ABC : float
        coalescent rate of the three-sequence CTMC.
    """
    # Define probabilities for quantiles
    quantiles_AB = np.array(list(range(n_int_ABC+1)))/n_int_ABC
    # Get quantiles
    cut_ABC = expon.ppf(quantiles_AB, scale=1/coal_ABC)
    return cut_ABC