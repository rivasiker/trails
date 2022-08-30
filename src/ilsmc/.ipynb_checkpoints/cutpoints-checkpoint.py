import numpy as np
from scipy.stats import truncexpon
from scipy.stats import expon

def cutpoints_AB(n_int_AB, t_AB, coal_AB):
    """
    This function returns a the cutpoints for the
    intervals for the two-sequence CTMC. The cutpoints
    will be defined by the quantiles of a truncated
    exponential distribution. 
    
    Parameters
    ----------
    n_int_AB : integer
        Number of intervals in the two-sequence CTMC.
    t_AB : float
        Total time interval of the two-sequence CTMC
    coal_AB : float
        coalescent rate of the two-sequence CTMC.
    """
    # Define probabilities for quantiles
    quantiles_AB = np.array(list(range(n_int_AB+1)))/n_int_AB
    # Define truncexpon shape parameters
    lower, upper, scale = 0, t_AB, 1/coal_AB
    # Get quantiles
    cut_AB = truncexpon.ppf(quantiles_AB, b=(upper-lower)/scale, loc=lower, scale=scale)
    return cut_AB 

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