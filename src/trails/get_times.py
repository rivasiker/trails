def get_times(cut, intervals):
    """
    This functions returns a list of times representing
    the time within each of the specified intervals. It 
    does so by using a list of all possible cutpoints and
    a list of indices representing the interval cutpoints
    in order.
    
    Parameters
    ----------
    cut : list of floats
        List of ordered cutpoints
    intervals : list of integers
        Ordered indices of cutpoints
    """
    return [cut[intervals[i+1]]-cut[intervals[i]] for i in range(len(intervals)-1)]