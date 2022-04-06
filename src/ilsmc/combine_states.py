import pandas as pd

def combine_states(iter_lst_a, iter_lst_b, probs_a, probs_b):
    """
    Given two lists of states and their probabilities, this
    function returns a list of combined states and their
    probabilities. 
    
    Each state is represented by a list of tuples. Each
    tuple of size 2 corresponds to two consecutive sites, so
    left and right respectively. 0 corresponds to no site, 1
    to an uncoalesced site, 2 to a site with one coalescent 
    from 2 sites, 3 to 2 coalescents from 3 sites, and so on.
    For example, (0, 1) corresponds to (  -o), while (1, 2)
    represents (o--x). A list of tuples, thus, represents a 
    state of a CTMC.
    
    Parameters
    ----------
    iter_lst_a : list of lists of tuples
        Each nested list represents the CTMC states
    iter_lst_b : list of lists of tuples
        Each nested list represents the CTMC states
    probs_a : list
        Probabilities corresponding to iter_lst_a
    probs_a : list
        Probabilities corresponding to iter_lst_b
    """
    iter_lst_ab = []
    probs_ab = []
    for i in range(len(iter_lst_a)):
        for j in range(len(iter_lst_b)):
            iter_lst_ab.append(sorted(iter_lst_a[i]+iter_lst_b[j]))
            probs_ab.append(probs_a[i]*probs_b[j])
    # Define new data frame
    df = pd.DataFrame()
    # Save names of state
    df['name'] = [str(i) for i in iter_lst_ab]
    # Save probabilities
    df['value'] = probs_ab
    # Group by state and sum probabilities
    df = df.groupby("name", as_index=False).sum()
    return list(df['name']), list(df['value'])