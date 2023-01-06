import sys
import numpy as np
import pandas as pd
from scipy.linalg import expm
from scipy.stats import truncexpon
from scipy.stats import expon
from scipy.special import comb
import ast
import multiprocess as mp

from cutpoints_ABC import cutpoints_ABC
from get_ABC_inf_bis import get_ABC_inf_bis
from vanloan_3 import vanloan_3
from get_times import get_times
from get_tab_AB import get_tab_AB
from get_ordered import get_ordered
from vanloan_2 import vanloan_2
from cutpoints_AB import cutpoints_AB
from instant_mat import instant_mat
from vanloan_1 import vanloan_1
from get_ABC import get_ABC
from get_tab_ABC import get_tab_ABC
from get_joint_prob_mat import get_joint_prob_mat
from combine_states import combine_states
from load_trans_mat import load_trans_mat
from trans_mat_num import trans_mat_num

def load_trans_mat(n_seq):
    """
    This functions returns a string matrix for the CTMC.
    
    Parameters
    ----------
    n_seq : integer
        Number of sequences of the CTMC
    """
    
    # Read string data frame for the CTMC
    df = pd.read_csv('../02_state_space/trans_mats/trans_mat_full_'+str(n_seq)+'.csv')
    
    # Create dictionary with all CTMC state names and their index
    df_2 = {
        'names': pd.concat([df['from_str'], df['to_str']]),
        'values': pd.concat([df['from'], df['to']])
    }
    # Convert names to data frame, drop duplicated states and sort by index
    df_2 = pd.DataFrame(data=df_2).drop_duplicates().sort_values(by=['values'])
    
    # Pivot data frame to matrix, and fill missing values with '0'
    df_1 = df[['value', 'from', 'to']].pivot(index='from',columns='to',values='value').fillna('0')
    # Reset column names
    df_1.columns.name = None
    df_1 = df_1.reset_index().iloc[:, 1:]
    
    return np.array(df_1), list(df_2['names'])