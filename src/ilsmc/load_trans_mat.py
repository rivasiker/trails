import numpy as np
import pandas as pd

def load_trans_mat(n_seq):
    """
    This functions returns a string matrix for the CTMC.
    
    Parameters
    ----------
    n_seq : integer
        Number of sequences of the CTMC
    """
    
    # Read string data frame for the CTMC
    df = pd.read_csv('trans_mat/trans_mat_full_'+str(n_seq)+'.csv')
    
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

def trans_mat_num(trans_mat, coal, rho):
    """
    This function returns a transition matrix given a 
    string matrix whose values are either '0', or 'R'
    or 'C' preceeded by a number, corresponding to a
    multiplication factor for the recombination and 
    the coalescence rate, respectively. The user can
    specify these two numerical rates. The function
    calculates the rates in the diagonals as  
    (-1)*rowSums
    
    Parameters
    ----------
    trans_mat : string numpy matrix
        Matrix containing coalescent and recombination
        rates in strings.
    coal : float
        Coalescent rate.
    rho : float
        Recombination rate
    """
    # Define the number of rows and columns from string matrix shape
    num_rows, num_cols = trans_mat.shape
    # Define numeric matrix of that shape
    trans_mat_num = np.full((num_rows, num_cols), 0.0)
    # For each row
    for i in range(num_rows):
        # For each column
        for j in range(num_cols):
            # If the string matrix is '0'
            if trans_mat[i,j] == '0':
                # Add numeric 0 to numeric matrix
                trans_mat_num[i,j] = 0.0
            # Otherwise
            else:
                # Multiply factor from string matrix to either coalescent or recombination rate
                trans_mat_num[i,j] = (coal if trans_mat[i,j]=='C' else rho)
    # Calculate diagonal as (-1)*rowSums
    for i in range(num_rows):
        trans_mat_num[i,i]=-sum(trans_mat_num[i])
    return trans_mat_num