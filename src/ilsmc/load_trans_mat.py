def load_trans_mat(n_seq):
    """
    This functions returns a string matrix for the CTMC.
    
    Parameters
    ----------
    n_seq : integer
        Number of sequences of the CTMC
    """
    
    # Read string data frame for the CTMC
    df = pd.read_csv('../trans_mat/trans_mat_full_'+str(n_seq)+'.csv')
    
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