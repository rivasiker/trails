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