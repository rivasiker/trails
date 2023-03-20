from numba import njit
import numpy as np

@njit
def get_obs_state_dct():
    lst = []
    for a in ['A', 'C', 'T', 'G']:
        for b in ['A', 'C', 'T', 'G']:
            for c in ['A', 'C', 'T', 'G']:
                for d in ['A', 'C', 'T', 'G']:
                    lst.append(a+b+c+d)
    for a in ['A', 'C', 'T', 'G', 'N']:
        for b in ['A', 'C', 'T', 'G', 'N']:
            for c in ['A', 'C', 'T', 'G', 'N']:
                for d in ['A', 'C', 'T', 'G', 'N']:
                    if (a+b+c+d) not in lst:
                        lst.append(a+b+c+d)
    return lst

@njit
def get_idx_state(state):
    lst = get_obs_state_dct()
    st = lst[state]
    idx = st.find('N')
    if idx == -1:
        return np.array([state])
    else:
        return np.concatenate((
            get_idx_state(lst.index(st[:idx] + 'A' + st[idx+1:])),
            get_idx_state(lst.index(st[:idx] + 'C' + st[idx+1:])),
            get_idx_state(lst.index(st[:idx] + 'T' + st[idx+1:])),
            get_idx_state(lst.index(st[:idx] + 'G' + st[idx+1:])),
        ))
    
def maf_to_loglik(a, b, pi, file):
    """
    Posterior probabilities.
    
    Parameters
    ----------
    a : numpy array
        Transition probability matrix
    b : numpy array
        Emission probability matrix
    pi : numpy array
        Vector of starting probabilities of the hidden states
    file : str
        Path to MAF file
    """
    # Start loglik accumulator
    loglik_acc = 0
    # For each block
    for multiple_alignment in AlignIO.parse(file, "maf"):
        # Save sequence
        dct = {}
        for seqrec in multiple_alignment:
            if seqrec.name.split('.')[0] in sp_lst:
                dct[seqrec.name.split('.')[0]] = str(seqrec.seq).replace('-', 'N')
        # Convert sequence to index
        idx_lst = np.zeros((len(seqrec.seq)), dtype = np.int64)
        for i in range(len(seqrec.seq)):
            idx_lst[i] = order_st.index(''.join([dct[j][i] for j in sp_lst]).upper())
        # Compute loglik
        loglik_acc += forward_loglik(a, b, pi, idx_lst)
    return loglik_acc
