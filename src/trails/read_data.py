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
