import pickle as pkl
import random

def init_worker(rand_id):
    print('Initializing...')
    print(rand_id)
    global shared_data
    with open(f"{rand_id}.pkl", 'rb') as pickle_file:
        shared_data = pkl.load(pickle_file)
    
def write_info_AB(pi_ABC, om, omega_tot_ABC, pr, cut_ABC, dct_num, trans_mat_ABC, n_int_AB, names_tab_AB):
    shared_data = (pi_ABC, om, omega_tot_ABC, pr, cut_ABC, dct_num, trans_mat_ABC, n_int_AB, names_tab_AB)
    rand_id = '%030x' % random.randrange(16**30)
    with open(f"{rand_id}.pkl", 'wb') as pickle_file:
        pkl.dump(shared_data, pickle_file)
    return rand_id

def write_info_ABC(pi, om, omega_tot_ABC, pr, cut_ABC, dct_num, trans_mat_ABC):
    shared_data = (pi, om, omega_tot_ABC, pr, cut_ABC, dct_num, trans_mat_ABC)
    rand_id = '%030x' % random.randrange(16**30)
    with open(f"{rand_id}.pkl", 'wb') as pickle_file:
        pkl.dump(shared_data, pickle_file)
    return rand_id
