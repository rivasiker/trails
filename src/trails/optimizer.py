import numpy as np
import pandas as pd
from trails.get_emission_prob_mat import get_emission_prob_mat, get_emission_prob_mat_introgression
from trails.get_joint_prob_mat import get_joint_prob_mat
from trails.get_joint_prob_mat_introgression import get_joint_prob_mat_introgression
from scipy.optimize import minimize
from trails.cutpoints import cutpoints_ABC
from numba import njit
from trails.read_data import get_obs_state_dct_new_method
import time
from trails.read_data import get_idx_state, get_idx_state_new_method
from numba.typed import List
from ray.util.multiprocessing import Pool
import multiprocessing as mp
import os


def forward_loglik_par(a, b, pi, V, order):
    """
    Log-likelihood (parallelized)
    
    Parameters
    ----------
    a : numpy array
        Transition probability matrix
    b : numpy array
        Emission probability matrix
    pi : numpy array
        Vector of starting probabilities of the hidden states
    V : numpy array
        Vector of observed states, as integer indices
    """
    order = List(order)
    return forward_loglik(a, b, pi, V, order)

def loglik_wrapper_par(a, b, pi, V_lst):
    """
    Log-likelihood wrapper (parallelized)
    
    Parameters
    ----------
    a : numpy array
        Transition probability matrix
    b : numpy array
        Emission probability matrix
    pi : numpy array
        Vector of starting probabilities of the hidden states
    V : list of numpy arrays
        List of vectors of observed states, as integer indices
    """
    order = list()
    for i in range(624+1):
        order.append(get_idx_state(i))
    pool_lst = []
    for i in range(len(V_lst)):
        pool_lst.append((a, b, pi, V_lst[i], order))
    try:
        ncpus = int(os.environ["SLURM_JOB_CPUS_PER_NODE"])
    except KeyError:
        ncpus = mp.cpu_count()
    pool = Pool(ncpus)
    res = pool.starmap_async(forward_loglik_par, pool_lst)
    acc = 0
    for i in res.get():
        acc += i
    return acc

def loglik_wrapper_par_new_method(a, b, pi, V_lst):
    """
    Log-likelihood wrapper (parallelized)
    
    Parameters
    ----------
    a : numpy array
        Transition probability matrix
    b : numpy array
        Emission probability matrix
    pi : numpy array
        Vector of starting probabilities of the hidden states
    V : list of numpy arrays
        List of vectors of observed states, as integer indices
    """
    order = list()
    for i in range(125):
        order.append(get_idx_state_new_method(i))
    pool_lst = []
    for i in range(len(V_lst)):
        pool_lst.append((a, b, pi, V_lst[i], order))
    try:
        ncpus = int(os.environ["SLURM_JOB_CPUS_PER_NODE"])
    except KeyError:
        ncpus = mp.cpu_count()
    pool = Pool(ncpus)
    res = pool.starmap_async(forward_loglik_par, pool_lst)
    acc = 0
    for i in res.get():
        acc += i
    return acc

def loglik_wrapper(a, b, pi, V_lst):
    """
    Log-likelihood wrapper.
    
    Parameters
    ----------
    a : numpy array
        Transition probability matrix
    b : numpy array
        Emission probability matrix
    pi : numpy array
        Vector of starting probabilities of the hidden states
    V : list of numpy arrays
        List of vectors of observed states, as integer indices
    """
    order = List()
    for i in range(624+1):
        order.append(get_idx_state(i))
    acc = 0
    prev_time = time.time()
    events_count = len(V_lst)
    for i in range(len(V_lst)):
        acc += forward_loglik(a, b, pi, V_lst[i], order)
        if (time.time() - prev_time) > 1:
            # print('{}%'.format(round(100*(i/events_count), 3)), end = '\r')
            prev_time = time.time()   
    return acc

def loglik_wrapper_new_method(a, b, pi, V_lst):
    """
    Log-likelihood wrapper.
    
    Parameters
    ----------
    a : numpy array
        Transition probability matrix
    b : numpy array
        Emission probability matrix
    pi : numpy array
        Vector of starting probabilities of the hidden states
    V : list of numpy arrays
        List of vectors of observed states, as integer indices
    """
    order = List()
    for i in range(125):
        order.append(get_idx_state_new_method(i))
    acc = 0
    prev_time = time.time()
    events_count = len(V_lst)
    for i in range(len(V_lst)):
        acc += forward_loglik(a, b, pi, V_lst[i], order)
        if (time.time() - prev_time) > 1:
            # print('{}%'.format(round(100*(i/events_count), 3)), end = '\r')
            prev_time = time.time()   
    return acc

@njit
def forward_loglik(a, b, pi, V, order):
    """
    Log-likelihood.
    
    Parameters
    ----------
    a : numpy array
        Transition probability matrix
    b : numpy array
        Emission probability matrix
    pi : numpy array
        Vector of starting probabilities of the hidden states
    V : numpy array
        Vector of observed states, as integer indices
    """
    alpha = forward(a, b, pi, V, order)
    x = alpha[-1, :].max()
    return np.log(np.exp(alpha[len(V)-1]-x).sum())+x

@njit
def forward(a, b, pi, V, order):
    """
    Forward algorithm, that allows for missing data.
    
    Parameters
    ----------
    a : numpy array
        Transition probability matrix
    b : numpy array
        Emission probability matrix
    pi : numpy array
        Vector of starting probabilities of the hidden states
    V : numpy array
        Vector of observed states, as integer indices
    """
    alpha = np.zeros((V.shape[0], a.shape[0]))
    alpha[0, :] = np.log(pi * b[:, order[V[0]]].sum(axis = 1))
    for t in range(1, V.shape[0]):
        x = alpha[t-1, :].max()
        alpha[t, :] = np.log((np.exp(alpha[t - 1]-x) @ a) * b[:, order[V[t]]].sum(axis = 1))+x
    return alpha

@njit
def backward(a, b, V, order):
    """
    Backward algorithm.
    
    Parameters
    ----------
    a : numpy array
        Transition probability matrix
    b : numpy array
        Emission probability matrix
    V : numpy array
        Vector of observed states, as integer indices
    """
    beta = np.zeros((V.shape[0], a.shape[0]))
    beta[V.shape[0] - 1] = np.zeros((a.shape[0]))
    for t in range(V.shape[0] - 2, -1, -1):
        x = beta[t+1, :].max()
        beta[t, :] = np.log((np.exp(beta[t + 1]-x) * b[:, order[V[t+1]]].sum(axis = 1)) @ a)+x
    return beta


def post_prob(a, b, pi, V, order):
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
    V : numpy array
        Vector of observed states, as integer indices
    """
    alpha = forward(a, b, pi, V, order)
    beta = backward(a, b, V, order)
    post_prob = (alpha+beta)
    max_row = post_prob.max(1).reshape(-1, 1)
    post_prob = np.exp(post_prob-max_row)/np.exp(post_prob-max_row).sum(1).reshape(-1, 1)
    return post_prob

def post_prob_wrapper(a, b, pi, V_lst):
    """
    Posterior probability wrapper.
    
    Parameters
    ----------
    a : numpy array
        Transition probability matrix
    b : numpy array
        Emission probability matrix
    pi : numpy array
        Vector of starting probabilities of the hidden states
    V : list of numpy arrays
        List of vectors of observed states, as integer indices
    """
    res_lst = []
    order = List()
    for i in range(624+1):
        order.append(get_idx_state(i))
    prev_time = time.time()
    events_count = len(V_lst)
    for i in range(len(V_lst)):
        res_lst.append(post_prob(a, b, pi, V_lst[i], order))
    return res_lst

def post_prob_par(a, b, pi, V, order):
    """
    Log-likelihood (parallelized)
    
    Parameters
    ----------
    a : numpy array
        Transition probability matrix
    b : numpy array
        Emission probability matrix
    pi : numpy array
        Vector of starting probabilities of the hidden states
    V : numpy array
        Vector of observed states, as integer indices
    """
    order = List(order)
    return post_prob(a, b, pi, V, order)

def post_prob_wrapper_par(a, b, pi, V_lst):
    """
    Posterior decoding wrapper (parallelized)
    
    Parameters
    ----------
    a : numpy array
        Transition probability matrix
    b : numpy array
        Emission probability matrix
    pi : numpy array
        Vector of starting probabilities of the hidden states
    V : list of numpy arrays
        List of vectors of observed states, as integer indices
    """
    order = list()
    for i in range(624+1):
        order.append(get_idx_state(i))
    pool_lst = []
    for i in range(len(V_lst)):
        pool_lst.append((a, b, pi, V_lst[i], order))
    try:
        ncpus = int(os.environ["SLURM_JOB_CPUS_PER_NODE"])
    except KeyError:
        ncpus = mp.cpu_count()
    pool = Pool(ncpus)
    res = pool.starmap_async(post_prob_par, pool_lst)
    acc = []
    for i in res.get():
        acc.append(i)
    return acc

@njit
def viterbi_old(a, b, pi, V, order):
    """
    Viterbi path
    
    Parameters
    ----------
    a : numpy array
        Transition probability matrix
    b : numpy array
        Emission probability matrix
    pi : numpy array
        Vector of starting probabilities of the hidden states
    V : numpy array
        Vector of observed states
    """
    T = V.shape[0]
    M = a.shape[0]
    omega = np.zeros((T, M))
    omega[0, :] = np.log(pi * b[:, order[V[0]]].sum(axis = 1))
    prev = np.zeros((T - 1, M))
    for t in range(1, T):
        for j in range(M):
            probability = omega[t - 1] + np.log(a[:, j]) + np.log(b[j, order[V[t]]].sum())
            prev[t - 1, j] = np.argmax(probability)
            omega[t, j] = np.max(probability)
    S = np.zeros(T)
    last_state = np.argmax(omega[T - 1, :])
    S[0] = last_state
    backtrack_index = 1
    for i in range(T - 2, -1, -1):
        S[backtrack_index] = prev[i, int(last_state)]
        last_state = prev[i, int(last_state)]
        backtrack_index += 1
    S = np.flip(S)
    return S

def viterbi(a, b, pi, V, order):
    """
    Viterbi path
    
    Parameters
    ----------
    a : numpy array
        Transition probability matrix
    b : numpy array
        Emission probability matrix
    pi : numpy array
        Vector of starting probabilities of the hidden states
    V : numpy array
        Vector of observed states
    """
    T = V.shape[0]
    M = a.shape[0]
    omega = np.zeros((T, M))
    omega[0, :] = np.log(pi * b[:, order[V[0]]].sum(axis = 1))
    prev = np.zeros((T - 1, M))
    for t in range(1, T):
        probability_matrix = omega[t - 1][:, np.newaxis] + np.log(a) + np.log(b[:, order[V[t]]].sum(axis = 1))
        prev[t - 1, :] = np.argmax(probability_matrix, axis=0)
        omega[t, :] = np.max(probability_matrix, axis=0)
    return omega, prev

def backtrack_viterbi(omega, prev):
    T = omega.shape[0]
    S = np.zeros(T)
    last_state = np.argmax(omega[T - 1, :])
    S[0] = last_state
    backtrack_index = 1
    for i in range(T - 2, -1, -1):
        S[backtrack_index] = prev[i, int(last_state)]
        last_state = prev[i, int(last_state)]
        backtrack_index += 1
    S = np.flip(S)
    return S

def viterbi_wrapper(a, b, pi, V_lst):
    """
    Posterior probability wrapper.
    
    Parameters
    ----------
    a : numpy array
        Transition probability matrix
    b : numpy array
        Emission probability matrix
    pi : numpy array
        Vector of starting probabilities of the hidden states
    V : list of numpy arrays
        List of vectors of observed states, as integer indices
    """
    res_lst = []
    order = List()
    for i in range(624+1):
        order.append(get_idx_state(i))
    for i in range(len(V_lst)):
        (omega, prev) = viterbi(a, b, pi, V_lst[i], order)
        res_lst.append(backtrack_viterbi(omega, prev))
    return res_lst

def write_list(lst, res_name):
    """
    This function appends a list to a csv file.
    
    Parameters
    ----------
    lst : list
        List of values to append
    res_name : str
        File name to append to
    """
    with open(f'{res_name}', 'a') as f:
        for i in range(len(lst)):
            f.write(str(lst[i]))
            if i != (len(lst)-1):
                f.write(',')
        f.write('\n')

def trans_emiss_calc(t_A, t_B, t_C, t_2, t_upper, t_out, N_AB, N_ABC, r, n_int_AB, n_int_ABC,
                     cut_AB = 'standard', cut_ABC = 'standard', tmp_path = './'):
    """
    This function calculates the emission and transition probabilities
    given a certain set of parameters. 
    
    Parameters
    ----------
    t_A : numeric
        Time in generations from present to the first speciation event for species A
        (times mutation rate)
    t_B : numeric
        Time in generations from present to the first speciation event for species B
        (times mutation rate)
    t_C : numeric
        Time in generations from present to the second speciation event for species C
        (times mutation rate)
    t_2 : numeric
        Time in generations from the first speciation event to the second speciation event
        (times mutation rate)
    t_upper : numeric
        Time in generations between the end of the second-to-last interval and the third
        speciation event (times mutation rate)
    t_out : numeric
        Time in generations from present to the third speciation event for species D, plus
        the divergence between the ancestor of D and the ancestor of A, B and C at the time
        of the third speciation event (times mutation rate)
    N_AB : numeric
        Effective population size between speciation events (times mutation rate)
    N_ABC : numeric
        Effective population size in deep coalescence, before the second speciation event
        (times mutation rate)
    r : numeric
        Recombination rate per site per generation (divided by mutation rate)
    n_int_AB : integer
        Number of discretized time intervals between speciation events
    n_int_ABC : integer
        Number of discretized time intervals in deep coalescent
    """
    # Reference Ne (for normalization)
    N_ref = N_ABC
    # Speciation times (in coalescent units, i.e. number of generations / N_ref)
    t_A = t_A/N_ref
    t_B = t_B/N_ref
    t_AB = t_2/N_ref
    t_C = t_C/N_ref
    t_upper = t_upper/N_ref
    t_out = t_out/N_ref
    # Recombination rates (r = rec. rate per site per generation)
    rho_A = N_ref*r
    rho_B = N_ref*r
    rho_AB = N_ref*r
    rho_C = N_ref*r
    rho_ABC = N_ref*r
    # Coalescent rates
    coal_A = N_ref/N_AB
    coal_B = N_ref/N_AB
    coal_AB = N_ref/N_AB
    coal_C = N_ref/N_AB
    coal_ABC = N_ref/N_ABC
    # Mutation rates (mu = mut. rate per site per generation)
    mu_A = N_ref*(4/3)
    mu_B = N_ref*(4/3)
    mu_C = N_ref*(4/3)
    mu_D = N_ref*(4/3)
    mu_AB = N_ref*(4/3)
    mu_ABC = N_ref*(4/3)
    
    tr = get_joint_prob_mat(
        t_A,    t_B,    t_AB,    t_C, 
        rho_A,  rho_B,  rho_AB,  rho_C,  rho_ABC, 
        coal_A, coal_B, coal_AB, coal_C, coal_ABC,
        n_int_AB, n_int_ABC,
        cut_AB, cut_ABC, tmp_path
    )
    tr = pd.DataFrame(tr, columns=['From', 'To', 'Prob']).pivot(index = ['From'], columns = ['To'], values = ['Prob'])
    tr.columns = tr.columns.droplevel()
    hidden_names = list(tr.columns)
    hidden_names = dict(zip(range(len(hidden_names)), hidden_names))
    arr = np.array(tr).astype(np.float64)
    pi = arr.sum(axis=1)
    a = arr/pi[:,None]
    
    em = get_emission_prob_mat(
        t_A,    t_B,    t_AB,    t_C,    t_upper,   t_out,
        rho_A,  rho_B,  rho_AB,  rho_C,  rho_ABC, 
        coal_A, coal_B, coal_AB, coal_C, coal_ABC,
        n_int_AB, n_int_ABC,
        mu_A, mu_B, mu_C, mu_D, mu_AB, mu_ABC,
        cut_AB, cut_ABC
    )
    em.hidden_state = em.hidden_state.astype("category")
    em.hidden_state.cat.set_categories(hidden_names)
    em = em.sort_values(["hidden_state"])
    em = em.iloc[: , 1:]
    observed_names = list(em.columns)
    observed_names = dict(zip(range(len(observed_names)), observed_names))
    b = np.array(em)
    
    return a, b, pi, hidden_names, observed_names

def trans_emiss_calc_introgression(
        t_A, t_B, t_C, t_2, t_upper, t_out, t_m, N_AB, N_BC, N_ABC, r, m, n_int_AB, n_int_ABC,
        cut_AB = 'standard', cut_ABC = 'standard', tmp_path = './'):
    """
    This function calculates the emission and transition probabilities
    given a certain set of parameters. 
    
    Parameters
    ----------
    t_A : numeric
        Time in generations from present to the first speciation event for species A
        (times mutation rate)
    t_B : numeric
        Time in generations from present to the migration event for species B
        (times mutation rate)
    t_C : numeric
        Time in generations from present to the migration event for species C
        (times mutation rate)
    t_2 : numeric
        Time in generations from the first speciation event to the second speciation event
        (times mutation rate)
    t_upper : numeric
        Time in generations between the end of the second-to-last interval and the third
        speciation event (times mutation rate)
    t_out : numeric
        Time in generations from present to the third speciation event for species D, plus
        the divergence between the ancestor of D and the ancestor of A, B and C at the time
        of the third speciation event (times mutation rate)
    t_m : numeric
        Time in generagions from admixture time until first speciation time
    N_AB : numeric
        Effective population size between speciation events (times mutation rate) for AB
    N_BC : numeric
        Effective population size between speciation events (times mutation rate) for BC
    N_ABC : numeric
        Effective population size in deep coalescence, before the second speciation event
        (times mutation rate)
    r : numeric
        Recombination rate per site per generation (divided by mutation rate)
    m : numeric
        Migration rate (admixture proportion)
    n_int_AB : integer
        Number of discretized time intervals between speciation events
    n_int_ABC : integer
        Number of discretized time intervals in deep coalescent
    """

    # Reference Ne (for normalization)
    N_ref = N_ABC
    # Speciation times (in coalescent units, i.e. number of generations / N_ref)
    t_A = t_A/N_ref
    t_B = t_B/N_ref
    t_AB = t_2/N_ref
    t_C = t_C/N_ref
    t_upper = t_upper/N_ref
    t_out = t_out/N_ref
    t_m = t_m/N_ref
    # Recombination rates (r = rec. rate per site per generation)
    rho_A = N_ref*r
    rho_B = N_ref*r
    rho_AB = N_ref*r
    rho_C = N_ref*r
    rho_ABC = N_ref*r
    # Coalescent rates
    coal_A = N_ref/N_AB
    coal_B = N_ref/N_AB
    coal_AB = N_ref/N_AB
    coal_BC = N_ref/N_BC
    coal_C = N_ref/N_BC
    coal_ABC = N_ref/N_ABC
    # Mutation rates (mu = mut. rate per site per generation)
    mu_A = N_ref*(4/3)
    mu_B = N_ref*(4/3)
    mu_C = N_ref*(4/3)
    mu_D = N_ref*(4/3)
    mu_AB = N_ref*(4/3)
    mu_ABC = N_ref*(4/3)
        
    tr = get_joint_prob_mat_introgression(
        t_A,    t_B,    t_AB,    t_C,    t_m,
        rho_A,  rho_B,  rho_AB,  rho_C,  rho_ABC, 
        coal_A, coal_B, coal_AB, coal_BC, coal_C, coal_ABC,
        m,
        n_int_AB, n_int_ABC, 
        cut_AB, cut_ABC, tmp_path)
    tr = pd.DataFrame(tr, columns=['From', 'To', 'Prob']).pivot(index = ['From'], columns = ['To'], values = ['Prob'])
    tr.columns = tr.columns.droplevel()
    hidden_names = list(tr.columns)
    hidden_names = dict(zip(range(len(hidden_names)), hidden_names))
    arr = np.array(tr).astype(np.float64)
    pi = arr.sum(axis=1)
    a = arr/pi[:,None]
    
    em = get_emission_prob_mat_introgression(
        t_A,    t_B,    t_AB,    t_C,    t_upper,   t_out,   t_m, 
        rho_A,  rho_B,  rho_AB,  rho_C,  rho_ABC, 
        coal_A, coal_B, coal_AB, coal_BC, coal_C, coal_ABC,
        n_int_AB, n_int_ABC,
        mu_A, mu_B, mu_C, mu_D, mu_AB, mu_ABC
    )
    em.hidden_state = em.hidden_state.astype("category")
    em.hidden_state.cat.set_categories(hidden_names)
    em = em.sort_values(["hidden_state"])
    em = em.iloc[: , 1:]
    observed_names = list(em.columns)
    observed_names = dict(zip(range(len(observed_names)), observed_names))
    b = np.array(em)
    
    return a, b, pi, hidden_names, observed_names

def trans_emiss_calc_introgression_new_method(
        t_A, t_B, t_C, t_2, t_upper, t_out, t_m, N_AB, N_BC, N_ABC, r, m, n_int_AB, n_int_ABC,
        cut_AB = 'standard', cut_ABC = 'standard', tmp_path = './'):
    """
    This function calculates the emission and transition probabilities
    given a certain set of parameters. 
    
    Parameters
    ----------
    t_A : numeric
        Time in generations from present to the first speciation event for species A
        (times mutation rate)
    t_B : numeric
        Time in generations from present to the migration event for species B
        (times mutation rate)
    t_C : numeric
        Time in generations from present to the migration event for species C
        (times mutation rate)
    t_2 : numeric
        Time in generations from the first speciation event to the second speciation event
        (times mutation rate)
    t_upper : numeric
        Time in generations between the end of the second-to-last interval and the third
        speciation event (times mutation rate)
    t_out : numeric
        Time in generations from present to the third speciation event for species D, plus
        the divergence between the ancestor of D and the ancestor of A, B and C at the time
        of the third speciation event (times mutation rate)
    t_m : numeric
        Time in generagions from admixture time until first speciation time
    N_AB : numeric
        Effective population size between speciation events (times mutation rate) for AB
    N_BC : numeric
        Effective population size between speciation events (times mutation rate) for BC
    N_ABC : numeric
        Effective population size in deep coalescence, before the second speciation event
        (times mutation rate)
    r : numeric
        Recombination rate per site per generation (divided by mutation rate)
    m : numeric
        Migration rate (admixture proportion)
    n_int_AB : integer
        Number of discretized time intervals between speciation events
    n_int_ABC : integer
        Number of discretized time intervals in deep coalescent
    """

    # Reference Ne (for normalization)
    N_ref = N_ABC
    # Speciation times (in coalescent units, i.e. number of generations / N_ref)
    t_A = t_A/N_ref
    t_B = t_B/N_ref
    t_AB = t_2/N_ref
    t_C = t_C/N_ref
    t_upper = t_upper/N_ref
    t_out = t_out/N_ref
    t_m = t_m/N_ref
    # Recombination rates (r = rec. rate per site per generation)
    rho_A = N_ref*r
    rho_B = N_ref*r
    rho_AB = N_ref*r
    rho_C = N_ref*r
    rho_ABC = N_ref*r
    # Coalescent rates
    coal_A = N_ref/N_AB
    coal_B = N_ref/N_AB
    coal_AB = N_ref/N_AB
    coal_BC = N_ref/N_BC
    coal_C = N_ref/N_BC
    coal_ABC = N_ref/N_ABC
    # Mutation rates (mu = mut. rate per site per generation)
    mu_A = N_ref*(4/3)
    mu_B = N_ref*(4/3)
    mu_C = N_ref*(4/3)
    mu_D = N_ref*(4/3)
    mu_AB = N_ref*(4/3)
    mu_ABC = N_ref*(4/3)
        
    tr = get_joint_prob_mat_introgression(
        t_A,    t_B,    t_AB,    t_C,    t_m,
        rho_A,  rho_B,  rho_AB,  rho_C,  rho_ABC, 
        coal_A, coal_B, coal_AB, coal_BC, coal_C, coal_ABC,
        m,
        n_int_AB, n_int_ABC, 
        cut_AB, cut_ABC, tmp_path)
    tr = pd.DataFrame(tr, columns=['From', 'To', 'Prob']).pivot(index = ['From'], columns = ['To'], values = ['Prob'])
    tr.columns = tr.columns.droplevel()
    hidden_names = list(tr.columns)
    hidden_names = dict(zip(range(len(hidden_names)), hidden_names))
    arr = np.array(tr).astype(np.float64)
    pi = arr.sum(axis=1)
    a = arr/pi[:,None]
    
    em = get_emission_prob_mat_introgression(
        t_A,    t_B,    t_AB,    t_C,    t_upper,   t_out,   t_m, 
        rho_A,  rho_B,  rho_AB,  rho_C,  rho_ABC, 
        coal_A, coal_B, coal_AB, coal_BC, coal_C, coal_ABC,
        n_int_AB, n_int_ABC,
        mu_A, mu_B, mu_C, mu_D, mu_AB, mu_ABC
    )
    em.hidden_state = em.hidden_state.astype("category")
    em.hidden_state.cat.set_categories(hidden_names)
    em = em.sort_values(["hidden_state"])
    em = em.iloc[: , 1:]
    observed_names = list(em.columns)
    observed_names = dict(zip(range(len(observed_names)), observed_names))
    b = np.array(em)

    new_dct = {}
    for i in observed_names:
        new_obs = observed_names[i][0:2]+observed_names[i][3]
        if new_obs not in new_dct:
            new_dct[new_obs] = [i]
        else:
            new_dct[new_obs].append(i)

    new_emissions = np.zeros((len(hidden_names), len(new_dct)))
    new_observed_states = {}
    for i, key in enumerate(new_dct):
        new_emissions[:,i] = b[:,new_dct[key]].sum(axis=1)
        new_observed_states[get_obs_state_dct_new_method().index(key)] = key
    
    return a, new_emissions, pi, hidden_names, new_observed_states

def optimization_wrapper(arg_lst, d, V_lst, res_name, info):
    # Define time model (optimized parameters)
    if len(arg_lst) == 6:
        t_1, t_2, t_upper, N_AB, N_ABC, r = arg_lst
        t_A = t_B = t_1
        t_C = t_1 + t_2
        cut_ABC = cutpoints_ABC(d['n_int_ABC'], 1)
        t_out = t_1 + t_2 + cut_ABC[d['n_int_ABC']-1]*N_ABC + t_upper + 2*N_ABC
    elif len(arg_lst) == 9:
        t_A, t_B, t_C, t_2, t_upper, t_out, N_AB, N_ABC, r = arg_lst
    elif len(arg_lst) == 8:
        t_A, t_B, t_C, t_2, t_upper, N_AB, N_ABC, r = arg_lst
        cut_ABC = cutpoints_ABC(d['n_int_ABC'], 1)
        t_out = (((t_A+t_B)/2+t_2)+t_C)/2 + cut_ABC[d['n_int_ABC']-1]*N_ABC + t_upper + 2*N_ABC
    # Calculate transition and emission probabilities
    a, b, pi, hidden_names, observed_names = trans_emiss_calc(
        t_A, t_B, t_C, t_2, t_upper, t_out, N_AB, N_ABC, r, 
        d['n_int_AB'], d['n_int_ABC'], 'standard', 'standard', info['tmp_path']
    )
    # Save indices for hidden and observed states
    if info['Nfeval'] == 0:
        pd.DataFrame({'idx': list(hidden_names.keys()), 'hidden': list(hidden_names.values())}).to_csv('hidden_states.csv', index = False)
        pd.DataFrame({'idx': list(observed_names.keys()), 'observed': list(observed_names.values())}).to_csv('observed_states.csv', index = False)
    try:
        ncpus = int(os.environ["SLURM_JOB_CPUS_PER_NODE"])
    except KeyError:
        ncpus = mp.cpu_count()
    # Calculate log-likelihood
    if len(V_lst) >= ncpus:
        loglik = loglik_wrapper_par(a, b, pi, V_lst)
    else:
        loglik = loglik_wrapper(a, b, pi, V_lst)
    # Write parameter estimates, likelihood and time
    write_list([info['Nfeval']] + arg_lst.tolist() + [loglik, time.time() - info['time']], res_name)
    # Update optimization cycle
    info['Nfeval'] += 1
    return -loglik


def optimizer(optim_params, fixed_params, V_lst, res_name, method = 'Nelder-Mead', header = True, tmp_path = './'):
    """
    Optimization function. 
    
    Parameters
    ----------
    optim_params : dictionary
        Dictionary containing the initial values for the 
        parameters to be optimized, and their optimization
        bounds. The structure of the dictionary should be
        as follows: 
            dct['variable'] = [initial_value, lower_bound, upper_bound]
        The dictionary must contain either 6 (t_1, t_2, t_upper, N_AB, N_ABC, r)
        or 9 (t_A, t_B, t_C, t_2, t_upper, t_out, N_AB, N_ABC, r) entries,
        in that specific order. 
    fixed params : dictionary
        Dictionary containing the values for the fixed parameters.
        The dictionary must contain entries n_int_AB and n_int_ABC (in no particular order).
    V_lst : list of numpy arrays
        List of arrays of integers corresponding to the the observed states.
    res_name : str
        Location and name of the gile where the results should be 
        saved (in csv format).
    """
    init_params = np.array([i[0] for i in optim_params.values()])
    bnds = [(i[1], i[2]) for i in optim_params.values()]
    if header:
        write_list(['n_eval'] + list(optim_params.keys()) + ['loglik', 'time'], res_name)
    options = {
        'maxiter': 10000,
        'disp': True
    }
    # if method in ['L-BFGS-B', 'TNC']:
    #     if len(optim_params) == 6:
    #         options['eps'] = np.array([10, 1, 10, 1, 1, 1e-9])
    #     elif len(optim_params) == 9:
    #         options['eps'] = np.array([10, 10, 10, 1, 10, 10, 1, 1, 1e-9])
    res = minimize(
        optimization_wrapper, 
        x0 = init_params,
        args = (fixed_params, V_lst, res_name, {'Nfeval': 0, 'time': time.time(), 'tmp_path': tmp_path}),
        method = method,
        bounds = bnds, 
        options = options
    )

def optimization_wrapper_introgression(arg_lst, d, V_lst, res_name, info):
    # Define time model (optimized parameters)
    if "proportional" in d:
        t_1, t_2, t_upper, t_m_prop, N_AB, N_BC, N_ABC, r, m = arg_lst
        t_A = t_1
        t_m = t_1*t_m_prop
        t_B = t_C = t_1-t_m
        cut_ABC = cutpoints_ABC(d['n_int_ABC'], 1)
        t_out = t_1 + t_2 + cut_ABC[d['n_int_ABC']-1]*N_ABC + t_upper + 2*N_ABC
    elif len(arg_lst) == 9:
        t_1, t_2, t_upper, t_m, N_AB, N_BC, N_ABC, r, m = arg_lst
        t_A = t_1
        t_B = t_C = t_1-t_m
        cut_ABC = cutpoints_ABC(d['n_int_ABC'], 1)
        t_out = t_1 + t_2 + cut_ABC[d['n_int_ABC']-1]*N_ABC + t_upper + 2*N_ABC
    elif len(arg_lst) == 10:
        t_1, t_C, t_2, t_upper, t_m, N_AB, N_BC, N_ABC, r, m = arg_lst
        t_A = t_1
        t_B = t_1-t_m
        cut_ABC = cutpoints_ABC(d['n_int_ABC'], 1)
        t_out = t_1 + t_2 + cut_ABC[d['n_int_ABC']-1]*N_ABC + t_upper + 2*N_ABC
    elif len(arg_lst) == 12:
        t_A, t_B, t_C, t_2, t_upper, t_out, t_m, N_AB, N_BC, N_ABC, r, m = arg_lst
    elif len(arg_lst) == 11:
        t_A, t_B, t_C, t_2, t_upper, t_m, N_AB, N_BC, N_ABC, r, m = arg_lst
        cut_ABC = cutpoints_ABC(d['n_int_ABC'], 1)
        t_out = (((t_A+(t_B+t_m))/2+t_2)+(t_C+t_m+t_2))/2 + cut_ABC[d['n_int_ABC']-1]*N_ABC + t_upper + 2*N_ABC
    # Calculate transition and emission probabilities
    a, b, pi, hidden_names, observed_names = trans_emiss_calc_introgression(
        t_A, t_B, t_C, t_2, t_upper, t_out, t_m, N_AB, N_BC, N_ABC, r, m,
        d['n_int_AB'], d['n_int_ABC'], 'standard', 'standard', info['tmp_path']
    )
    # Save indices for hidden and observed states
    if info['Nfeval'] == 0:
        pd.DataFrame({'idx': list(hidden_names.keys()), 'hidden': list(hidden_names.values())}).to_csv('hidden_states.csv', index = False)
        pd.DataFrame({'idx': list(observed_names.keys()), 'observed': list(observed_names.values())}).to_csv('observed_states.csv', index = False)
    # Calculate log-likelihood
    try:
        ncpus = int(os.environ["SLURM_JOB_CPUS_PER_NODE"])
    except KeyError:
        ncpus = mp.cpu_count()
    # Calculate log-likelihood
    if len(V_lst) >= ncpus:
        loglik = loglik_wrapper_par(a, b, pi, V_lst)
    else:
        loglik = loglik_wrapper(a, b, pi, V_lst)
    # Write parameter estimates, likelihood and time
    write_list([info['Nfeval']] + arg_lst.tolist() + [loglik, time.time() - info['time']], res_name)
    # Update optimization cycle
    info['Nfeval'] += 1
    return -loglik

def optimization_wrapper_introgression_new_method(arg_lst, d, V_lst, res_name, info):
    # Define time model (optimized parameters)
    if len(arg_lst) == 9:
        t_1, t_2, t_upper, t_m, N_AB, N_BC, N_ABC, r, m = arg_lst
        t_A = t_1
        t_B = t_C = t_1-t_m
        cut_ABC = cutpoints_ABC(d['n_int_ABC'], 1)
        t_out = t_1 + t_2 + cut_ABC[d['n_int_ABC']-1]*N_ABC + t_upper + 2*N_ABC
    if len(arg_lst) == 10:
        t_1, t_C, t_2, t_upper, t_m, N_AB, N_BC, N_ABC, r, m = arg_lst
        t_A = t_1
        t_B = t_1-t_m
        cut_ABC = cutpoints_ABC(d['n_int_ABC'], 1)
        t_out = t_1 + t_2 + cut_ABC[d['n_int_ABC']-1]*N_ABC + t_upper + 2*N_ABC
    elif len(arg_lst) == 12:
        t_A, t_B, t_C, t_2, t_upper, t_out, t_m, N_AB, N_BC, N_ABC, r, m = arg_lst
    elif len(arg_lst) == 11:
        t_A, t_B, t_C, t_2, t_upper, t_m, N_AB, N_BC, N_ABC, r, m = arg_lst
        cut_ABC = cutpoints_ABC(d['n_int_ABC'], 1)
        t_out = (((t_A+(t_B+t_m))/2+t_2)+(t_C+t_m+t_2))/2 + cut_ABC[d['n_int_ABC']-1]*N_ABC + t_upper + 2*N_ABC
    # Calculate transition and emission probabilities
    a, b, pi, hidden_names, observed_names = trans_emiss_calc_introgression_new_method(
        t_A, t_B, t_C, t_2, t_upper, t_out, t_m, N_AB, N_BC, N_ABC, r, m,
        d['n_int_AB'], d['n_int_ABC'], 'standard', 'standard', info['tmp_path']
    )

    # Save indices for hidden and observed states
    if info['Nfeval'] == 0:
        pd.DataFrame({'idx': list(hidden_names.keys()), 'hidden': list(hidden_names.values())}).to_csv('hidden_states.csv', index = False)
        pd.DataFrame({'idx': list(observed_names.keys()), 'observed': list(observed_names.values())}).to_csv('observed_states.csv', index = False)
    # Calculate log-likelihood
    try:
        ncpus = int(os.environ["SLURM_JOB_CPUS_PER_NODE"])
    except KeyError:
        ncpus = mp.cpu_count()
    # Calculate log-likelihood
    if len(V_lst) >= ncpus:
        loglik = loglik_wrapper_par_new_method(a, b, pi, V_lst)
    else:
        loglik = loglik_wrapper_new_method(a, b, pi, V_lst)
    # Write parameter estimates, likelihood and time
    write_list([info['Nfeval']] + arg_lst.tolist() + [loglik, time.time() - info['time']], res_name)
    # Update optimization cycle
    info['Nfeval'] += 1
    return -loglik
    
def optimizer_introgression(optim_params, fixed_params, V_lst, res_name, method = 'Nelder-Mead', header = True, tmp_path = './'):
    """
    Optimization function. 
    
    Parameters
    ----------
    optim_params : dictionary
        Dictionary containing the initial values for the 
        parameters to be optimized, and their optimization
        bounds. The structure of the dictionary should be
        as follows: 
            dct['variable'] = [initial_value, lower_bound, upper_bound]
        The dictionary must contain either 8 (t_1, t_2, t_upper, t_m, N_AB, N_ABC, r, m),
        10 (t_A, t_B, t_C, t_2, t_upper, t_m, N_AB, N_ABC, r, m),
        or 11 (t_A, t_B, t_C, t_2, t_upper, t_out, t_m, N_AB, N_ABC, r, m) entries,
        in that specific order. 
    fixed params : dictionary
        Dictionary containing the values for the fixed parameters.
        The dictionary must contain entries n_int_AB and n_int_ABC (in no particular order).
    V_lst : list of numpy arrays
        List of arrays of integers corresponding to the the observed states.
    res_name : str
        Location and name of the gile where the results should be 
        saved (in csv format).
    """
    init_params = np.array([i[0] for i in optim_params.values()])
    bnds = [(i[1], i[2]) for i in optim_params.values()]
    if header:
        write_list(['n_eval'] + list(optim_params.keys()) + ['loglik', 'time'], res_name)
    options = {
        'maxiter': 10000,
        'disp': True
    }
        
    res = minimize(
        optimization_wrapper_introgression, 
        x0 = init_params,
        args = (fixed_params, V_lst, res_name, {'Nfeval': 0, 'time': time.time(), 'tmp_path': tmp_path}),
        method = method,
        bounds = bnds, 
        options = options
    )

def optimizer_introgression_new_method(optim_params, fixed_params, V_lst, res_name, method = 'Nelder-Mead', header = True, tmp_path = './'):
    """
    Optimization function. 
    
    Parameters
    ----------
    optim_params : dictionary
        Dictionary containing the initial values for the 
        parameters to be optimized, and their optimization
        bounds. The structure of the dictionary should be
        as follows: 
            dct['variable'] = [initial_value, lower_bound, upper_bound]
        The dictionary must contain either 8 (t_1, t_2, t_upper, t_m, N_AB, N_ABC, r, m),
        10 (t_A, t_B, t_C, t_2, t_upper, t_m, N_AB, N_ABC, r, m),
        or 11 (t_A, t_B, t_C, t_2, t_upper, t_out, t_m, N_AB, N_ABC, r, m) entries,
        in that specific order. 
    fixed params : dictionary
        Dictionary containing the values for the fixed parameters.
        The dictionary must contain entries n_int_AB and n_int_ABC (in no particular order).
    V_lst : list of numpy arrays
        List of arrays of integers corresponding to the the observed states.
    res_name : str
        Location and name of the gile where the results should be 
        saved (in csv format).
    """
    init_params = np.array([i[0] for i in optim_params.values()])
    bnds = [(i[1], i[2]) for i in optim_params.values()]
    if header:
        write_list(['n_eval'] + list(optim_params.keys()) + ['loglik', 'time'], res_name)
    options = {
        'maxiter': 10000,
        'disp': True
    }
        
    res = minimize(
        optimization_wrapper_introgression_new_method, 
        x0 = init_params,
        args = (fixed_params, V_lst, res_name, {'Nfeval': 0, 'time': time.time(), 'tmp_path': tmp_path}),
        method = method,
        bounds = bnds, 
        options = options
    )
    
