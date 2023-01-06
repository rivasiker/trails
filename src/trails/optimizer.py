import numpy as np
import pandas as pd
from trails.get_emission_prob_mat import get_emission_prob_mat
from trails.get_joint_prob_mat import get_joint_prob_mat
from scipy.optimize import minimize
from trails.cutpoints import cutpoints_ABC
from numba import njit
import time


@njit
def forward_loglik(a, b, pi, V):
    alpha = forward(a, b, pi, V)
    x = alpha[-1, :].max()
    return np.log(np.exp(alpha[len(V)-1]-x).sum())+x

@njit
def forward(a, b, pi, V):
    alpha = np.zeros((V.shape[0], a.shape[0]))
    alpha[0, :] = np.log(pi * b[:, V[0]])
    for t in range(1, V.shape[0]):
        x = alpha[t-1, :].max()
        alpha[t, :] = np.log((np.exp(alpha[t - 1]-x) @ a) * b[:, V[t]])+x
    return alpha

@njit
def backward(a, b, V):
    beta = np.zeros((V.shape[0], a.shape[0]))
    beta[V.shape[0] - 1] = np.zeros((a.shape[0]))
    for t in range(V.shape[0] - 2, -1, -1):
        x = beta[t+1, :].max()
        beta[t, :] = np.log((np.exp(beta[t + 1]-x) * b[:, V[t + 1]]) @ a)+x
    return beta


def post_prob(a, b, pi, V):
    alpha = forward(a, b, pi, V)
    beta = backward(a, b, V)
    post_prob = (alpha+beta)
    max_row = post_prob.max(1).reshape(-1, 1)
    post_prob = np.exp(post_prob-max_row)/np.exp(post_prob-max_row).sum(1).reshape(-1, 1)
    return post_prob

@njit
def viterbi(a, b, pi, V):
    T = V.shape[0]
    M = a.shape[0]
    omega = np.zeros((T, M))
    omega[0, :] = np.log(pi * b[:, V[0]])
    prev = np.zeros((T - 1, M))
    for t in range(1, T):
        for j in range(M):
            probability = omega[t - 1] + np.log(a[:, j]) + np.log(b[j, V[t]])
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

def write_list(lst, res_name):
    with open('{}.csv'.format(res_name), 'a') as f:
        for i in range(len(lst)):
            f.write(str(lst[i]))
            if i != (len(lst)-1):
                f.write(',')
        f.write('\n')

def trans_emiss_calc(t_A, t_B, t_C, t_2, t_upper, t_out, N_AB, N_ABC, r, mu_A, mu_B, mu_C, mu_D, mu_AB, mu_ABC, n_int_AB, n_int_ABC):
    # Reference Ne (for normalization)
    N_ref = N_ABC
    # Speciation times (in coalescent units, i.e. number of generations / N_ref)
    t_A = t_A/N_ref
    t_B = t_B/N_ref
    t_AB = t_2/N_ref
    t_C = (t_C+t_2)/N_ref
    t_upper = t_upper/N_ref
    t_out = t_out/N_ref
    # Recombination rates (r = rec. rate per site per generation)
    rho_A = 2*N_ref*r
    rho_B = 2*N_ref*r
    rho_AB = 2*N_ref*r
    rho_C = 2*N_ref*r
    rho_ABC = 2*N_ref*r
    # Coalescent rates
    coal_A = N_ref/N_AB
    coal_B = N_ref/N_AB
    coal_AB = N_ref/N_AB
    coal_C = N_ref/N_AB
    coal_ABC = N_ref/N_ABC
    # Mutation rates (mu = mut. rate per site per generation)
    mu_A = 2*N_ref*mu_A*(4/3)
    mu_B = 2*N_ref*mu_B*(4/3)
    mu_C = 2*N_ref*mu_C*(4/3)
    mu_D = 2*N_ref*mu_D*(4/3)
    mu_AB = 2*N_ref*mu_AB*(4/3)
    mu_ABC = 2*N_ref*mu_ABC*(4/3)
    
    tr = get_joint_prob_mat(
        t_A,    t_B,    t_AB,    t_C, 
        rho_A,  rho_B,  rho_AB,  rho_C,  rho_ABC, 
        coal_A, coal_B, coal_AB, coal_C, coal_ABC,
        n_int_AB, n_int_ABC)
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
        mu_A, mu_B, mu_C, mu_D, mu_AB, mu_ABC)
    em.hidden_state = em.hidden_state.astype("category")
    em.hidden_state.cat.set_categories(hidden_names)
    em = em.sort_values(["hidden_state"])
    em = em.iloc[: , 1:]
    observed_names = list(em.columns)
    observed_names = dict(zip(range(len(observed_names)), observed_names))
    b = np.array(em)
    
    return a, b, pi, hidden_names, observed_names

def optimization_wrapper_no_mu(arg_lst, n_int_AB, n_int_ABC, V, res_name, verbose, info, mu):
    t_1, t_2, t_upper, N_AB, N_ABC, r = arg_lst
    cut_ABC = cutpoints_ABC(n_int_ABC, 1)
    t_out = t_1+t_2+cut_ABC[n_int_ABC-1]+t_upper+2
    a, b, pi, hidden_names, observed_names = trans_emiss_calc(
        t_1, t_1, t_1, t_2, t_upper, t_out, N_AB, N_ABC, r, mu, mu, mu, mu, mu, mu, n_int_AB, n_int_ABC
    )
    loglik = forward_loglik(a, b, pi, V)
    write_list([info['Nfeval'], t_1, t_2, t_upper, N_AB, N_ABC, r, loglik, time.time()-info['time']], res_name)
    if verbose:
        print(
            '{0:4d}   {1: .5e}   {2: .5e}   {3: .5e}   {4: .5e}   {5: .5e}   {6: .5e}   {7: 3.6f}   {8: 3.6f}'.format(
                info['Nfeval'], 
                arg_lst[0], arg_lst[1], arg_lst[2], arg_lst[3], 
                arg_lst[4], arg_lst[5], loglik, time.time()-info['time']
            )
        )
    info['Nfeval'] += 1
    return -loglik

def optimizer_no_mu(t_1, t_2, t_upper, N_AB, N_ABC, r, mu, n_int_AB, n_int_ABC, V, res_name, verbose = False):
    init_params = np.array([t_1, t_2, t_upper, N_AB, N_ABC, r])
    
    b_t = (1e4, 2e6)
    b_N = (1000, 100000)
    b_r = (1e-10, 1e-7)
    # b_mu = (1e-9, 1e-7)
    bnds = (b_t, b_t, b_t, b_N, b_N, b_r)
    res = minimize(
        optimization_wrapper_no_mu, 
        x0 = init_params,
        args = (n_int_AB, n_int_ABC, V, res_name, verbose, {'Nfeval':0, 'time':time.time()}, mu),
        method = 'Nelder-Mead',
        bounds = bnds, 
        options = {
            'maxiter': 3000,
            'disp': True
        }
    )

def optimization_wrapper_no_mu_t(arg_lst, n_int_AB, n_int_ABC, V, res_name, verbose, info, mu):
    t_A, t_B, t_C, t_2, t_upper, t_out, N_AB, N_ABC, r = arg_lst
    a, b, pi, hidden_names, observed_names = trans_emiss_calc(
        t_A, t_B, t_C, t_2, t_upper, t_out, N_AB, N_ABC, r, mu, mu, mu, mu, mu, mu, n_int_AB, n_int_ABC
    )
    loglik = forward_loglik(a, b, pi, V)
    write_list([info['Nfeval'], t_A, t_B, t_C, t_2, t_upper, N_AB, N_ABC, r, loglik, time.time()-info['time']], res_name)
    if verbose:
        print(
            '{0:4d}   {1: .5e}   {2: .5e}   {3: .5e}   {4: .5e}   {5: .5e}   {6: .5e}   {7: .5e}   {8: .5e}   {9: .5e}   {10: 3.6f}   {11: 3.6f}'.format(
                info['Nfeval'], 
                arg_lst[0], arg_lst[1], arg_lst[2], arg_lst[3], 
                arg_lst[4], arg_lst[5], arg_lst[6], arg_lst[7],
                arg_lst[8],
                loglik, time.time()-info['time']
            )
        )
    info['Nfeval'] += 1
    return -loglik

def optimizer_no_mu_t(t_A, t_B, t_C, t_2, t_upper, t_out, N_AB, N_ABC, r, mu, n_int_AB, n_int_ABC, V, res_name, verbose = False):
    init_params = np.array([t_A, t_B, t_C, t_2, t_upper, t_out, N_AB, N_ABC, r])
    b_t = (1e4, 2e6)
    b_out = (1e5, 1e7)
    b_N = (1000, 100000)
    b_r = (1e-10, 1e-7)
    # b_mu = (1e-9, 1e-7)
    bnds = (b_t, b_t, b_t, b_t, b_t, b_t, b_out, b_N, b_N, b_r)
    res = minimize(
        optimization_wrapper_no_mu_t, 
        x0 = init_params,
        args = (n_int_AB, n_int_ABC, V, res_name, verbose, {'Nfeval':0, 'time':time.time()}, mu),
        method = 'Nelder-Mead',
        bounds = bnds, 
        options = {
            'maxiter': 3000,
            'disp': True
        }
    )
    