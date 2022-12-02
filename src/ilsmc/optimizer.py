import numpy as np
import pandas as pd
from ilsmc.get_emission_prob_mat import get_emission_prob_mat
from ilsmc.get_joint_prob_mat import get_joint_prob_mat
from scipy.optimize import minimize, basinhopping, fmin_tnc
from csv import writer
from dlib import find_max_global
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

def trans_emiss_calc(t_1, t_2, t_upper, N_AB, N_ABC, r, mu, n_int_AB, n_int_ABC):
    # Reference Ne (for normalization)
    N_ref = N_AB
    # Speciation times (in coalescent units, i.e. number of generations / N_ref)
    t_A = t_1/N_ref
    t_B = t_1/N_ref
    t_AB = t_2/N_ref
    t_C = (t_1+t_2)/N_ref
    t_upper = t_upper/N_ref
    t_peak = 2*(N_ABC/N_ref)
    # Recombination rates (r = rec. rate per site per generation)
    rho_A = 2*N_ref*r
    rho_B = 2*N_ref*r
    rho_AB = 2*N_AB*r
    rho_C = 2*N_ref*r
    rho_ABC = 2*N_ABC*r
    # Coalescent rates
    coal_A = N_ref/N_ref
    coal_B = N_ref/N_ref
    coal_AB = N_AB/N_ref
    coal_C = N_ref/N_ref
    coal_ABC = N_ref/N_ABC
    # Mutation rates (mu = mut. rate per site per generation)
    mu_ = mu*(4/3) # For Jukes-Cantor model
    mu_A = 2*N_ref*mu_
    mu_B = 2*N_ref*mu_
    mu_C = 2*N_ref*mu_
    mu_D = 2*N_ref*mu_
    mu_AB = 2*N_ref*mu_
    mu_ABC = 2*N_ref*mu_
    
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
        t_A,    t_B,    t_AB,    t_C,    t_upper,   t_peak,
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

def optimization_wrapper(arg_lst, n_int_AB, n_int_ABC, V, res_name, verbose, info):
    t_1, t_2, t_upper, N_AB, N_ABC, r, mu = arg_lst
    a, b, pi, hidden_names, observed_names = trans_emiss_calc(
        t_1, t_2, t_upper, N_AB, N_ABC, r, mu, n_int_AB, n_int_ABC
    )
    loglik = forward_loglik(a, b, pi, V)
    write_list([info['Nfeval'], t_1, t_2, t_upper, N_AB, N_ABC, r, mu, loglik], res_name)
    if verbose:
        print(
            '{0:4d}   {1: .5e}   {2: .5e}   {3: .5e}   {4: .5e}   {5: .5e}   {6: .5e}   {7: .5e}   {8: 3.6f}'.format(
                info['Nfeval'], 
                arg_lst[0], arg_lst[1], arg_lst[2], arg_lst[3], 
                arg_lst[4], arg_lst[5], arg_lst[6], loglik
            )
        )
    info['Nfeval'] += 1
    return -loglik






def optimization_wrapper_no_mu(arg_lst, n_int_AB, n_int_ABC, V, res_name, verbose, info, mu):
    t_1, t_2, t_upper, N_AB, N_ABC, r = arg_lst
    a, b, pi, hidden_names, observed_names = trans_emiss_calc(
        t_1, t_2, t_upper, N_AB, N_ABC, r, mu, n_int_AB, n_int_ABC
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
    
    
    
    
    
    
    

    
def optimization_wrapper_no_mu_rho(arg_lst, n_int_AB, n_int_ABC, V, res_name, verbose, info, mu, r):
    t_1, t_2, t_upper, N_AB, N_ABC = arg_lst
    a, b, pi, hidden_names, observed_names = trans_emiss_calc(
        t_1, t_2, t_upper, N_AB, N_ABC, r, mu, n_int_AB, n_int_ABC
    )
    loglik = forward_loglik(a, b, pi, V)
    write_list([info['Nfeval'], t_1, t_2, t_upper, N_AB, N_ABC, loglik, time.time()-info['time']], res_name)
    if verbose:
        print(
            '{0:4d}   {1: .5e}   {2: .5e}   {3: .5e}   {4: .5e}   {5: .5e}   {6: 3.6f}   {7: 3.6f}'.format(
                info['Nfeval'], 
                arg_lst[0], arg_lst[1], arg_lst[2], arg_lst[3], 
                arg_lst[4], loglik, time.time()-info['time']
            )
        )
    info['Nfeval'] += 1
    return -loglik

def optimizer_no_mu_rho(t_1, t_2, t_upper, N_AB, N_ABC, r, mu, n_int_AB, n_int_ABC, V, res_name, verbose = False):
    init_params = np.array([t_1, t_2, t_upper, N_AB, N_ABC])
    
    b_t = (1e4, 2e6)
    b_N = (1000, 100000)
    # b_r = (1e-10, 1e-7)
    # b_mu = (1e-9, 1e-7)
    bnds = (b_t, b_t, b_t, b_N, b_N)
    res = minimize(
        optimization_wrapper_no_mu_rho, 
        x0 = init_params,
        args = (n_int_AB, n_int_ABC, V, res_name, verbose, {'Nfeval':0, 'time':time.time()}, mu, r),
        method = 'Nelder-Mead',
        bounds = bnds, 
        options = {
            'maxiter': 3000,
            'disp': True
        }
    )
    



def optimizer(t_1, t_2, t_upper, N_AB, N_ABC, r, mu, n_int_AB, n_int_ABC, V, res_name, verbose = False):
    init_params = np.array([t_1, t_2, t_upper, N_AB, N_ABC, r, mu])
    
    b_t = (1e4, 2e6)
    b_N = (1000, 100000)
    b_r = (1e-9, 1e-7)
    b_mu = (1e-9, 1e-7)
    bnds = (b_t, b_t, b_t, b_N, b_N, b_r, b_mu)
    res = minimize(
        optimization_wrapper, 
        x0 = init_params,
        args = (n_int_AB, n_int_ABC, V, res_name, verbose, {'Nfeval':0}),
        method = 'Nelder-Mead',
        bounds = bnds, 
        options = {
            'maxiter': 3000,
            'disp': True
        }
    )
    
    # res = basinhopping(
    #     optimization_wrapper,
    #     x0 = init_params,
    #     minimizer_kwargs = {
    #         'args' : (n_int_AB, n_int_ABC, V, res_name, verbose, {'Nfeval':0}),
    #         'method' : "Nelder-Mead",
    #         'bounds' : bnds,
    #         'options' : {
    #             'maxiter': 100,
    #             'disp': True
    #         }
    #     }
    # )
    
    # res = fmin_tnc(
    #     optimization_wrapper, 
    #     x0 = init_params,
    #     args = (n_int_AB, n_int_ABC, V, res_name, verbose, {'Nfeval':0}),
    #     bounds = bnds, 
    #     approx_grad = True,
    #     maxfun = 3000,
    #     disp = True
    # )
    
    return res





def optimization_wrapper_lipo(t_1, t_2, t_upper, N_AB, N_ABC, r, mu):
    a, b, pi, hidden_names, observed_names = trans_emiss_calc(
        t_1, t_2, t_upper, N_AB, N_ABC, r, mu, n_int_AB_l, n_int_ABC_l
    )
    loglik = forward_loglik(a, b, pi, V_l)
    write_list([info['Nfeval'], t_1, t_2, t_upper, N_AB, N_ABC, r, mu, loglik], res_name_l)
    if verbose_l:
        print(
            '{0:4d}   {1: .5e}   {2: .5e}   {3: .5e}   {4: .5e}   {5: .5e}   {6: .5e}   {7: .5e}   {8: 3.6f}'.format(
                info['Nfeval'], t_1, t_2, t_upper, N_AB, N_ABC, r, mu, loglik
            )
        )
    info['Nfeval'] += 1
    return loglik

def optimizer_lipo(n_int_AB, n_int_ABC, V, res_name, verbose = False):
    
    global n_int_AB_l
    global n_int_ABC_l
    global V_l
    global res_name_l
    global verbose_l
    global info
    
    info = {'Nfeval':0}
    n_int_AB_l = n_int_AB
    n_int_ABC_l = n_int_ABC
    V_l = V
    res_name_l = res_name
    verbose_l = verbose
    
    b_t_1 = (1e5, 3e5)
    b_t_2 = (3e4, 4e4)
    b_t_upper = (4e5, 6e5)
    b_N_AB = (20000, 40000)
    b_N_ABC = (30000, 50000)
    b_r = (5e-9, 2e-8)
    b_mu = (1e-8, 3e-8)
    lower_bnds = [b_t_1[0], b_t_2[0], b_t_upper[0], b_N_AB[0], b_N_ABC[0], b_r[0], b_mu[0]]
    upper_bnds = [b_t_1[1], b_t_2[1], b_t_upper[1], b_N_AB[1], b_N_ABC[1], b_r[1], b_mu[1]]
    
    
    res, y = find_max_global(
        optimization_wrapper_lipo, 
        lower_bnds, 
        upper_bnds,
        5000
    )
    
    return res, y
