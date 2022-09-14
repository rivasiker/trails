import numpy as np
import pandas as pd
from ilsmc.get_emission_prob_mat import get_emission_prob_mat
from ilsmc.get_joint_prob_mat import get_joint_prob_mat
from scipy.optimize import minimize, shgo
from csv import writer

def forward_loglik(a, b, pi, V):
    alpha = np.zeros((V.shape[0], a.shape[0]))
    alpha[0, :] = np.log(pi * b[:, V[0]])
    for t in range(1, V.shape[0]):
        x = alpha[t-1, :].max()
        for j in range(a.shape[0]):
            alpha[t, j] = np.log(np.exp(alpha[t - 1]-x).dot(a[:, j]) * b[j, V[t]])+x
    return np.log(np.exp(alpha[len(V)-1]-x).sum())+x

def write_list(lst):
    with open('results.csv', 'a') as f:
        for i in range(len(lst)):
            f.write(str(lst[i]))
            if i != (len(lst)-1):
                f.write(',')
        f.write('\n')

def trans_emiss_calc(t_1, t_2, t_upper, N_AB, N_ABC, r, mu, n_int_AB, n_int_ABC):
    N_ref = N_AB
    t_A = t_1
    t_B = t_1
    t_AB = t_2
    t_C = t_1+t_2
    t_peak = 2*(N_ABC/N_ref)
    rho_A = 2*N_AB*r
    rho_B = 2*N_AB*r
    rho_AB = 2*N_AB*r
    rho_C = 2*N_AB*r
    rho_ABC = 2*N_ABC*r
    coal_A = N_ref/N_ref
    coal_B = N_ref/N_ref
    coal_AB = N_AB/N_ref
    coal_C = N_ref/N_ref
    coal_ABC = N_ABC/N_ref
    mu_A = mu
    mu_B = mu
    mu_C = mu
    mu_D = mu
    mu_AB = mu
    mu_ABC = mu
    
    tr = get_joint_prob_mat(
        t_A,    t_B,    t_AB,    t_C, 
        rho_A,  rho_B,  rho_AB,  rho_C,  rho_ABC, 
        coal_A, coal_B, coal_AB, coal_C, coal_ABC,
        n_int_AB, n_int_ABC)
    tr = pd.DataFrame(tr, columns=['From', 'To', 'Prob']).pivot(index = ['From'], columns = ['To'], values = ['Prob'])
    tr.columns = tr.columns.droplevel()
    hidden_names = list(tr.columns)
    hidden_names = dict(zip(hidden_names, range(len(hidden_names))))
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
    observed_names = dict(zip(observed_names, range(len(observed_names))))
    b = np.array(em)
    
    return a, b, pi, hidden_names, observed_names

def optimization_wrapper(arg_lst, n_int_AB, n_int_ABC, V, info):
    t_1, t_2, t_upper, N_AB, N_ABC, r, mu = arg_lst
    a, b, pi, hidden_names, observed_names = trans_emiss_calc(
        t_1, t_2, t_upper, N_AB, N_ABC, r, mu, n_int_AB, n_int_ABC
    )
    loglik = forward_loglik(a, b, pi, V)
    write_list([info['Nfeval'], t_1, t_2, t_upper, N_AB, N_ABC, r, mu, loglik])
    print(
        '{0:4d}   {1: 3.6f}   {2: 3.6f}   {3: 3.6f}   {4: 3.6f}   {5: 3.6f}   {6: 3.6f}   {7: 3.6f}   {8: 3.6f}'.format(
            info['Nfeval'], 
            arg_lst[0], arg_lst[1], arg_lst[2], arg_lst[3], 
            arg_lst[4], arg_lst[5], arg_lst[6], loglik
        )
    )
    info['Nfeval'] += 1
    return -loglik

def optimizer(t_1, t_2, t_upper, N_AB, N_ABC, r, mu, n_int_AB, n_int_ABC, V):
    init_params = np.array([t_1, t_2, t_upper, N_AB, N_ABC, r, mu])
    b_t = (0.01, 10)
    b_N = (1, 500000)
    b_r = (0.000000001, 0.00001)
    b_mu = (0.0001, 0.1)
    bnds = (b_t, b_t, b_t, b_N, b_N, b_r, b_mu)
    res = minimize(
        optimization_wrapper, 
        x0 = init_params,
        args = (n_int_AB, n_int_ABC, V, {'Nfeval':0}),
        method = 'Nelder-Mead',
        bounds = bnds, 
        options = {
            'disp': True
        }
    )
    
    return res