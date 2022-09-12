import numpy as np
import pandas as pd
from ilsmc.get_emission_prob_mat import get_emission_prob_mat
from ilsmc.get_joint_prob_mat import get_joint_prob_mat

def forward_loglik(V, a, b, pi):
    alpha = np.zeros((V.shape[0], a.shape[0]))
    alpha[0, :] = np.log(pi * b[:, V[0]])
    for t in range(1, V.shape[0]):
        x = max(alpha[t-1, :])
        for j in range(a.shape[0]):
            alpha[t, j] = np.log(np.exp(alpha[t - 1]-x).dot(a[:, j]) * b[j, V[t]])+x
    return np.log(np.exp(alpha[len(V)-1]-x).sum())+x

def optimization_wrapper(t_1, t_2, t_upper, N_A, N_B, N_C, N_D, N_AB, N_ABC, r, mu, n_int_AB, n_int_ABC):
    N_ref = N_AB
    t_A = t_1
    t_B = t_1
    t_AB = t_2
    t_C = t_1+t_2
    t_peak = 2*(N_ABC/N_ref)
    rho_A = 2*N_A*r
    rho_B = 2*N_B*r
    rho_AB = 2*N_AB*r
    rho_C = 2*N_C*r
    rho_ABC = 2*N_ABC*r
    coal_A = N_A/N_ref
    coal_B = N_B/N_ref
    coal_AB = N_AB/N_ref
    coal_C = N_C/N_ref
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
    arr = np.array(tr)
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