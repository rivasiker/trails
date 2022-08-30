import numpy as np
from scipy.integrate import dblquad
from scipy.integrate import quad
import pandas as pd
from scipy.linalg import expm
from scipy.special import comb
from ilsmc.cutpoints import cutpoints_AB, cutpoints_ABC

def rate_mat_JC69(mu):
    """
    This function returns the rate matrix for the JC69 model. 
    
    Parameters
    ----------
    mu : numeric
        Mutation rate
    """
    return np.full((4, 4), mu/4)-np.diag([mu, mu, mu, mu])

def p_b_given_a(t, Q):
    """
    This function calculates the probability of observing the 
    nucleotide b given a, t and Q. a is the starting nucleotide,
    while b is the end nucleotide. t is the total time of the interval. 
    
    P(b = bb | a == aa, Q, t)
    
    Parameters
    ----------
    t : numeric
        Total time of the interval (from a/b to c)
    Q : numpy array
        A 4x4 rate matrix for any substitution model
    """
    nt = ['A', 'G', 'C', 'T']
    mat = np.zeros((4, 4))
    for i in range(len(t)):
        mat = mat+t[i]*Q[i]
    arr = np.empty((4**2, 3))
    acc = 0
    mat = expm(mat)
    for aa in range(4):
        for bb in range(4):
            arr[acc] = [aa,bb,mat[aa,bb]]
            acc += 1
    df = pd.DataFrame(arr, columns = ['a', 'b', 'prob'])
    df['a'] = [nt[int(i)] for i in df['a']]
    df['b'] = [nt[int(i)] for i in df['b']]
    return df

def g_single_coal_JC69(mu, aa, bb, cc, dd, t, u):
    prm = np.zeros(3)
    prm[0] = 3/4 if aa==dd else -1/4
    prm[1] = 3/4 if dd==bb else -1/4
    prm[2] = 3/4 if dd==cc else -1/4
    tmp = 1
    tm = [-mu*u,-mu*u,-mu*(t-u)]
    for i in range(3):
        tmp = tmp*(1/4+prm[i]*np.exp(tm[i]))
    tmp = tmp*np.exp(-u)
    return tmp
def p_b_c_given_a_JC69(t, mu):
    nt = ['A', 'G', 'C', 'T']
    arr = np.empty((4**3, 4))
    acc = 0
    for aa in range(4):
        for bb in range(4):
            for cc in range(4):
                cumsum = 0
                for dd in range(4):
                    res, err = quad(lambda u: g_single_coal_JC69(mu, aa, bb, cc, dd, t, u), 0, t)
                    cumsum += res/(1-np.exp(-t))
                arr[acc] = [aa,bb,cc,cumsum]
                acc += 1
    df = pd.DataFrame(arr, columns = ['a', 'b', 'c', 'prob'])
    df['a'] = [nt[int(i)] for i in df['a']]
    df['b'] = [nt[int(i)] for i in df['b']]
    df['c'] = [nt[int(i)] for i in df['c']]
    return df

def g_double_coal_JC69(mu, aa, bb, cc, dd, ee, ff, t, u, v):
    prm = np.zeros(5)
    prm[0] = 3/4 if aa==ee else -1/4
    prm[1] = 3/4 if ee==bb else -1/4
    prm[2] = 3/4 if ee==ff else -1/4
    prm[3] = 3/4 if ff==cc else -1/4
    prm[4] = 3/4 if ff==dd else -1/4
    tmp = 1
    tm = [-mu*u,-mu*u,-mu*(v-u),-mu*v,-mu*(t-v)]
    for i in range(5):
        tmp = tmp*(1/4+prm[i]*np.exp(tm[i]))
    tmp = tmp*3*np.exp(-3*u)*np.exp(-(v-u))
    return tmp
def p_b_c_d_given_a_JC69(t, mu):
    nt = ['A', 'G', 'C', 'T']
    arr = np.empty((4**4, 5))
    acc = 0
    for aa in range(4):
        for bb in range(4):
            for cc in range(4):
                for dd in range(4):
                    cumsum = 0
                    for ee in range(4):
                        for ff in range(4):
                            res, err = dblquad(lambda v, u: g_double_coal_JC69(mu, aa, bb, cc, dd, ee, ff, t, u, v), 0, t, lambda u: u, t)
                            cumsum += res
                    arr[acc] = [aa,bb,cc,dd,cumsum]
                    acc += 1
    df = pd.DataFrame(arr, columns = ['a', 'b', 'c', 'd', 'prob'])
    df['a'] = [nt[int(i)] for i in df['a']]
    df['b'] = [nt[int(i)] for i in df['b']]
    df['c'] = [nt[int(i)] for i in df['c']]
    df['d'] = [nt[int(i)] for i in df['d']]
    df['prob'] = [i/(1+0.5*np.exp(-3*t)-1.5*np.exp(-t)) for i in df['prob']]
    return df


def b_c_d_given_a_to_dict_a_b_c_d(df):
    """
    This function converts the data frame as outputted
    by p_b_c_given_a_single_coal or p_b_c_given_a_double_coal
    into a dictionary. How to use the dictionary:
    
        P(b, c, d | a) = dct[a][b][c][d]
    
    Parameters
    ----------
    df : data frame
        As outputted by p_b_c_given_a_double_coal
    """
    # df = df.groupby(['a', 'b', 'c', 'd']).sum().reset_index()
    df = df.groupby(['a', 'b', 'c']).apply(lambda x: dict(zip(x.d, x.prob))).reset_index()
    df.columns = ['a', 'b', 'c', 'val']
    df = df.groupby(['a', 'b']).apply(lambda x: dict(zip(x.c, x.val))).reset_index()
    df.columns = ['a', 'b', 'val']
    df = df.groupby('a').apply(lambda x: dict(zip(x.b, x.val))).to_dict()
    return df


def b_c_given_a_to_dict_a_b_c(df):
    """
    This function converts the data frame as outputted
    by p_b_c_given_a_single_coal or p_b_c_given_a_double_coal
    into a dictionary. How to use the dictionary:
    
        P(b, c | a) = dct[a][b][c]
    
    Parameters
    ----------
    df : data frame
        As outputted by p_b_c_given_a_single_coal
    """
    # df = df.groupby(['a', 'b', 'c']).sum().reset_index()
    df = df.groupby(['a', 'b']).apply(lambda x: dict(zip(x.c, x.prob))).reset_index()
    df.columns = ['a', 'b', 'val']
    df = df.groupby('a').apply(lambda x: dict(zip(x.b, x.val))).to_dict()
    return df


def b_given_a_to_dict_a_b(df):
    """
    This function converts the data frame as outputted
    by p_b_given_a into a dictionary. How to use the dictionary:
    
        P(b | a) = dct[a][b]
    
    Parameters
    ----------
    df : data frame
        As outputted by p_b_given_a
    """
    return df.groupby(['a']).apply(lambda x: dict(zip(x.b, x.prob))).to_dict()

def calc_emissions_single_JC69(
    a0_a1_t_vec, b0_b1_t_vec, a1b1_ab0_t, ab0_ab1_t_vec, 
    ab1c1_abc0_t, c0_c1_t_vec, d0_abc0_t_vec,
    a0_a1_mu_vec, b0_b1_mu_vec, a1b1_ab0_mu, ab0_ab1_mu_vec, 
    ab1c1_abc0_mu, c0_c1_mu_vec, d0_abc0_mu_vec
):
  
    # a0 to a1
    Q_vec = [rate_mat_JC69(i) for i in a0_a1_mu_vec]
    df_a = p_b_given_a(t = a0_a1_t_vec, Q = Q_vec)
    df_a = b_given_a_to_dict_a_b(df_a)
    # df_a[a0][a1]
    
    # b1 to b0
    Q_vec = [rate_mat_JC69(i) for i in list(reversed(b0_b1_mu_vec))]
    df_b = p_b_given_a(t = list(reversed(b0_b1_t_vec)), Q = Q_vec)
    df_b = b_given_a_to_dict_a_b(df_b)
    # df_b[b1][b0]
    
    # c1 to c0
    Q_vec = [rate_mat_JC69(i) for i in list(reversed(c0_c1_mu_vec))]
    df_c = p_b_given_a(t = list(reversed(c0_c1_t_vec)), Q = Q_vec)
    df_c = b_given_a_to_dict_a_b(df_c)
    # df_c[c1][c0]
    
    # abc0 to d0
    Q_vec = [rate_mat_JC69(i) for i in list(reversed(d0_abc0_mu_vec))]
    df_d = p_b_given_a(t = list(reversed(d0_abc0_t_vec)), Q = Q_vec)
    df_d = b_given_a_to_dict_a_b(df_d)
    # df_d[abc0][d0]
    
    # ab0 to ab1
    Q_vec = [rate_mat_JC69(i) for i in ab0_ab1_mu_vec]
    df_ab = p_b_given_a(t = ab0_ab1_t_vec, Q = Q_vec)
    df_ab = b_given_a_to_dict_a_b(df_ab)
    # df_ab[ab0][ab1]
    
    # First coalescent
    df_first = p_b_c_given_a_JC69(t = a1b1_ab0_t, mu = a1b1_ab0_mu)
    df_first = b_c_given_a_to_dict_a_b_c(df_first)
    # df_first[a1][b1][ab0]
    
    # Second coalescent
    df_second = p_b_c_given_a_JC69(t = ab1c1_abc0_t, mu = ab1c1_abc0_mu)
    df_second = b_c_given_a_to_dict_a_b_c(df_second)
    # df_second[a1][b1][ab0]
    
    emissions = {}
    for a0 in ['A', 'C', 'T', 'G']:
        for b0 in ['A', 'C', 'T', 'G']:
            for c0 in ['A', 'C', 'T', 'G']:
                for d0 in ['A', 'C', 'T', 'G']:
                    acc = 0
                    for a1 in ['A', 'C', 'T', 'G']:
                        for b1 in ['A', 'C', 'T', 'G']:
                            for c1 in ['A', 'C', 'T', 'G']:
                                for ab0 in ['A', 'C', 'T', 'G']:
                                    for ab1 in ['A', 'C', 'T', 'G']:
                                        for abc0 in ['A', 'C', 'T', 'G']:
                                            res = 1
                                            res = res*df_a[a0][a1]
                                            res = res*df_b[b1][b0]
                                            res = res*df_first[a1][ab0][b1]
                                            res = res*df_ab[ab0][ab1]
                                            res = res*df_second[ab1][abc0][c1]
                                            res = res*df_c[c1][c0]
                                            res = res*df_d[abc0][d0]
                                            acc += res
                    emissions[a0+b0+c0+d0] = acc/4
                
    return emissions


def calc_emissions_double_JC69(
    a0_a1_t_vec, b0_b1_t_vec, c0_c1_t_vec, a1b1c1_abc0_t, d0_abc0_t_vec,
    a0_a1_mu_vec, b0_b1_mu_vec, c0_c1_mu_vec, a1b1c1_abc0_mu, d0_abc0_mu_vec
):
    # a0 to a1
    Q_vec = [rate_mat_JC69(i) for i in a0_a1_mu_vec]
    df_a = p_b_given_a(t = a0_a1_t_vec, Q = Q_vec)
    df_a = b_given_a_to_dict_a_b(df_a)
    # df_a[a0][a1]
    
    # b1 to b0
    Q_vec = [rate_mat_JC69(i) for i in list(reversed(b0_b1_mu_vec))]
    df_b = p_b_given_a(t = list(reversed(b0_b1_t_vec)), Q = Q_vec)
    df_b = b_given_a_to_dict_a_b(df_b)
    # df_b[b1][b0]
    
    # c1 to c0
    Q_vec = [rate_mat_JC69(i) for i in list(reversed(c0_c1_mu_vec))]
    df_c = p_b_given_a(t = list(reversed(c0_c1_t_vec)), Q = Q_vec)
    df_c = b_given_a_to_dict_a_b(df_c)
    # df_c[c1][c0]
    
    # abc0 to d0 
    Q_vec = [rate_mat_JC69(i) for i in list(reversed(d0_abc0_mu_vec))]
    df_d = p_b_given_a(t = list(reversed(d0_abc0_t_vec)), Q = Q_vec)
    df_d = b_given_a_to_dict_a_b(df_d)
    # df_d[abc0][d0]
    
    # Double coalescent
    df_double = p_b_c_d_given_a_JC69(t = a1b1c1_abc0_t, mu = a1b1c1_abc0_mu)
    df_double = b_c_d_given_a_to_dict_a_b_c_d(df_double)
    # df_double[a1][b1][c1][abc0]
    
    emissions = {}
    for a0 in ['A', 'C', 'T', 'G']:
        for b0 in ['A', 'C', 'T', 'G']:
            for c0 in ['A', 'C', 'T', 'G']:
                for d0 in ['A', 'C', 'T', 'G']:
                    acc = 0
                    for a1 in ['A', 'C', 'T', 'G']:
                        for b1 in ['A', 'C', 'T', 'G']:
                            for c1 in ['A', 'C', 'T', 'G']:
                                for abc0 in ['A', 'C', 'T', 'G']:
                                    res = 1
                                    res = res*df_a[a0][a1]
                                    res = res*df_b[b1][b0]
                                    res = res*df_c[c1][c0]
                                    res = res*df_double[a1][b1][c1][abc0]
                                    res = res*df_d[abc0][d0]
                                    acc += res
                    emissions[a0+b0+c0+d0] = acc/4
    return emissions

def get_emission_prob_mat(t_A,    t_B,    t_AB,    t_C,    t_upper,   t_peak,
                      rho_A,  rho_B,  rho_AB,  rho_C,  rho_ABC, 
                      coal_A, coal_B, coal_AB, coal_C, coal_ABC,
                      n_int_AB, n_int_ABC,
                      mu_A, mu_B, mu_C, mu_D, mu_AB, mu_ABC
):
    
    
    n_markov_states = n_int_AB*n_int_ABC+n_int_ABC*3+3*comb(n_int_ABC, 2, exact = True)
    cut_AB = cutpoints_AB(n_int_AB, t_AB, coal_AB)
    cut_ABC = cutpoints_ABC(n_int_ABC, coal_ABC)
    
    # V1 double coal
    
    V1_probs = np.zeros((n_int_ABC, 4**4))
    V1_states = np.empty((n_int_ABC), dtype=object)
    acc = 0
    for i in range(n_int_ABC):
        markov = (1, i, i)
        V1_states[acc] = markov
        
        a0_a1_t_vec = [t_A, t_AB, cut_ABC[i]]
        a0_a1_mu_vec = [mu_A, mu_AB, mu_ABC]
        b0_b1_t_vec = [t_B, t_AB, cut_ABC[i]]
        b0_b1_mu_vec = [mu_B, mu_AB, mu_ABC]
        c0_c1_t_vec = [t_C, cut_ABC[i]]
        c0_c1_mu_vec = [mu_C, mu_ABC]
        a1b1c1_abc0_t = (cut_ABC[i+1]-cut_ABC[i]) if i!=(n_int_ABC-1) else t_upper
        a1b1c1_abc0_mu = mu_ABC
        add = t_upper+cut_ABC[n_int_ABC-1]-cut_ABC[i+1] if i!=(n_int_ABC-1) else 0
        d0_abc0_t_vec = [t_A+t_AB+cut_ABC[n_int_ABC-1]+t_upper]+[t_peak+add]
        d0_abc0_mu_vec = [mu_D, mu_ABC]

        emissions = calc_emissions_double_JC69(
            a0_a1_t_vec, b0_b1_t_vec, c0_c1_t_vec, a1b1c1_abc0_t, d0_abc0_t_vec,
            a0_a1_mu_vec, b0_b1_mu_vec, c0_c1_mu_vec, a1b1c1_abc0_mu, d0_abc0_mu_vec
        )
        V1_probs[acc] = list(emissions.values())
        acc += 1
    V1_probs = pd.DataFrame(V1_probs)
    V1_probs.columns = list(emissions.keys())
    V1_probs.insert(0, 'hidden_state', V1_states)
    
    return V1_probs
    
    
    # V1 single coal
    
    V1_probs = np.zeros((comb(n_int_ABC, 2, exact = True), 4**4))
    V1_states = np.empty((comb(n_int_ABC, 2, exact = True)), dtype=object)
    acc = 0
    for i in range(n_int_ABC):
        for j in range(i+1, n_int_ABC):
            markov = (1, i, j)
            V1_states[acc] = markov
            
            a0_a1_t_vec = [t_A, t_AB, cut_ABC[i]]
            a0_a1_mu_vec = [mu_A, mu_AB, mu_ABC]
            b0_b1_t_vec = [t_B, t_AB, cut_ABC[i]]
            b0_b1_mu_vec = [mu_B, mu_AB, mu_ABC]
            c0_c1_t_vec = [t_C, cut_ABC[j]]
            c0_c1_mu_vec = [mu_C, mu_ABC]
            ab0_ab1_t_vec = [cut_ABC[j]-cut_ABC[i+1]]
            ab0_ab1_mu_vec = [mu_ABC]
            a1b1_ab0_t = cut_ABC[i+1]-cut_ABC[i]
            a1b1_ab0_mu = mu_ABC
            ab1c1_abc0_t = (cut_ABC[j+1]-cut_ABC[j]) if j!=(n_int_ABC-1) else t_upper
            ab1c1_abc0_mu = mu_ABC
            add = t_upper+cut_ABC[n_int_ABC-1]-cut_ABC[j+1] if j!=(n_int_ABC-1) else 0
            d0_abc0_t_vec = [t_A+t_AB+cut_ABC[n_int_ABC-1]+t_upper]+[t_peak+add]
            d0_abc0_mu_vec = [mu_D, mu_ABC]
            
            emissions = calc_emissions_single_JC69(
                a0_a1_t_vec, b0_b1_t_vec, a1b1_ab0_t, ab0_ab1_t_vec, 
                ab1c1_abc0_t, c0_c1_t_vec, d0_abc0_t_vec,
                a0_a1_mu_vec, b0_b1_mu_vec, a1b1_ab0_mu, ab0_ab1_mu_vec, 
                ab1c1_abc0_mu, c0_c1_mu_vec, d0_abc0_mu_vec
            )
            V1_probs[acc] = list(emissions.values())
            acc += 1
    V1_probs = pd.DataFrame(V1_probs)
    V1_probs.columns = list(emissions.keys())
    V1_probs.insert(0, 'hidden_state', V1_states)
    
    return V1_probs
            
    
    # V0
    
    V0_probs = np.zeros((n_int_AB*n_int_ABC, 4**4))
    V0_states = np.empty((n_int_AB*n_int_ABC), dtype=object)
    acc = 0
    for i in range(n_int_AB):
        for j in range(n_int_ABC):
            markov = (0, i, j)
            V0_states[acc] = markov
            a0_a1_t_vec = [t_A, cut_AB[i]]
            a0_a1_mu_vec = [mu_A, mu_AB]
            b0_b1_t_vec = [t_B, cut_AB[i]]
            b0_b1_mu_vec = [mu_B, mu_AB]
            c0_c1_t_vec = [t_C, cut_ABC[j]]
            c0_c1_mu_vec = [mu_C, mu_ABC]
            ab0_ab1_t_vec = [t_AB-cut_AB[i+1], cut_ABC[j]]
            ab0_ab1_mu_vec = [mu_AB, mu_ABC]
            a1b1_ab0_t = cut_AB[i+1]-cut_AB[i]
            a1b1_ab0_mu = mu_AB
            ab1c1_abc0_t = cut_ABC[j+1]-cut_ABC[j] if j!=(n_int_ABC-1) else t_upper
            ab1c1_abc0_mu = mu_ABC
            add = t_upper+cut_ABC[n_int_ABC-1]-cut_ABC[j+1] if j!=(n_int_ABC-1) else 0
            d0_abc0_t_vec = [t_A+t_AB+cut_ABC[n_int_ABC-1]+t_upper]+[t_peak+add]
            d0_abc0_mu_vec = [mu_D, mu_ABC]
            
            emissions = calc_emissions_single_JC69(
                a0_a1_t_vec, b0_b1_t_vec, a1b1_ab0_t, ab0_ab1_t_vec, 
                ab1c1_abc0_t, c0_c1_t_vec, d0_abc0_t_vec,
                a0_a1_mu_vec, b0_b1_mu_vec, a1b1_ab0_mu, ab0_ab1_mu_vec, 
                ab1c1_abc0_mu, c0_c1_mu_vec, d0_abc0_mu_vec
            )
            V0_probs[acc] = list(emissions.values())
            acc += 1
    V0_probs = pd.DataFrame(V0_probs)
    V0_probs.columns = list(emissions.keys())
    V0_probs.insert(0, 'hidden_state', V0_states)
            
    return V0_probs