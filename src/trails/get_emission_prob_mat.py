import numpy as np
import pandas as pd
from scipy.linalg import expm
from scipy.special import comb
from trails.cutpoints import cutpoints_AB, cutpoints_ABC
from numba import njit

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
    
    P(b = bb | a == aa, Q, t)
    
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

@njit
def JC69_analytical_integral(aa, bb, cc, dd, t, mu):
    """
    This function calculates the probability of observing the 
    nucleotides bb, cc and dd given aa, t and mu. aa and bb are the starting 
    nucleotides, while cc is the end nucleotide. dd is the nucleotide at 
    the time of coalescent. t is the total time of the interval. The 
    returned value corresponds to integrating the coalescent to d over
    the entirety of t. 
    
    P(b = bb, c == cc, d == dd | a == aa, mu, t)
    
          c     ^
          |     |
        __d__   | t
       |     |  |
       a     b  |
    
    Parameters
    ----------
    aa, bb, cc, dd : integer or string
        nucleotide at a, b, c and d respectively
    t : numeric
        Total time of the interval (from a/b/c to d)
    mu : numeric
        The mutation rate for the JC69 model
    """
    alpha = 3/4 if aa==dd else -1/4
    beta  = 3/4 if dd==bb else -1/4
    gamma = 3/4 if dd==cc else -1/4
    
    res = (1 + (16*beta*gamma)/np.exp(mu*t) - 
           (4*gamma)/(np.exp(mu*t)*(-1 + mu)) + 
           (4*beta)/(1 + mu) + 
           4*alpha*((1 + mu)**(-1) + 
                    (4*gamma*(1 + 4*beta + mu))/
                    (np.exp(mu*t)*(1 + mu)) + 
                    (4*beta)/(1 + 2*mu)) + 
           (-1 - (16*beta*gamma)/np.exp(mu*t) + 
            (4*gamma)/(-1 + mu) - 
            (4*beta)/(np.exp(mu*t)*(1 + mu)) - 
            (4*alpha*(np.exp(mu*t)*(1 + 2*mu)*
                      (1 + 4*gamma*(1 + mu)) + 
                      4*beta*(1 + mu + gamma*
                              (4 + 8*mu))))/(np.exp(2*mu*t)*
                                             (1 + mu)*(1 + 2*mu)))/np.exp(t))/(64*(1 - np.exp(-t)))
    return res
def p_b_c_given_a_JC69_analytical(t, mu):
    """
    This function returns a data frame with the values of
    P(b, c | a) for all combinations of nucleotides. 
    
    Parameters
    ----------
    t : numeric
        Total time of the interval (from a/b/c to d)
    mu : numeric
        The mutation rate for the JC69 model
    """
    nt = ['A', 'G', 'C', 'T']
    arr = np.empty((4**3, 4))
    acc = 0
    for aa in range(4):
        for bb in range(4):
            for cc in range(4):
                cumsum = 0
                for dd in range(4):
                    cumsum += JC69_analytical_integral(aa, bb, cc, dd, t, mu)
                arr[acc] = [aa,bb,cc,cumsum]
                acc += 1
    df = pd.DataFrame(arr, columns = ['a', 'b', 'c', 'prob'])
    df['a'] = [nt[int(i)] for i in df['a']]
    df['b'] = [nt[int(i)] for i in df['b']]
    df['c'] = [nt[int(i)] for i in df['c']]
    return df

@njit
def JC69_analytical_integral_double(aa, bb, cc, dd, ee, ff, t, mu):
    """
    This function calculates the probability of observing the 
    nucleotides bb, cc, dd, ee and ff given aa, t and mu. aa, bb 
    and cc are the starting nucleotides, while dd is the end nucleotide. 
    ee is the nucleotide at the time of the first coalescent, while
    ff is the nucleotide at the time of the second coalescent. t is the 
    total time of the interval. The returned value corresponds to integrating 
    the coalescent to e and f over the entirety of t. 
    
    P(b = bb, c == cc, d == dd, e == ee, f == ff | a == aa, Q, t)
    
              d       ^
              |       |
           ___f___    |
          |       |   | t
        __e__     |   |
       |     |    |   |
       a     b    c   |
    
    Parameters
    ----------
    aa, bb, cc, dd, ee, ff : integer or string
        nucleotide at a, b, c, d, e and f, respectively
    t : numeric
        Total time of the interval (from a/b/c to d)
    mu : numeric
        The mutation rate for the JC69 model
    """
    
    alpha   = 3/4 if aa==ee else -1/4
    beta    = 3/4 if ee==bb else -1/4
    gamma   = 3/4 if ee==ff else -1/4
    delta   = 3/4 if ff==cc else -1/4
    epsilon = 3/4 if ff==dd else -1/4
    
    res = (3*((-2*delta*(-2 - 8*gamma + mu))/
        (-6 + mu + mu**2) - 
       (32*alpha*beta*delta*(2 + mu + 
          8*gamma*(1 + mu)))/
        (3*(1 + mu)**2*(2 + mu)) - 
       (32*alpha*beta*epsilon*
         (2 + mu + 8*gamma*(1 + mu)))/
        (np.exp(mu*t)*(1 + mu)*(2 + mu)*
         (3 + mu)) - (8*alpha*beta*
         (1 + (16*delta*epsilon)/
           np.exp(mu*t))*(2 + mu + 
          8*gamma*(1 + mu)))/
        ((1 + mu)*(2 + mu)*(3 + 2*mu)) + 
       (16*delta*gamma*
         ((-1 + 2*beta*(-2 + mu))*
           (2 + mu) + 2*alpha*(-2 + mu)*
           (2 + 8*beta + mu)))/
        ((-2 + mu)*(2 + mu)*(1 + 2*mu)) - 
       (4*(alpha + beta)*
         (1 + 2*gamma*(2 + mu))*
         ((3 + 2*mu)*(3*np.exp(mu*t) + 
            4*epsilon*(3 + mu)) + 
          12*delta*(np.exp(mu*t)*(3 + mu) + 
            4*epsilon*(3 + 2*mu))))/
        (3*np.exp(mu*t)*(2 + mu)*(3 + mu)*
         (3 + 2*mu)) - 
       (2*epsilon*((2 + 8*gamma - mu)/
           ((-3 + mu)*(-2 + mu)) + 
          ((1 + mu)*(2 + 8*beta + mu) + 
            8*alpha*(1 + mu + 2*beta*
               (2 + mu)))/((-1 + mu)*
            (1 + mu)*(2 + mu))))/
        np.exp(mu*t) - 
       (-16*delta*epsilon*(2 + 8*gamma - 
           mu)*(2 + 3*mu + mu**2) + 
         np.exp(mu*t)*(-2 - 8*gamma + mu)*
          (2 + 3*mu + mu**2) - 
         3*np.exp(mu*t)*(-2 + mu)*
          ((1 + mu)*(2 + 8*beta + mu) + 
           8*alpha*(1 + mu + 2*beta*
              (2 + mu))) - 48*epsilon*
          (2*gamma*(1 + mu)*
            ((-1 + 2*beta*(-2 + mu))*
              (2 + mu) + 2*alpha*
              (-2 + mu)*(2 + 8*beta + 
               mu)) + delta*(-2 + mu)*
            ((1 + mu)*(2 + 8*beta + mu) + 
             8*alpha*(1 + mu + 2*beta*
                (2 + mu)))))/
        (6*np.exp(mu*t)*(-2 + mu)*(1 + mu)*
         (2 + mu)) + 
       (2*(2*np.exp(mu*t)*gamma*(1 + mu)*
           ((-1 + 2*beta*(-2 + mu))*
             (2 + mu) + 2*alpha*(-2 + mu)*
             (2 + 8*beta + mu)) + 
          delta*(32*epsilon*gamma*
             (1 + mu)*
             ((-1 + 2*beta*(-2 + mu))*
               (2 + mu) + 2*alpha*
               (-2 + mu)*(2 + 8*beta + 
                mu)) + np.exp(mu*t)*(-2 + mu)*
             ((1 + mu)*(2 + 8*beta + 
                mu) + 8*alpha*(1 + mu + 
                2*beta*(2 + mu))))))/
        (np.exp(mu*t)*(1 + mu)**2*
         (-4 + mu**2)) + 
       ((32*alpha*beta*delta*(2 + mu + 
            8*gamma*(1 + mu)))/
          (3*(1 + mu)**2*(2 + mu)) + 
         (32*alpha*beta*np.exp(mu*t)*epsilon*
           (2 + mu + 8*gamma*(1 + mu)))/
          ((1 + mu)*(2 + mu)*(3 + mu)) + 
         (8*alpha*beta*np.exp(mu*t)*
           (1 + (16*delta*epsilon)/
             np.exp(mu*t))*(2 + mu + 
            8*gamma*(1 + mu)))/
          ((1 + mu)*(2 + mu)*
           (3 + 2*mu)) + 
         (4*(alpha + beta)*
           (1 + 2*gamma*(2 + mu))*
           ((3 + 2*mu)*(3*np.exp(2*mu*t) + 
              4*np.exp(2*mu*t)*epsilon*
               (3 + mu)) + 12*delta*
             (np.exp(mu*t)*(3 + mu) + 
              4*np.exp(mu*t)*epsilon*
               (3 + 2*mu))))/(3*(2 + mu)*
           (3 + mu)*(3 + 2*mu)) + 
         np.exp(2*(1 + mu)*t)*
          ((2*delta*(-2 - 8*gamma + mu))/
            (np.exp(2*t)*(-6 + mu + mu**2)) - 
           (16*delta*gamma*
             ((-1 + 2*beta*(-2 + mu))*
               (2 + mu) + 2*alpha*
               (-2 + mu)*(2 + 8*beta + 
                mu)))/(np.exp(mu*t)*(-2 + mu)*
             (2 + mu)*(1 + 2*mu)) + 
           2*np.exp(mu*t)*epsilon*
            ((2 + 8*gamma - mu)/
              (np.exp(2*t)*(-3 + mu)*
               (-2 + mu)) + 
             ((1 + mu)*(2 + 8*beta + 
                mu) + 8*alpha*(1 + mu + 
                2*beta*(2 + mu)))/
              ((-1 + mu)*(1 + mu)*
               (2 + mu))) + 
           (-16*delta*epsilon*
              (2 + 8*gamma - mu)*
              (2 + 3*mu + mu**2) + 
             np.exp(mu*t)*(-2 - 8*gamma + mu)*
              (2 + 3*mu + mu**2) - 
             3*np.exp(2*t + mu*t)*(-2 + mu)*
              ((1 + mu)*(2 + 8*beta + 
                mu) + 8*alpha*(1 + mu + 
                2*beta*(2 + mu))) - 
             48*np.exp(2*t)*epsilon*
              (2*gamma*(1 + mu)*
                ((-1 + 2*beta*(-2 + mu))*
                (2 + mu) + 2*alpha*
                (-2 + mu)*(2 + 8*beta + 
                mu)) + delta*(-2 + mu)*
                ((1 + mu)*(2 + 8*beta + 
                mu) + 8*alpha*(1 + mu + 2*
                beta*(2 + mu)))))/
            (6*np.exp(2*t)*(-2 + mu)*(1 + mu)*
             (2 + mu)) - 
           (2*(2*np.exp(mu*t)*gamma*(1 + mu)*
               ((-1 + 2*beta*(-2 + mu))*
                (2 + mu) + 2*alpha*
                (-2 + mu)*(2 + 8*beta + 
                mu)) + delta*
               (32*epsilon*gamma*(1 + mu)*
                ((-1 + 2*beta*(-2 + mu))*(
                2 + mu) + 2*alpha*(-2 + 
                mu)*(2 + 8*beta + mu)) + 
                np.exp(mu*t)*(-2 + mu)*
                ((1 + mu)*(2 + 8*beta + 
                mu) + 8*alpha*(1 + mu + 
                2*beta*(2 + mu))))))/
            (np.exp(mu*t)*(1 + mu)**2*
             (-4 + mu**2))))/
        np.exp(3*(1 + mu)*t)))/(1024*(1 + 0.5/np.exp(3*t) - 1.5/np.exp(t)))
    return res

def p_b_c_d_given_a_JC69_analytical(t, mu):
    """
    This function returns a data frame with the values of
    P(b, c, d | a) for all combinations of nucleotides. 
    
    Parameters
    ----------
    t : numeric
        Total time of the interval (from a/b/c to d)
    mu : numeric
        The mutation rate for the JC69 model
    """
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
                            res = JC69_analytical_integral_double(aa, bb, cc, dd, ee, ff, t, mu)
                            cumsum += res
                    arr[acc] = [aa,bb,cc,dd,cumsum]
                    acc += 1
    df = pd.DataFrame(arr, columns = ['a', 'b', 'c', 'd', 'prob'])
    df['a'] = [nt[int(i)] for i in df['a']]
    df['b'] = [nt[int(i)] for i in df['b']]
    df['c'] = [nt[int(i)] for i in df['c']]
    df['d'] = [nt[int(i)] for i in df['d']]
    return df


def b_c_d_given_a_to_dict_a_b_c_d(df):
    """
    This function converts the data frame as outputted
    by p_b_c_given_a_single_coal or p_b_c_given_a_double_coal
    into a dictionary. How to use the dictionary:
    
        P(b, c, d | a) = dct[a][b][c][d]
    
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
    
        P(b, c | a) = dct[a][b][c]
    
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
    
        P(b | a) = dct[a][b]
    
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
    """
    This function returns the emission probabilities of a hidden
    state contining two coalescent events at different time intervals.
        
                    _________
                   |         |
         ---------abc0-----  |
               ____|___      |
              |        |     |
         ----ab1-------c1--  |
              |        |     |
         ----ab0-----  |     |
            __|__      |     |
           |     |     |     |
         --a1----b1--  |     |
           |     |     |     |
           a0    b0    c0    d0
        
    Parameters
    ----------
    a0_a1_t_vec, b0_b1_t_vec, c0_c1_t_vec, 
    d0_abc0_t_vec, ab0_ab1_t_vec : numeric list
        Each list contains the interval time for a site to mutate
        with a certain mutation rate, specified by *mu_vec
    a1b1_ab0_t, ab1c1_abc0_t : numeric
        Time interval when the first and the second coalescent
        can happen, respectively.
    a0_a1_mu_vec, b0_b1_mu_vec, c0_c1_mu_vec, 
    d0_abc0_mu_vec, ab0_ab1_mu_vec : numeric list
        Each list contains the mutation rates for each interval 
        defined in *t_vec
    a1b1_ab0_mu, ab1c1_abc0_mu : numeric
        Mutation rates for the first and second coalescent intervals, 
        respectively.
    """
  
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
    df_first = p_b_c_given_a_JC69_analytical(t = a1b1_ab0_t, mu = a1b1_ab0_mu)
    df_first = b_c_given_a_to_dict_a_b_c(df_first)
    # df_first[a1][b1][ab0]
    
    # Second coalescent
    df_second = p_b_c_given_a_JC69_analytical(t = ab1c1_abc0_t, mu = ab1c1_abc0_mu)
    df_second = b_c_given_a_to_dict_a_b_c(df_second)
    # df_second[ab1][c1][abc0]
    
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
                                            res = res*df_first[a1][b1][ab0]
                                            res = res*df_ab[ab0][ab1]
                                            res = res*df_second[ab1][c1][abc0]
                                            res = res*df_c[c1][c0]
                                            res = res*df_d[abc0][d0]
                                            acc += res
                    emissions[a0+b0+c0+d0] = acc/4
                
    return emissions


def calc_emissions_double_JC69(
    a0_a1_t_vec, b0_b1_t_vec, c0_c1_t_vec, a1b1c1_abc0_t, d0_abc0_t_vec,
    a0_a1_mu_vec, b0_b1_mu_vec, c0_c1_mu_vec, a1b1c1_abc0_mu, d0_abc0_mu_vec
):
    """
    This function returns the emission probabilities of a hidden
    state contining two coalescent events at the same time interval.
        
                    _________
                   |         |
         ---------abc0-----  |
               ____|___      |
            __|__      |     |
           |     |     |     |
         --a1----b1----c1--  |
           |     |     |     |
           a0    b0    c0    d0
        
    Parameters
    ----------
    a0_a1_t_vec, b0_b1_t_vec, c0_c1_t_vec, d0_abc0_t_vec : numeric list
        Each list contains the interval time for a site to mutate
        with a certain mutation rate, specified by *mu_vec
    a1b1c1_abc0_t : numeric
        Time interval for the coalescent events to happen.
    a0_a1_mu_vec, b0_b1_mu_vec, c0_c1_mu_vec, d0_abc0_mu_vec : numeric list
        Each list contains the mutation rates for each interval 
        defined in *t_vec
    a1b1c1_abc0_mu : numeric
        Mutation rates for the interval where coalescents happen.
    """
    
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
    df_double = p_b_c_d_given_a_JC69_analytical(t = a1b1c1_abc0_t, mu = a1b1c1_abc0_mu)
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




def get_emission_prob_mat(t_A,    t_B,    t_AB,    t_C,    t_upper,   t_out,
                          rho_A,  rho_B,  rho_AB,  rho_C,  rho_ABC, 
                          coal_A, coal_B, coal_AB, coal_C, coal_ABC,
                          n_int_AB, n_int_ABC,
                          mu_A, mu_B, mu_C, mu_D, mu_AB, mu_ABC,
                          cut_AB = 'standard', cut_ABC = 'standard'):
    """
    This function returns the emission probabilities of all hidden states
    given a set of population genetics parameters. 
        
            |          |
            |  ABC  |\ \
            | AB |\ \ \ \
            / /\ \ \ \ \ \
           / /  \ \ \ \ \ \
           A      B   C   D
        
    Parameters
    ----------
    t_A, t_B : numeric
        Time between present time and the first speciation time for
        species A and B, respectively. They show be equal. 
    t_AB : numeric
        Time between speciation events.
    t_C : numeric
        Time between present time and the second speciation time for 
        species C. It should be t_A (or t_B) + t_AB.
    t_upper : numeric
        Time between the last ABC interval and the third speciation time.
    t_peak : numeric
        Mean divergence time between ABC and D after the third speciation time.
        It should be 4*coal_ABC (or it can be estimated instead). 
    rho_A, rho_B, rho_AB, rho_C, rho_ABC : numeric
        Recombination rates for the A, B, AB, C and ABC intervals.
    coal_A, coal_B, coal_AB, coal_C, coal_ABC : numeric
        Coalescent rates for the A, B, AB, C and ABC intervals.
    n_int_AB, n_int_ABC : integer
        Number of intervals in the AB and ABC parts of the tree. 
    mu_A, mu_B, mu_C, mu_D, mu_AB, mu_ABC : numeric
        Mutation rate for the A, B, C, D, AB and ABC intervals.
    """
    n_markov_states = n_int_AB*n_int_ABC+n_int_ABC*3+3*comb(n_int_ABC, 2, exact = True)
    if cut_AB == 'standard':
        cut_AB = cutpoints_AB(n_int_AB, t_AB, coal_AB)
    if cut_ABC == 'standard':
        cut_ABC = cutpoints_ABC(n_int_ABC, coal_ABC)
    probs = np.empty((n_markov_states), dtype=object)
    states = np.empty((n_markov_states), dtype=object)
    
    # Deep coalescence, two single coalescents
    acc = 0
    for i in range(n_int_ABC):
        for j in range(i+1, n_int_ABC):
            
            a0_a1_t_vec = [t_A, t_AB, cut_ABC[i]]
            a0_a1_mu_vec = [mu_A, mu_AB, mu_ABC]
            b0_b1_t_vec = [t_B, t_AB, cut_ABC[i]]
            b0_b1_mu_vec = [mu_B, mu_AB, mu_ABC]
            c0_c1_t_vec = [t_C, cut_ABC[i]]
            c0_c1_mu_vec = [mu_C, mu_ABC]
            ab0_ab1_t_vec = [cut_ABC[j]-cut_ABC[i+1]]
            ab0_ab1_mu_vec = [mu_ABC]
            a1b1_ab0_t = cut_ABC[i+1]-cut_ABC[i]
            a1b1_ab0_mu = mu_ABC
            ab1c1_abc0_t = (cut_ABC[j+1]-cut_ABC[j]) if j!=(n_int_ABC-1) else t_upper
            ab1c1_abc0_mu = mu_ABC
            add = t_upper+cut_ABC[n_int_ABC-1]-cut_ABC[j+1] if j!=(n_int_ABC-1) else 0
            # d0_abc0_t_vec = [t_A+t_AB+cut_ABC[n_int_ABC-1]+t_upper]+[t_peak+add]
            d0_abc0_t_vec = [t_out]+[add]
            # d0_abc0_mu_vec = [mu_D, mu_ABC]
            d0_abc0_mu_vec = [mu_D, mu_ABC]
            
            # V1 states
            emissions = calc_emissions_single_JC69(
                a0_a1_t_vec, b0_b1_t_vec, a1b1_ab0_t, ab0_ab1_t_vec, 
                ab1c1_abc0_t, c0_c1_t_vec, d0_abc0_t_vec,
                a0_a1_mu_vec, b0_b1_mu_vec, a1b1_ab0_mu, ab0_ab1_mu_vec, 
                ab1c1_abc0_mu, c0_c1_mu_vec, d0_abc0_mu_vec
            )
            states[acc] = (1, i, j)
            probs[acc] = emissions
            acc += 1
            
            # V2 states
            emissions = calc_emissions_single_JC69(
                a0_a1_t_vec, c0_c1_t_vec, a1b1_ab0_t, ab0_ab1_t_vec, 
                ab1c1_abc0_t, b0_b1_t_vec, d0_abc0_t_vec,
                a0_a1_mu_vec, c0_c1_mu_vec, a1b1_ab0_mu, ab0_ab1_mu_vec, 
                ab1c1_abc0_mu, b0_b1_mu_vec, d0_abc0_mu_vec
            )
            new_emissions = {}
            for k in list(emissions.keys()):
                new_emissions[k[0]+k[2]+k[1]+k[3]] = emissions[k]     
            states[acc] = (2, i, j)
            probs[acc] = new_emissions
            acc += 1
            
            # V3 states
            emissions = calc_emissions_single_JC69(
                b0_b1_t_vec, c0_c1_t_vec, a1b1_ab0_t, ab0_ab1_t_vec, 
                ab1c1_abc0_t, a0_a1_t_vec, d0_abc0_t_vec,
                b0_b1_mu_vec, c0_c1_mu_vec, a1b1_ab0_mu, ab0_ab1_mu_vec, 
                ab1c1_abc0_mu, a0_a1_mu_vec, d0_abc0_mu_vec
            )
            new_emissions = {}
            for k in list(emissions.keys()):
                new_emissions[k[2]+k[0]+k[1]+k[3]] = emissions[k]     
            states[acc] = (3, i, j)
            probs[acc] = new_emissions
            acc += 1
        
    # Deep coalescence, one double coalescent
    for i in range(n_int_ABC):
        
        a0_a1_t_vec = [t_A, t_AB, cut_ABC[i]]
        a0_a1_mu_vec = [mu_A, mu_AB, mu_ABC]
        b0_b1_t_vec = [t_B, t_AB, cut_ABC[i]]
        b0_b1_mu_vec = [mu_B, mu_AB, mu_ABC]
        c0_c1_t_vec = [t_C, cut_ABC[i]]
        c0_c1_mu_vec = [mu_C, mu_ABC]
        a1b1c1_abc0_t = (cut_ABC[i+1]-cut_ABC[i]) if i!=(n_int_ABC-1) else t_upper
        a1b1c1_abc0_mu = mu_ABC
        add = t_upper+cut_ABC[n_int_ABC-1]-cut_ABC[i+1] if i!=(n_int_ABC-1) else 0
        # d0_abc0_t_vec = [t_A+t_AB+cut_ABC[n_int_ABC-1]+t_upper]+[t_peak+add]
        d0_abc0_t_vec = [t_out]+[add]
        # d0_abc0_mu_vec = [mu_D, mu_ABC]
        d0_abc0_mu_vec = [mu_D, mu_ABC]
        
        # V1 states
        emissions = calc_emissions_double_JC69(
            a0_a1_t_vec, b0_b1_t_vec, c0_c1_t_vec, a1b1c1_abc0_t, d0_abc0_t_vec,
            a0_a1_mu_vec, b0_b1_mu_vec, c0_c1_mu_vec, a1b1c1_abc0_mu, d0_abc0_mu_vec
        )
        markov = (1, i, i)
        states[acc] = markov
        probs[acc] = emissions
        acc += 1
        
        # V2 states
        emissions = calc_emissions_double_JC69(
            a0_a1_t_vec, c0_c1_t_vec, b0_b1_t_vec, a1b1c1_abc0_t, d0_abc0_t_vec,
            a0_a1_mu_vec, c0_c1_mu_vec, b0_b1_mu_vec, a1b1c1_abc0_mu, d0_abc0_mu_vec
        )
        new_emissions = {}
        for k in list(emissions.keys()):
            new_emissions[k[0]+k[2]+k[1]+k[3]] = emissions[k]     
        markov = (2, i, i)
        states[acc] = markov
        probs[acc] = new_emissions
        acc += 1
        
        # V3 states
        emissions = calc_emissions_double_JC69(
            b0_b1_t_vec, c0_c1_t_vec, a0_a1_t_vec, a1b1c1_abc0_t, d0_abc0_t_vec,
            b0_b1_mu_vec, c0_c1_mu_vec, a0_a1_mu_vec, a1b1c1_abc0_mu, d0_abc0_mu_vec
        )
        new_emissions = {}
        for k in list(emissions.keys()):
            new_emissions[k[2]+k[0]+k[1]+k[3]] = emissions[k]     
        markov = (3, i, i)
        states[acc] = markov
        probs[acc] = new_emissions
        acc += 1            
    
    # V0 states
    for i in range(n_int_AB):
        for j in range(n_int_ABC):
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
            # d0_abc0_t_vec = [t_A+t_AB+cut_ABC[n_int_ABC-1]+t_upper]+[t_peak+add]
            d0_abc0_t_vec = [t_out]+[add]
            # d0_abc0_mu_vec = [mu_D, mu_ABC]
            d0_abc0_mu_vec = [mu_D, mu_ABC]
            
            emissions = calc_emissions_single_JC69(
                a0_a1_t_vec, b0_b1_t_vec, a1b1_ab0_t, ab0_ab1_t_vec, 
                ab1c1_abc0_t, c0_c1_t_vec, d0_abc0_t_vec,
                a0_a1_mu_vec, b0_b1_mu_vec, a1b1_ab0_mu, ab0_ab1_mu_vec, 
                ab1c1_abc0_mu, c0_c1_mu_vec, d0_abc0_mu_vec
            )
            states[acc] = (0, i, j)
            probs[acc] = emissions
            acc += 1
   
    probs = pd.DataFrame(list(probs))
    probs.insert(0, 'hidden_state', states)
            
    return probs
