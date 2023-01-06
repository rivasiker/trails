import sys
import numpy as np
import pandas as pd
from scipy.linalg import expm
from scipy.stats import truncexpon
from scipy.stats import expon
from scipy.special import comb
import ast
import multiprocessing as mp

from cutpoints_ABC import cutpoints_ABC
from get_ABC_inf_bis import get_ABC_inf_bis
from vanloan_3 import vanloan_3
from get_times import get_times
from get_tab_AB import get_tab_AB
from get_ordered import get_ordered
from vanloan_2 import vanloan_2
from cutpoints_AB import cutpoints_AB
from instant_mat import instant_mat
from vanloan_1 import vanloan_1
from get_ABC import get_ABC
from get_tab_ABC import get_tab_ABC
from get_joint_prob_mat import get_joint_prob_mat
from combine_states import combine_states
from load_trans_mat import load_trans_mat
from trans_mat_num import trans_mat_num

def get_tab_ABC(state_space_ABC, trans_mat_ABC, cut_ABC, pi_ABC, names_tab_AB, n_int_AB):
    """
    This functions returns a table with joint probabilities of
    the states of the HMM after running a three-sequence CTMC
    
    Parameters
    ----------
    state_space_ABC : list of lists of tuples
        States of the whole state space of the three-sequence CTMC
    trans_mat_ABC : numeric numpy matrix
        Transition rate matrix of the three-sequence CTMC
    cut_ABC : list of floats
        Ordered cutpoints of the three-sequence CTMC
    pi_ABC : list of floats
        Starting probabilities after merging a one-sequence and a
        two-sequence CTMCs. 
    names_tab_AB : list of tuples
        List of fates for the starting probabilities, as outputted
        by get_tab_AB().
    n_int_AB : integer
        Number of intervals in the two-sequence CTMC
    """
    
    ###############################
    ### State-space information ###
    ###############################
    
    dct_omegas = {}
    flatten = [list(sum(i, ())) for i in state_space_ABC]    
    for l in [0, 3, 5, 6, 7]:
        for r in [0, 3, 5, 6, 7]:
            if (l in [3, 5, 6, 7]) and (r in [3, 5, 6, 7]):
                dct_omegas['omega_%s%s' % (l, r)] = [i for i in range(203) if (l in flatten[i][::2]) and (r in flatten[i][1::2])]
            elif (l == 0) and (r in [3, 5, 6, 7]):
                dct_omegas['omega_%s%s' % (l, r)] = [i for i in range(203) if (all(x not in [3, 5, 6, 7] for x in flatten[i][::2])) and (r in flatten[i][1::2])]
            elif (l  in [3, 5, 6, 7]) and (r == 0):
                dct_omegas['omega_%s%s' % (l, r)] = [i for i in range(203) if (l in flatten[i][::2]) and (all(x not in [3, 5, 6, 7] for x in flatten[i][1::2]))]
            elif l == r == 0:
                dct_omegas['omega_%s%s' % (l, r)] = [i for i in range(203) if all(x not in [3, 5, 6, 7] for x in flatten[i])]
    for k in dct_omegas:
        globals()[k] = dct_omegas[k]
    omega_tot_ABC = [i for i in range(203)]
    omega_71 = sorted(omega_73+omega_75+omega_76)
    dct_omegas['omega_71'] = omega_71
    omega_17 = sorted(omega_37+omega_57+omega_67)
    dct_omegas['omega_17'] = omega_17
    omega_10 = sorted(omega_30+omega_50+omega_60)
    dct_omegas['omega_10'] = omega_10
    omega_13 = sorted(omega_33+omega_53+omega_63)
    dct_omegas['omega_13'] = omega_13
    omega_15 = sorted(omega_35+omega_55+omega_65)
    dct_omegas['omega_15'] = omega_15
    omega_16 = sorted(omega_36+omega_56+omega_66)
    dct_omegas['omega_16'] = omega_16
    omega_11 = sorted(omega_13+omega_15+omega_16)
    
    dct_num = {3:1, 5:2, 6:3}
    
    # Number of final states
    n_int_ABC = len(cut_ABC)-1
    n_markov_states = n_int_AB*n_int_ABC+n_int_ABC*3+3*comb(n_int_ABC, 2, exact = True)
    
    # Create empty transition probability matrix
    tab=np.empty((n_markov_states**2, 3), dtype=object)
    # Create accumulator for keeping track of the indices for the table
    acc_tot = 0
    
    
    
    ################
    ### V0 -> V0 ###
    ################
   
    # A pair of sites whose fate is to be V0 states is represented as ((0, l, L), (0, r, R)), where
    # l is the index of the interval where the first left coalescent happens, r is the same
    # for the first right coalescent, L is the same for the second left coalescent, and R is the 
    # second right coalescent. Remember that the probability of ((0, l, L) -> (0, r, R)) equals
    # that of ((0, r, R), (0, l, L)).
    for l in range(n_int_AB):
        for r in range(n_int_AB):
            cond = [i == ((0, l),(0, r)) for i in names_tab_AB]
            pi = pi_ABC[cond]
            for L in range(n_int_ABC):
                for R in range(n_int_ABC):
                    if L < R:                        
                        times_ABC = get_times(cut_ABC, [0, L, L+1, R, R+1])
                        omegas_ABC = [omega_tot_ABC, omega_11, omega_71, omega_71, omega_77]
                        p_ABC = get_ABC_inf_bis(trans_mat_ABC, times_ABC, omegas_ABC)
                        tab[acc_tot]   = [(0, l, L), (0, r, R), (pi@p_ABC).sum()]
                        tab[acc_tot+1] = [(0, r, R), (0, l, L), tab[acc_tot][2]]
                        acc_tot += 2
                    elif L == R:
                        times_ABC = get_times(cut_ABC, [0, L, L+1])
                        omegas_ABC = [omega_tot_ABC, omega_11, omega_77]
                        p_ABC = get_ABC_inf_bis(trans_mat_ABC, times_ABC, omegas_ABC)
                        tab[acc_tot] = [(0, l, L), (0, r, R), (pi@p_ABC).sum()]
                        acc_tot += 1
                    else:
                        continue
  
    
    ##############################
    ### V0 -> deep coalescence ###
    ### Deep coalescence -> V0 ###
    ##############################

    
    # A pair of sites where the left site is in V0 and the right site is of deep coalescence
    # is represented as ((0, l, L), (i, r, R)), where l is the index of the interval where the 
    # first left coalescent happens, L is the same for the second left coalescent, r is the 
    # same for the first right coalescent and R is the same for the second right coalescent. The index
    # i can take values from 1 to 4, where 1 to 3 represents deep coalescent and l < L, and 4 
    # represents a multiple merger event where l = L. Remember that the probability of 
    # ((0, l, L) -> (i, r, R)) and that of ((i, r, R) -> (0, l, L)) is the same. Also,
    # ((0, l, L) -> (1, r, R)) = ((0, l, L) -> (2, r, R)) = ((0, l, L) -> (3, r, R)), following ILS.
    for l in range(n_int_AB):
        cond = [i == ((0, l), 'D') for i in names_tab_AB]
        pi = pi_ABC[cond]
        for L in range(n_int_ABC):
            for r in range(n_int_ABC):
                for R in range(r, n_int_ABC):
                    if L < r < R:
                        times_ABC = get_times(cut_ABC, [0, L, L+1, r, r+1, R, R+1])
                        omegas_ABC = [omega_tot_ABC, omega_10, omega_70, omega_70]
                        for i in range(1, 4):
                            if i == 1:
                                omegas_ABC = omegas_ABC+[omega_73, omega_73, omega_77]
                                p_ABC = get_ABC_inf_bis(trans_mat_ABC, times_ABC, omegas_ABC)
                            elif i == 2:
                                omegas_ABC = omegas_ABC+[omega_75, omega_75, omega_77]
                                p_ABC = get_ABC_inf_bis(trans_mat_ABC, times_ABC, omegas_ABC)
                            elif i == 3:
                                omegas_ABC = omegas_ABC+[omega_76, omega_76, omega_77]
                                p_ABC = get_ABC_inf_bis(trans_mat_ABC, times_ABC, omegas_ABC)
                            tab[acc_tot]   = [(0, l, L), (i, r, R), (pi@p_ABC).sum()]
                            tab[acc_tot+1] = [(i, r, R), (0, l, L), tab[acc_tot][2]]
                            acc_tot += 2
                    elif L == r < R:
                        times_ABC = get_times(cut_ABC, [0, L, L+1, R, R+1])
                        omegas_ABC = [omega_tot_ABC, omega_10]
                        for i in range(1, 4):
                            if i == 1:
                                omegas_ABC = omegas_ABC+[omega_73, omega_73, omega_77]
                                p_ABC = get_ABC_inf_bis(trans_mat_ABC, times_ABC, omegas_ABC)
                            elif i == 2:
                                omegas_ABC = omegas_ABC+[omega_75, omega_75, omega_77]
                                p_ABC = get_ABC_inf_bis(trans_mat_ABC, times_ABC, omegas_ABC)
                            elif i == 3:
                                omegas_ABC = omegas_ABC+[omega_76, omega_76, omega_77]
                                p_ABC = get_ABC_inf_bis(trans_mat_ABC, times_ABC, omegas_ABC)
                            tab[acc_tot]   = [(0, l, L), (i, r, R), (pi@p_ABC).sum()]
                            tab[acc_tot+1] = [(i, r, R), (0, l, L), tab[acc_tot][2]]
                            acc_tot += 2
                    elif r < L < R:
                        times_ABC = get_times(cut_ABC, [0, r, r+1, L, L+1, R, R+1])
                        omegas_ABC = [omega_tot_ABC, omega_10]
                        for i in range(1, 4):
                            if i == 1:
                                omegas_ABC = omegas_ABC+[omega_13, omega_13, omega_73, omega_73, omega_77]
                                p_ABC = get_ABC_inf_bis(trans_mat_ABC, times_ABC, omegas_ABC)
                            elif i == 2:
                                omegas_ABC = omegas_ABC+[omega_15, omega_15, omega_75, omega_75, omega_77]
                                p_ABC = get_ABC_inf_bis(trans_mat_ABC, times_ABC, omegas_ABC)
                            elif i == 3:
                                omegas_ABC = omegas_ABC+[omega_16, omega_16, omega_76, omega_76, omega_77]
                                p_ABC = get_ABC_inf_bis(trans_mat_ABC, times_ABC, omegas_ABC)
                            tab[acc_tot]   = [(0, l, L), (i, r, R), (pi@p_ABC).sum()]
                            tab[acc_tot+1] = [(i, r, R), (0, l, L), tab[acc_tot][2]]
                            acc_tot += 2
                    elif r < L == R:
                        times_ABC = get_times(cut_ABC, [0, r, r+1, L, L+1])
                        omegas_ABC = [omega_tot_ABC, omega_10]
                        for i in range(1, 4):
                            if i == 1:
                                omegas_ABC = omegas_ABC+[omega_13, omega_13, omega_77]
                                p_ABC = get_ABC_inf_bis(trans_mat_ABC, times_ABC, omegas_ABC)
                            elif i == 2:
                                omegas_ABC = omegas_ABC+[omega_15, omega_15, omega_77]
                                p_ABC = get_ABC_inf_bis(trans_mat_ABC, times_ABC, omegas_ABC)
                            elif i == 3:
                                omegas_ABC = omegas_ABC+[omega_16, omega_16, omega_77]
                                p_ABC = get_ABC_inf_bis(trans_mat_ABC, times_ABC, omegas_ABC)
                            tab[acc_tot]   = [(0, l, L), (i, r, R), (pi@p_ABC).sum()]
                            tab[acc_tot+1] = [(i, r, R), (0, l, L), tab[acc_tot][2]]
                            acc_tot += 2
                    elif r < R < L:
                        times_ABC = get_times(cut_ABC, [0, r, r+1, R, R+1, L, L+1])
                        omegas_ABC = [omega_tot_ABC, omega_10]
                        for i in range(1, 4):
                            if i == 1:
                                omegas_ABC = omegas_ABC+[omega_13, omega_13, omega_17, omega_17, omega_77]
                                p_ABC = get_ABC_inf_bis(trans_mat_ABC, times_ABC, omegas_ABC)
                            elif i == 2:
                                omegas_ABC = omegas_ABC+[omega_15, omega_15, omega_17, omega_17, omega_77]
                                p_ABC = get_ABC_inf_bis(trans_mat_ABC, times_ABC, omegas_ABC)
                            elif i == 3:
                                omegas_ABC = omegas_ABC+[omega_16, omega_16, omega_17, omega_17, omega_77]
                                p_ABC = get_ABC_inf_bis(trans_mat_ABC, times_ABC, omegas_ABC)
                            tab[acc_tot]   = [(0, l, L), (i, r, R), (pi@p_ABC).sum()]
                            tab[acc_tot+1] = [(i, r, R), (0, l, L), tab[acc_tot][2]]
                            acc_tot += 2
                    elif L < r == R:
                        omegas_ABC = [omega_tot_ABC, omega_10, omega_70, omega_70]
                        times_ABC = get_times(cut_ABC, [0, L, L+1, r])
                        p_ABC = get_ABC_inf_bis(trans_mat_ABC, times_ABC, omegas_ABC)
                        for i in [3, 5, 6]:
                            # 70 -> 7i
                            if cut_ABC[r+1] != np.inf:
                                res = vanloan_1(
                                    trans_mat_ABC, (omega_70, dct_omegas['omega_7%s'%i]),
                                    omega_70, omega_77, cut_ABC[r+1]-cut_ABC[r])
                                ii = dct_num[i]
                                tab[acc_tot]   = [(0, l, L), (ii, r, R), (pi@p_ABC@res).sum()]
                                tab[acc_tot+1] = [(ii, r, R), (0, l, L), tab[acc_tot][2]]
                                acc_tot += 2
                            else:
                                A_mat = instant_mat(omega_70, dct_omegas['omega_7%s'%i], trans_mat_ABC)
                                res = (-np.linalg.inv(trans_mat_ABC[:-2,:-2])@(A_mat[:-2,:-2]))[omega_70][:,dct_omegas['omega_7%s'%i]]
                                ii = dct_num[i]
                                tab[acc_tot]   = [(0, l, L), (ii, r, R), (pi@p_ABC@res).sum()]
                                tab[acc_tot+1] = [(ii, r, R), (0, l, L), tab[acc_tot][2]]
                                acc_tot += 2
                    elif L == r == R:
                        omegas_ABC = [omega_tot_ABC, omega_10, omega_77]
                        times_ABC = get_times(cut_ABC, [0, L, L+1])
                        p_ABC = get_ABC_inf_bis(trans_mat_ABC, times_ABC, omegas_ABC)
                        if cut_ABC[r+1] == np.inf:
                            for i in [3, 5, 6]:
                                # 10 -> 1i
                                A_mat = instant_mat(omega_10, dct_omegas['omega_1%s'%i], trans_mat_ABC)
                                res_1 = (-np.linalg.inv(trans_mat_ABC[:-2,:-2])@(A_mat[:-2,:-2]))[omega_10][:,dct_omegas['omega_1%s'%i]]
                                # 10 -> 7i
                                A_mat = instant_mat(omega_10, dct_omegas['omega_7%s'%i], trans_mat_ABC)
                                res_2 = (-np.linalg.inv(trans_mat_ABC[:-2,:-2])@(A_mat[:-2,:-2]))[omega_10][:,dct_omegas['omega_7%s'%i]]
                                # 10 -> 70 -> 7i
                                A_mat_1 = instant_mat(omega_10, omega_70, trans_mat_ABC)
                                A_mat_2 = instant_mat(omega_70, dct_omegas['omega_7%s'%i], trans_mat_ABC)
                                C_mat_upper =  np.concatenate((trans_mat_ABC[:-2,:-2], A_mat_1[:-2,:-2]), axis = 1)
                                C_mat_lower = np.concatenate((np.zeros((201,201)), trans_mat_ABC[:-2,:-2]), axis = 1)
                                C_mat = np.concatenate((C_mat_upper, C_mat_lower), axis = 0)
                                res_3 = ((-np.linalg.inv(C_mat)[0:201,-201:])@(A_mat_2[:-2,:-2]))[omega_10][:,dct_omegas['omega_7%s'%i]]
                                ii = dct_num[i]
                                tab[acc_tot]   = [(0, l, L), (ii, r, R), (pi@p_ABC@res_1).sum()+(pi@p_ABC@sum([res_2, res_3])).sum()]
                                tab[acc_tot+1] = [(ii, r, R), (0, l, L), tab[acc_tot][2]]
                                acc_tot += 2
                        else:
                            omegas_ABC = [omega_tot_ABC, omega_10]
                            times_ABC = get_times(cut_ABC, [0, L])
                            p_ABC = get_ABC_inf_bis(trans_mat_ABC, times_ABC, omegas_ABC)
                            for i in [3, 5, 6]:
                                omega_lst = ['10', '1%s'%i, '17', '70', '7%s'%i, '77']
                                iter_lst = []
                                for y in range(1, len(omega_lst)):
                                    for z in range(y+1, len(omega_lst)):
                                        tup = (dct_omegas['omega_%s'%(omega_lst[0],)], 
                                               dct_omegas['omega_%s'%(omega_lst[y],)],
                                               dct_omegas['omega_%s'%(omega_lst[z],)])
                                        iter_lst.append(tup)
                                pool = mp.Pool(mp.cpu_count())
                                res_tot = []
                                res_tot = pool.starmap_async(
                                    vanloan_2, 
                                    [(trans_mat_ABC, tup, omega_10,
                                      omega_77, cut_ABC[r+1]-cut_ABC[r]) for tup in iter_lst]
                                ).get()
                                pool.close()
                                res_tot = (pi@p_ABC@sum(res_tot)).sum()
                                ii = dct_num[i]
                                tab[acc_tot]   = [(0, l, L), (ii, r, R), res_tot]
                                tab[acc_tot+1] = [(ii, r, R), (0, l, L), res_tot]
                                acc_tot += 2
                    elif r == R < L:
                        omegas_ABC = [omega_tot_ABC, omega_10]
                        times_ABC = get_times(cut_ABC, [0, r])
                        p_ABC_start = get_ABC_inf_bis(trans_mat_ABC, times_ABC, omegas_ABC)
                        omegas_ABC = [omega_17, omega_17, omega_77]
                        times_ABC = get_times(cut_ABC, [r+1, L, L+1])
                        p_ABC_end = get_ABC_inf_bis(trans_mat_ABC, times_ABC, omegas_ABC)
                        for i in [3, 5, 6]:
                            # 10 -> 1i
                            A_mat = instant_mat(omega_10, dct_omegas['omega_1%s'%i], trans_mat_ABC)
                            C_mat_upper = np.concatenate((trans_mat_ABC, A_mat), axis = 1)
                            C_mat_lower = np.concatenate((np.zeros((203,203)), trans_mat_ABC), axis = 1)
                            C_mat = np.concatenate((C_mat_upper, C_mat_lower), axis = 0)
                            res = (expm(C_mat*(cut_ABC[r+1]-cut_ABC[r]))[:203,-203:])[omega_10][:,omega_17]
                            ii = dct_num[i]
                            tab[acc_tot]   = [(0, l, L), (ii, r, R), (pi@p_ABC_start@res@p_ABC_end).sum()]
                            tab[acc_tot+1] = [(ii, r, R), (0, l, L), tab[acc_tot][2]]
                            acc_tot += 2
                    else:
                        continue
                        
    ############################################
    ### Deep coalescence -> deep coalescence ###
    ############################################
    
    # A pair of sites where both the left and the right site are of deep coalescence is 
    # represented as ((i, l, L), (j, r, R)), where l is the index of the interval where the 
    # first left coalescent happens, L is the same for the second left coalescent, r is the 
    # same for the first right coalescent and R is the same for the second right coalescent.
    # The indices i and j can take values from 1 to 4, where 1 to 3 represents deep coalescent 
    # and l < L, and 4  represents a multiple merger event where l = L. Remember that the probability 
    # of ((i, l, L) -> (j, r, R)) and that of ((j, r, R) -> (i, l, L)) is the same. Also,
    # ((i, l, L) -> (1, r, R)) = ((i, l, L) -> (2, r, R)) = ((i, l, L) -> (3, r, R)), following ILS.
    acc = 0
    cond = [i == ('D','D') for i in names_tab_AB]
    pi = pi_ABC[cond]
    for l in range(n_int_ABC):
        for L in range(l, n_int_ABC):
            for r in range(n_int_ABC):
                for R in range(r, n_int_ABC): 
                    if l < L < r < R:
                        times_ABC = get_times(cut_ABC, [0, l, l+1, L, L+1, r, r+1, R, R+1])
                        for i in [3, 5, 6]:
                            for j in [3, 5, 6]:
                                # 00 -> i0 -> 70 -> 7j -> 77
                                omegas_ABC = [omega_tot_ABC, omega_00, 
                                              dct_omegas['omega_%s0'%i], dct_omegas['omega_%s0'%i], 
                                              omega_70, omega_70, 
                                              dct_omegas['omega_7%s'%j], dct_omegas['omega_7%s'%j], 
                                              omega_77]
                                p_ABC = get_ABC_inf_bis(trans_mat_ABC, times_ABC, omegas_ABC)
                                ii = dct_num[i]
                                jj = dct_num[j]
                                tab[acc_tot]   = [(ii, l, L), (jj, r, R), (pi@p_ABC).sum()]
                                tab[acc_tot+1] = [(jj, r, R), (ii, l, L), tab[acc_tot][2]]
                                acc_tot += 2
                    elif l < L == r < R:
                        times_ABC = get_times(cut_ABC, [0, l, l+1, L, L+1, R, R+1])
                        for i in [3, 5, 6]:
                            for j in [3, 5, 6]:
                                # 00 -> i0 -> 7j -> 77
                                omegas_ABC = [omega_tot_ABC, omega_00, 
                                              dct_omegas['omega_%s0'%i], dct_omegas['omega_%s0'%i], 
                                              dct_omegas['omega_7%s'%j], dct_omegas['omega_7%s'%j], 
                                              omega_77]
                                p_ABC = get_ABC_inf_bis(trans_mat_ABC, times_ABC, omegas_ABC)
                                ii = dct_num[i]
                                jj = dct_num[j]
                                tab[acc_tot]   = [(ii, l, L), (jj, r, R), (pi@p_ABC).sum()]
                                tab[acc_tot+1] = [(jj, r, R), (ii, l, L), tab[acc_tot][2]]
                                acc_tot += 2
                    elif l == r < L < R:
                        times_ABC = get_times(cut_ABC, [0, l, l+1, L, L+1, R, R+1])
                        for i in [3, 5, 6]:
                            for j in [3, 5, 6]:
                                # 00 -> ij -> 7j -> 77
                                omegas_ABC = [omega_tot_ABC, omega_00, 
                                              dct_omegas['omega_%s%s'%(i,j)], dct_omegas['omega_%s%s'%(i,j)], 
                                              dct_omegas['omega_7%s'%j], dct_omegas['omega_7%s'%j], 
                                              omega_77]
                                p_ABC = get_ABC_inf_bis(trans_mat_ABC, times_ABC, omegas_ABC)
                                ii = dct_num[i]
                                jj = dct_num[j]
                                tab[acc_tot]   = [(ii, l, L), (jj, r, R), (pi@p_ABC).sum()]
                                tab[acc_tot+1] = [(jj, r, R), (ii, l, L), tab[acc_tot][2]]
                                acc_tot += 2
                    elif l < r < L < R:
                        times_ABC = get_times(cut_ABC, [0, l, l+1, r, r+1, L, L+1, R, R+1])
                        for i in [3, 5, 6]:
                            for j in [3, 5, 6]:
                                # 00 -> i0 -> ij -> 7j -> 77
                                omegas_ABC = [omega_tot_ABC, omega_00, 
                                              dct_omegas['omega_%s0'%i], dct_omegas['omega_%s0'%i], 
                                              dct_omegas['omega_%s%s'%(i,j)], dct_omegas['omega_%s%s'%(i,j)],
                                              dct_omegas['omega_7%s'%j], dct_omegas['omega_7%s'%j], 
                                              omega_77]
                                p_ABC = get_ABC_inf_bis(trans_mat_ABC, times_ABC, omegas_ABC)
                                ii = dct_num[i]
                                jj = dct_num[j]
                                tab[acc_tot]   = [(ii, l, L), (jj, r, R), (pi@p_ABC).sum()]
                                tab[acc_tot+1] = [(jj, r, R), (ii, l, L), tab[acc_tot][2]]
                                acc_tot += 2
                    elif r < l < L < R:
                        times_ABC = get_times(cut_ABC, [0, r, r+1, l, l+1, L, L+1, R, R+1])
                        for i in [3, 5, 6]:
                            for j in [3, 5, 6]:
                                # 00 -> 0j -> ij -> 7j -> 77
                                omegas_ABC = [omega_tot_ABC, omega_00, 
                                              dct_omegas['omega_0%s'%j], dct_omegas['omega_0%s'%j], 
                                              dct_omegas['omega_%s%s'%(i,j)], dct_omegas['omega_%s%s'%(i,j)],
                                              dct_omegas['omega_7%s'%j], dct_omegas['omega_7%s'%j], 
                                              omega_77]
                                p_ABC = get_ABC_inf_bis(trans_mat_ABC, times_ABC, omegas_ABC)
                                ii = dct_num[i]
                                jj = dct_num[j]
                                tab[acc_tot]   = [(ii, l, L), (jj, r, R), (pi@p_ABC).sum()]
                                tab[acc_tot+1] = [(jj, r, R), (ii, l, L), tab[acc_tot][2]]
                                acc_tot += 2
                    elif l == r < L == R:
                        times_ABC = get_times(cut_ABC, [0, l, l+1, L, L+1])
                        for i in [3, 5, 6]:
                            for j in [3, 5, 6]:
                                # 00 -> ij -> 77
                                omegas_ABC = [omega_tot_ABC, omega_00, 
                                              dct_omegas['omega_%s%s'%(i,j)], dct_omegas['omega_%s%s'%(i,j)],
                                              omega_77]
                                p_ABC = get_ABC_inf_bis(trans_mat_ABC, times_ABC, omegas_ABC)
                                
                                ii = dct_num[i]
                                jj = dct_num[j]                                
                                tab[acc_tot]   = [(ii, l, L), (jj, r, R), (pi@p_ABC).sum()]
                                acc_tot += 1
                    elif l < r < L == R:
                        times_ABC = get_times(cut_ABC, [0, l, l+1, r, r+1, L, L+1])
                        for i in [3, 5, 6]:
                            for j in [3, 5, 6]:
                                # 00 -> i0 -> ij -> 77
                                omegas_ABC = [omega_tot_ABC, omega_00, 
                                              dct_omegas['omega_%s0'%i], dct_omegas['omega_%s0'%i], 
                                              dct_omegas['omega_%s%s'%(i,j)], dct_omegas['omega_%s%s'%(i,j)],
                                              omega_77]
                                p_ABC = get_ABC_inf_bis(trans_mat_ABC, times_ABC, omegas_ABC)
                                ii = dct_num[i]
                                jj = dct_num[j]
                                tab[acc_tot]   = [(ii, l, L), (jj, r, R), (pi@p_ABC).sum()]
                                tab[acc_tot+1] = [(jj, r, R), (ii, l, L), tab[acc_tot][2]]
                                acc_tot += 2
                    elif l == r == L == R:
                        times_ABC = get_times(cut_ABC, [0, l, l+1])
                        omegas_ABC = [omega_tot_ABC, omega_00, omega_77]
                        p_ABC = get_ABC_inf_bis(trans_mat_ABC, times_ABC, omegas_ABC)
                        times_ABC = get_times(cut_ABC, [0, l])
                        omegas_ABC = [omega_tot_ABC, omega_00]
                        start = get_ABC_inf_bis(trans_mat_ABC, times_ABC, omegas_ABC)
                        for i in [3, 5, 6]:
                            for j in [3, 5, 6]:
                                res_tot = 0
                                if cut_ABC[r+1] == np.inf:
                                    res_tot = 0
                                    # 00 -> ij
                                    A_mat = instant_mat(omega_00, dct_omegas['omega_%s%s'%(i,j)], trans_mat_ABC)
                                    res_1 = (-np.linalg.inv(trans_mat_ABC[:-2,:-2])@(A_mat[:-2,:-2]))[omega_00][:,dct_omegas['omega_%s%s'%(i,j)]]
                                    # 00 -> i0 -> ij
                                    A_mat_1 = instant_mat(omega_00, dct_omegas['omega_%s0'%i], trans_mat_ABC)
                                    A_mat_2 = instant_mat(dct_omegas['omega_%s0'%i], dct_omegas['omega_%s%s'%(i,j)], trans_mat_ABC)
                                    C_mat_upper =  np.concatenate((trans_mat_ABC[:-2,:-2], A_mat_1[:-2,:-2]), axis = 1)
                                    C_mat_lower = np.concatenate((np.zeros((201,201)), trans_mat_ABC[:-2,:-2]), axis = 1)
                                    C_mat = np.concatenate((C_mat_upper, C_mat_lower), axis = 0)
                                    res_2 = ((-np.linalg.inv(C_mat)[0:201,-201:])@(A_mat_2[:-2,:-2]))[omega_00][:,dct_omegas['omega_%s%s'%(i,j)]]
                                    # 00 -> 0j -> ij
                                    A_mat_1 = instant_mat(omega_00, dct_omegas['omega_0%s'%j], trans_mat_ABC)
                                    A_mat_2 = instant_mat(dct_omegas['omega_0%s'%j], dct_omegas['omega_%s%s'%(i,j)], trans_mat_ABC)
                                    C_mat_upper =  np.concatenate((trans_mat_ABC[:-2,:-2], A_mat_1[:-2,:-2]), axis = 1)
                                    C_mat_lower = np.concatenate((np.zeros((201,201)), trans_mat_ABC[:-2,:-2]), axis = 1)
                                    C_mat = np.concatenate((C_mat_upper, C_mat_lower), axis = 0)
                                    res_3 = ((-np.linalg.inv(C_mat)[0:201,-201:])@(A_mat_2[:-2,:-2]))[omega_00][:,dct_omegas['omega_%s%s'%(i,j)]]
                                    # 00 -> 0j -> 07 -> i7
                                    A_mat_1 = instant_mat(omega_00, dct_omegas['omega_0%s'%j], trans_mat_ABC)
                                    A_mat_2 = instant_mat(dct_omegas['omega_0%s'%j], omega_07, trans_mat_ABC)
                                    A_mat_3 = instant_mat(omega_07, dct_omegas['omega_%s7'%i], trans_mat_ABC)
                                    C_mat_upper  =  np.concatenate((trans_mat_ABC[:-2,:-2], A_mat_1[:-2,:-2], np.zeros((201,201))), axis = 1)
                                    C_mat_middle =  np.concatenate((np.zeros((201,201)), trans_mat_ABC[:-2,:-2], A_mat_2[:-2,:-2]), axis = 1)
                                    C_mat_lower  =  np.concatenate((np.zeros((201,201)),np.zeros((201,201)), trans_mat_ABC[:-2,:-2]), axis = 1)
                                    C_mat = np.concatenate((C_mat_upper, C_mat_middle, C_mat_lower), axis = 0)
                                    res_4 = ((-np.linalg.inv(C_mat)[0:201,-201:])@(A_mat_3[:-2,:-2]))[omega_00][:,dct_omegas['omega_%s7'%i]]
                                    # 00 -> i0 -> 70 -> 7j
                                    A_mat_1 = instant_mat(omega_00, dct_omegas['omega_%s0'%i], trans_mat_ABC)
                                    A_mat_2 = instant_mat(dct_omegas['omega_%s0'%i], omega_70, trans_mat_ABC)
                                    A_mat_3 = instant_mat(omega_70, dct_omegas['omega_7%s'%j], trans_mat_ABC)
                                    C_mat_upper  =  np.concatenate((trans_mat_ABC[:-2,:-2], A_mat_1[:-2,:-2], np.zeros((201,201))), axis = 1)
                                    C_mat_middle =  np.concatenate((np.zeros((201,201)), trans_mat_ABC[:-2,:-2], A_mat_2[:-2,:-2]), axis = 1)
                                    C_mat_lower  =  np.concatenate((np.zeros((201,201)), np.zeros((201,201)), trans_mat_ABC[:-2,:-2]), axis = 1)
                                    C_mat = np.concatenate((C_mat_upper, C_mat_middle, C_mat_lower), axis = 0)
                                    res_5 = ((-np.linalg.inv(C_mat)[0:201,-201:])@(A_mat_3[:-2,:-2]))[omega_00][:,dct_omegas['omega_7%s'%j]]                               
                                    # 00 -> 0j -> j7
                                    A_mat_1 = instant_mat(omega_00, dct_omegas['omega_0%s'%j], trans_mat_ABC)
                                    A_mat_2 = instant_mat(dct_omegas['omega_0%s'%j], dct_omegas['omega_%s7'%i], trans_mat_ABC)
                                    C_mat_upper =  np.concatenate((trans_mat_ABC[:-2,:-2], A_mat_1[:-2,:-2]), axis = 1)
                                    C_mat_lower = np.concatenate((np.zeros((201,201)), trans_mat_ABC[:-2,:-2]), axis = 1)
                                    C_mat = np.concatenate((C_mat_upper, C_mat_lower), axis = 0)
                                    res_6 = ((-np.linalg.inv(C_mat)[0:201,-201:])@(A_mat_2[:-2,:-2]))[omega_00][:,dct_omegas['omega_%s7'%i]]
                                    # 00 -> i0 -> 7j
                                    A_mat_1 = instant_mat(omega_00, dct_omegas['omega_%s0'%i], trans_mat_ABC)
                                    A_mat_2 = instant_mat(dct_omegas['omega_%s0'%i], dct_omegas['omega_7%s'%j], trans_mat_ABC)
                                    C_mat_upper =  np.concatenate((trans_mat_ABC[:-2,:-2], A_mat_1[:-2,:-2]), axis = 1)
                                    C_mat_lower = np.concatenate((np.zeros((201,201)), trans_mat_ABC[:-2,:-2]), axis = 1)
                                    C_mat = np.concatenate((C_mat_upper, C_mat_lower), axis = 0)
                                    res_7 = ((-np.linalg.inv(C_mat)[0:201,-201:])@(A_mat_2[:-2,:-2]))[omega_00][:,dct_omegas['omega_7%s'%j]]
                                    # Sum results
                                    res_tot += (pi@start@sum([res_1, res_2, res_3])).sum()
                                    res_tot += (pi@start@sum([res_4, res_6])).sum()
                                    res_tot += (pi@start@sum([res_5, res_7])).sum()
                                    ii = dct_num[i]
                                    jj = dct_num[j]
                                    tab[acc_tot] = [(ii, l, L), (jj, r, R), res_tot]
                                    acc_tot += 1
                                else:
                                    iter_lst = []
                                    for y in ['%s0'%i, '0%s'%j]:
                                        for z in ['%s7'%i, '7%s'%j]:
                                            tup = (omega_00, 
                                                   dct_omegas['omega_%s'%(y,)],
                                                   dct_omegas['omega_%s'%(z,)],
                                                   dct_omegas['omega_77'])
                                            iter_lst.append(tup)
                                    for y in ['%s%s'%(i,j)]:
                                        for z in ['%s7'%i, '7%s'%j]:
                                            tup = (omega_00, 
                                               dct_omegas['omega_%s'%(y,)],
                                               dct_omegas['omega_%s'%(z,)],
                                               dct_omegas['omega_77'])
                                            iter_lst.append(tup)
                                    for y in ['%s0'%i, '0%s'%j]:
                                        for z in ['%s%s'%(i,j), '70', '07']:
                                            if (int(y[0])-int(z[0]))==-7 or (int(y[1])-int(z[1]))==-7:
                                                continue
                                            for v in ['%s7'%i, '7%s'%j, '77']:
                                                if (int(z[0]) > int(v[0])) or (int(z[1]) > int(v[1])):
                                                    continue
                                                if (int(z[0])-int(v[0]))==-7 or (int(z[1])-int(v[1]))==-7:
                                                    continue
                                                tup = (omega_00, 
                                                       dct_omegas['omega_%s'%(y,)],
                                                       dct_omegas['omega_%s'%(z,)],
                                                       dct_omegas['omega_%s'%(v,)])
                                                iter_lst.append(tup)
                                    pool = mp.Pool(mp.cpu_count())
                                    res_iter = []
                                    res_iter = pool.starmap_async(
                                        vanloan_3, 
                                        [(trans_mat_ABC, tup, omega_00, omega_77,
                                          cut_ABC[r+1]-cut_ABC[r]) for tup in iter_lst]
                                    ).get()
                                    pool.close()
                                    res_tot += (pi@start@sum(res_iter)).sum()
                                    res_test = vanloan_2(
                                        trans_mat_ABC, 
                                        (omega_00, dct_omegas['omega_%s%s'%(i,j)], dct_omegas['omega_77']),
                                        omega_00, omega_77,
                                        cut_ABC[r+1]-cut_ABC[r]
                                    )
                                    res_tot += (pi@start@res_test).sum()
                                    ii = dct_num[i]
                                    jj = dct_num[j]
                                    tab[acc_tot] = [(ii, l, L), (jj, r, R), res_tot]
                                    acc_tot += 1
                    elif l == L < r == R:
                        times_ABC = get_times(cut_ABC, [0, l])
                        omegas_ABC = [omega_tot_ABC, omega_00]
                        start = get_ABC_inf_bis(trans_mat_ABC, times_ABC, omegas_ABC)
                        times_ABC = get_times(cut_ABC, [l+1, r])
                        omegas_ABC = [omega_70, omega_70]
                        end = get_ABC_inf_bis(trans_mat_ABC, times_ABC, omegas_ABC)
                        for i in [3, 5, 6]:
                            res_1 = vanloan_1(
                                trans_mat_ABC, 
                                (omega_00, dct_omegas['omega_%s0'%i]),
                                omega_00, omega_70, cut_ABC[l+1]-cut_ABC[l])
                            for j in [3, 5, 6]:
                                # 00 -> 0i -> 70 -> 7j
                                if cut_ABC[r+1] == np.inf:
                                    A_mat = instant_mat(omega_70, dct_omegas['omega_7%s'%j], trans_mat_ABC)
                                    res_2 = (-np.linalg.inv(trans_mat_ABC[:-2,:-2])@(A_mat[:-2,:-2]))[omega_70][:,dct_omegas['omega_7%s'%j]]
                                    ii = dct_num[i]
                                    jj = dct_num[j]
                                    tab[acc_tot] = [(ii, l, L), (jj, r, R), (pi@start@res_1@end@res_2).sum()]
                                    acc_tot += 1
                                    tab[acc_tot] = [(jj, r, R), (ii, l, L), tab[acc_tot-1][2]]
                                    acc_tot += 1
                                else:
                                    res_2 = vanloan_1(
                                        trans_mat_ABC,
                                        (omega_70, dct_omegas['omega_7%s'%j]),
                                        omega_70, omega_77, cut_ABC[r+1]-cut_ABC[r])
                                    ii = dct_num[i]
                                    jj = dct_num[j]
                                    tab[acc_tot] = [(ii, l, L), (jj, r, R), (pi@start@res_1@end@res_2).sum()]
                                    acc_tot += 1
                                    tab[acc_tot] = [(jj, r, R), (ii, l, L), tab[acc_tot-1][2]]
                                    acc_tot += 1
                    elif l == L < r < R:
                        for j in [3, 5, 6]:
                            times_ABC = get_times(cut_ABC, [0, l])
                            omegas_ABC = [omega_tot_ABC, omega_00]
                            start = get_ABC_inf_bis(trans_mat_ABC, times_ABC, omegas_ABC)
                            times_ABC = get_times(cut_ABC, [l+1, r, r+1, R, R+1])
                            omegas_ABC = [omega_70, omega_70,
                                          dct_omegas['omega_7%s'%j], dct_omegas['omega_7%s'%j], 
                                          omega_77]
                            end = get_ABC_inf_bis(trans_mat_ABC, times_ABC, omegas_ABC)
                            for i in [3, 5, 6]:
                                # 00 -> i0 -> 70 -> 7j
                                res = vanloan_1(
                                        trans_mat_ABC,
                                        (omega_00, dct_omegas['omega_%s0'%i]),
                                        omega_00, omega_70,
                                        cut_ABC[l+1]-cut_ABC[l])
                                ii = dct_num[i]
                                jj = dct_num[j]
                                tab[acc_tot] = [(ii, l, L), (jj, r, R), (pi@start@res@end).sum()]
                                acc_tot += 1
                                tab[acc_tot] = [(jj, r, R), (ii, l, L), tab[acc_tot-1][2]]
                                acc_tot += 1
                    elif l == L == r < R:
                        for j in [3, 5, 6]:
                            for i in [3, 5, 6]:
                                times_ABC = get_times(cut_ABC, [0, l])
                                omegas_ABC = [omega_tot_ABC, omega_00]
                                start = get_ABC_inf_bis(trans_mat_ABC, times_ABC, omegas_ABC)
                                times_ABC = get_times(cut_ABC, [l+1, R, R+1])
                                omegas_ABC = [dct_omegas['omega_7%s'%j], dct_omegas['omega_7%s'%j],
                                              omega_77]
                                end = get_ABC_inf_bis(trans_mat_ABC, times_ABC, omegas_ABC)
                                omega_lst = ['00', '%s0'%i, '0%s'%j, '%s%s'%(i,j), '70', '7%s'%j]
                                iter_lst = []
                                for y in range(1, len(omega_lst)):
                                    for z in range(y+1, len(omega_lst)):
                                        tup = (dct_omegas['omega_%s'%(omega_lst[0],)], 
                                               dct_omegas['omega_%s'%(omega_lst[y],)],
                                               dct_omegas['omega_%s'%(omega_lst[z],)])
                                        iter_lst.append(tup)
                                pool = mp.Pool(mp.cpu_count())
                                res_tot = []
                                res_tot = pool.starmap_async(
                                    vanloan_2, 
                                    [(trans_mat_ABC, tup, omega_00, dct_omegas['omega_7%s'%j],
                                      cut_ABC[r+1]-cut_ABC[r]) for tup in iter_lst]
                                ).get()
                                pool.close()
                                res_tot = (pi@start@sum(res_tot)@end).sum()
                                ii = dct_num[i]
                                jj = dct_num[j]
                                tab[acc_tot] = [(ii, l, L), (jj, r, R), res_tot]
                                acc_tot += 1
                                tab[acc_tot] = [(jj, r, R), (ii, l, L), res_tot]
                                acc_tot += 1
                    elif l < L == r == R:
                        for i in [3, 5, 6]:
                            times_ABC = get_times(cut_ABC, [0, l, l+1, L])
                            omegas_ABC = [omega_tot_ABC, omega_00, 
                                      dct_omegas['omega_%s0'%i], dct_omegas['omega_%s0'%i]]
                            p_ABC = get_ABC_inf_bis(trans_mat_ABC, times_ABC, omegas_ABC)
                            for j in [3, 5, 6]:
                                if cut_ABC[L+1] == np.inf:
                                    # 00 -> i0 -> 1j
                                    A_mat = instant_mat(dct_omegas['omega_%s0'%i], dct_omegas['omega_1%s'%j], trans_mat_ABC)
                                    res_1 = (-np.linalg.inv(trans_mat_ABC[:-2,:-2])@(A_mat[:-2,:-2]))[dct_omegas['omega_%s0'%i]][:,dct_omegas['omega_1%s'%j]]
                                    # 00 -> i0 -> 7j
                                    A_mat = instant_mat(dct_omegas['omega_%s0'%i], dct_omegas['omega_7%s'%j], trans_mat_ABC)
                                    res_2 = (-np.linalg.inv(trans_mat_ABC[:-2,:-2])@(A_mat[:-2,:-2]))[dct_omegas['omega_%s0'%i]][:,dct_omegas['omega_7%s'%j]]
                                    # 00 -> i0 -> 70 -> 7j
                                    A_mat_1 = instant_mat(dct_omegas['omega_%s0'%i], omega_70, trans_mat_ABC)
                                    A_mat_2 = instant_mat(omega_70, dct_omegas['omega_7%s'%j], trans_mat_ABC)
                                    C_mat_upper =  np.concatenate((trans_mat_ABC[:-2,:-2], A_mat_1[:-2,:-2]), axis = 1)
                                    C_mat_lower = np.concatenate((np.zeros((201,201)), trans_mat_ABC[:-2,:-2]), axis = 1)
                                    C_mat = np.concatenate((C_mat_upper, C_mat_lower), axis = 0)
                                    res_3 = ((-np.linalg.inv(C_mat)[0:201,-201:])@(A_mat_2[:-2,:-2]))[dct_omegas['omega_%s0'%i]][:,dct_omegas['omega_7%s'%j]]
                                    # Sum of results
                                    ii = dct_num[i]
                                    jj = dct_num[j]
                                    tab[acc_tot] = [(ii, l, L), (jj, r, R), (pi@p_ABC@res_1).sum()+(pi@p_ABC@sum([res_2, res_3])).sum()]
                                    acc_tot += 1
                                    tab[acc_tot] = [(jj, r, R), (ii, l, L), tab[acc_tot-1][2]]
                                    acc_tot += 1
                                else:
                                    omega_lst = ['%s0'%i, '%s%s'%(i,j), '%s7'%i, '70', '7%s'%j, '77']
                                    iter_lst = []
                                    for y in range(1, len(omega_lst)):
                                        for z in range(y+1, len(omega_lst)):
                                            tup = (dct_omegas['omega_%s'%(omega_lst[0],)], 
                                                   dct_omegas['omega_%s'%(omega_lst[y],)],
                                                   dct_omegas['omega_%s'%(omega_lst[z],)])
                                            iter_lst.append(tup)
                                    pool = mp.Pool(mp.cpu_count())
                                    res_tot = []
                                    res_tot = pool.starmap_async(
                                        vanloan_2, 
                                        [(trans_mat_ABC, tup, dct_omegas['omega_%s0'%i],
                                          omega_77, cut_ABC[r+1]-cut_ABC[r]) for tup in iter_lst]
                                    ).get()
                                    pool.close()
                                    res_tot = (pi@p_ABC@sum(res_tot)).sum()
                                    ii = dct_num[i]
                                    jj = dct_num[j]
                                    tab[acc_tot] = [(ii, l, L), (jj, r, R), res_tot]
                                    acc_tot += 1
                                    tab[acc_tot] = [(jj, r, R), (ii, l, L), res_tot]
                                    acc_tot += 1
                    elif l < L < r == R:
                        for i in [3, 5, 6]:
                            for j in [3, 5, 6]:
                                # 00 -> 70 -> 7j
                                times_ABC = get_times(cut_ABC, [0, l, l+1, L, L+1, r])
                                omegas_ABC = [omega_tot_ABC, omega_00, 
                                              dct_omegas['omega_%s0'%i], dct_omegas['omega_%s0'%i], 
                                              omega_70, omega_70]
                                p_ABC = get_ABC_inf_bis(trans_mat_ABC, times_ABC, omegas_ABC)
                                if cut_ABC[r+1] != np.inf:
                                    res = vanloan_1(
                                        trans_mat_ABC,
                                        (omega_70, dct_omegas['omega_7%s'%j]),
                                        omega_70, omega_77,
                                        cut_ABC[r+1]-cut_ABC[r])
                                    ii = dct_num[i]
                                    jj = dct_num[j]
                                    tab[acc_tot]   = [(ii, l, L), (jj, r, R), (pi@p_ABC@res).sum()]
                                    tab[acc_tot+1] = [(jj, r, R), (ii, l, L), tab[acc_tot][2]]
                                    acc_tot += 2
                                else:
                                    A_mat = instant_mat(omega_70, dct_omegas['omega_7%s'%j], trans_mat_ABC)
                                    res = (-np.linalg.inv(trans_mat_ABC[:-2,:-2])@(A_mat[:-2,:-2]))[omega_70][:,dct_omegas['omega_7%s'%j]]
                                    ii = dct_num[i]
                                    jj = dct_num[j]
                                    tab[acc_tot] = [(ii, l, L), (jj, r, R), (pi@p_ABC@res).sum()]
                                    acc_tot += 1
                                    tab[acc_tot] = [(jj, r, R), (ii, l, L), tab[acc_tot-1][2]]
                                    acc_tot += 1
                    elif r < l == L < R:
                        for j in [3, 5, 6]:
                            times_ABC = get_times(cut_ABC, [0, r, r+1, l])
                            omegas_ABC = [omega_tot_ABC, omega_00,
                                          dct_omegas['omega_0%s'%j], dct_omegas['omega_0%s'%j]]
                            start = get_ABC_inf_bis(trans_mat_ABC, times_ABC, omegas_ABC)
                            times_ABC = get_times(cut_ABC, [l+1, R, R+1])
                            omegas_ABC = [dct_omegas['omega_7%s'%j], dct_omegas['omega_7%s'%j], 
                                          omega_77]
                            end = get_ABC_inf_bis(trans_mat_ABC, times_ABC, omegas_ABC)
                            for i in [3, 5, 6]:
                                # 00 -> 0j -> ij -> 7j -> 77
                                res = vanloan_1(
                                    trans_mat_ABC,
                                    (dct_omegas['omega_0%s'%j], dct_omegas['omega_%s%s'%(i, j)]),
                                    dct_omegas['omega_0%s'%j], dct_omegas['omega_7%s'%j],
                                    cut_ABC[l+1]-cut_ABC[l])
                                ii = dct_num[i]
                                jj = dct_num[j]
                                tab[acc_tot] = [(ii, l, L), (jj, r, R), (pi@start@res@end).sum()]
                                acc_tot += 1
                                tab[acc_tot] = [(jj, r, R), (ii, l, L), tab[acc_tot-1][2]]
                                acc_tot += 1
                    else:
                        continue
    return tab