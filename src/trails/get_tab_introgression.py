import numpy as np
import multiprocessing as mp
from ray.util.multiprocessing import Pool
import os
from scipy.special import comb
from trails.get_times import get_times
from trails.get_tab import precomp, get_ABC_precomp
from trails.vanloan import vanloan_1, vanloan_2, vanloan_3, instant_mat
from scipy.linalg import expm



def get_tab_ABC_introgression(state_space_ABC, trans_mat_ABC, cut_ABC, pi_ABC, names_tab_AB, n_int_AB):
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
    
    om = {}
    flatten = [list(sum(i, ())) for i in state_space_ABC]    
    for l in [0, 3, 5, 6, 7]:
        for r in [0, 3, 5, 6, 7]:
            if (l in [3, 5, 6, 7]) and (r in [3, 5, 6, 7]):
                om['%s%s' % (l, r)] = [i for i in range(203) if (l in flatten[i][::2]) and (r in flatten[i][1::2])]
            elif (l == 0) and (r in [3, 5, 6, 7]):
                om['%s%s' % (l, r)] = [i for i in range(203) if (all(x not in [3, 5, 6, 7] for x in flatten[i][::2])) and (r in flatten[i][1::2])]
            elif (l  in [3, 5, 6, 7]) and (r == 0):
                om['%s%s' % (l, r)] = [i for i in range(203) if (l in flatten[i][::2]) and (all(x not in [3, 5, 6, 7] for x in flatten[i][1::2]))]
            elif l == r == 0:
                om['%s%s' % (l, r)] = [i for i in range(203) if all(x not in [3, 5, 6, 7] for x in flatten[i])]
    omega_tot_ABC = [i for i in range(203)]
    om['71'] = sorted(om['73']+om['75']+om['76'])
    om['17'] = sorted(om['37']+om['57']+om['67'])
    om['10'] = sorted(om['30']+om['50']+om['60'])
    om['13'] = sorted(om['33']+om['53']+om['63'])
    om['15'] = sorted(om['35']+om['55']+om['65'])
    om['16'] = sorted(om['36']+om['56']+om['66'])
    om['11'] = sorted(om['13']+om['15']+om['16'])
    
    dct_num = {3:1, 5:2, 6:3}
    
    # Number of final states
    n_int_ABC = len(cut_ABC)-1
    n_markov_states = 2*n_int_AB*n_int_ABC+n_int_ABC*3+3*comb(n_int_ABC, 2, exact = True)
    
    # Create empty transition probability matrix
    tab=np.empty((n_markov_states**2, 3), dtype=object)
    # Create accumulator for keeping track of the indices for the table
    acc_tot = 0
    
    tm = get_times(cut_ABC, list(range(len(cut_ABC))))[:-1]
    pr = precomp(trans_mat_ABC, tm)
    
    ################
    ### V0 -> V0 ###
    ################

    # A pair of sites whose fate is to be V0 states is represented as ((0, l, L), (0, r, R)), where
    # l is the index of the interval where the first left coalescent happens, r is the same
    # for the first right coalescent, L is the same for the second left coalescent, and R is the 
    # second right coalescent. Remember that the probability of ((0, l, L) -> (0, r, R)) equals
    # that of ((0, r, R), (0, l, L)).
    # start = time.time()
    for l in range(n_int_AB):
        for r in range(n_int_AB):
            cond = [i == ((0, l),(0, r)) for i in names_tab_AB]
            pi = pi_ABC[cond]
            for L in range(n_int_ABC):
                for R in range(n_int_ABC):
                    if L < R:                        
                        omegas = [omega_tot_ABC]+[om['11']]*L+[om['71']]*(R-L)+[om['77']]
                        p_ABC = get_ABC_precomp(pr, omegas, list(range(R+int(cut_ABC[R+1]!=np.inf))))
                        tab[acc_tot]   = [(0, l, L), (0, r, R), (pi@p_ABC).sum()]
                        tab[acc_tot+1] = [(0, r, R), (0, l, L), tab[acc_tot][2]]
                        acc_tot += 2
                    elif L == R:
                        omegas = [omega_tot_ABC]+[om['11']]*L+[om['77']]
                        p_ABC = get_ABC_precomp(pr, omegas, list(range(L+int(cut_ABC[L+1]!=np.inf))))
                        tab[acc_tot] = [(0, l, L), (0, r, R), (pi@p_ABC).sum()]
                        acc_tot += 1
                    else:
                        continue

    ##############
    ### I -> I ###
    ##############

    # start = time.time()
    for l in range(n_int_AB):
        for r in range(n_int_AB):
            cond = [i == ((4, l),(4, r)) for i in names_tab_AB]
            pi = pi_ABC[cond]
            for L in range(n_int_ABC):
                for R in range(n_int_ABC):
                    if L < R:                        
                        omegas = [omega_tot_ABC]+[om['11']]*L+[om['71']]*(R-L)+[om['77']]
                        p_ABC = get_ABC_precomp(pr, omegas, list(range(R+int(cut_ABC[R+1]!=np.inf))))
                        tab[acc_tot]   = [(4, l, L), (4, r, R), (pi@p_ABC).sum()]
                        tab[acc_tot+1] = [(4, r, R), (4, l, L), tab[acc_tot][2]]
                        acc_tot += 2
                    elif L == R:
                        omegas = [omega_tot_ABC]+[om['11']]*L+[om['77']]
                        p_ABC = get_ABC_precomp(pr, omegas, list(range(L+int(cut_ABC[L+1]!=np.inf))))
                        tab[acc_tot] = [(4, l, L), (4, r, R), (pi@p_ABC).sum()]
                        acc_tot += 1
                    else:
                        continue
    # end = time.time()
    # print("(V0 -> V0) = %s" % (end - start))
    # print()

    ###############
    ### V0 -> I ###
    ###############
     
    # start = time.time()
    for l in range(n_int_AB):
        for r in range(n_int_AB):
            cond = [i == ((0, l),(4, r)) for i in names_tab_AB]
            pi = pi_ABC[cond]
            for L in range(n_int_ABC):
                for R in range(n_int_ABC):
                    if L < R:                        
                        omegas = [omega_tot_ABC]+[om['11']]*L+[om['71']]*(R-L)+[om['77']]
                        p_ABC = get_ABC_precomp(pr, omegas, list(range(R+int(cut_ABC[R+1]!=np.inf))))
                        tab[acc_tot]   = [(0, l, L), (4, r, R), (pi@p_ABC).sum()]
                        acc_tot += 1
                    elif L > R:                        
                        omegas = [omega_tot_ABC]+[om['11']]*R+[om['17']]*(L-R)+[om['77']]
                        p_ABC = get_ABC_precomp(pr, omegas, list(range(L+int(cut_ABC[L+1]!=np.inf))))
                        tab[acc_tot]   = [(0, l, L), (4, r, R), (pi@p_ABC).sum()]
                        acc_tot += 1
                    elif L == R:
                        omegas = [omega_tot_ABC]+[om['11']]*L+[om['77']]
                        p_ABC = get_ABC_precomp(pr, omegas, list(range(L+int(cut_ABC[L+1]!=np.inf))))
                        tab[acc_tot] = [(0, l, L), (4, r, R), (pi@p_ABC).sum()]
                        acc_tot += 1
                    else:
                        continue
    # end = time.time()
    # print("(V0 -> V0) = %s" % (end - start))
    # print()
    
    ###############
    ### I -> V0 ###
    ###############
    
    # start = time.time()
    for l in range(n_int_AB):
        for r in range(n_int_AB):
            cond = [i == ((4, l),(0, r)) for i in names_tab_AB]
            pi = pi_ABC[cond]
            for L in range(n_int_ABC):
                for R in range(n_int_ABC):
                    if L < R:                        
                        omegas = [omega_tot_ABC]+[om['11']]*L+[om['71']]*(R-L)+[om['77']]
                        p_ABC = get_ABC_precomp(pr, omegas, list(range(R+int(cut_ABC[R+1]!=np.inf))))
                        tab[acc_tot]   = [(4, l, L), (0, r, R), (pi@p_ABC).sum()]
                        acc_tot += 1
                    elif L > R:                        
                        omegas = [omega_tot_ABC]+[om['11']]*R+[om['17']]*(L-R)+[om['77']]
                        p_ABC = get_ABC_precomp(pr, omegas, list(range(L+int(cut_ABC[L+1]!=np.inf))))
                        tab[acc_tot]   = [(4, l, L), (0, r, R), (pi@p_ABC).sum()]
                        acc_tot += 1
                    elif L == R:
                        omegas = [omega_tot_ABC]+[om['11']]*L+[om['77']]
                        p_ABC = get_ABC_precomp(pr, omegas, list(range(L+int(cut_ABC[L+1]!=np.inf))))
                        tab[acc_tot] = [(4, l, L), (0, r, R), (pi@p_ABC).sum()]
                        acc_tot += 1
                    else:
                        continue
    # end = time.time()
    # print("(V0 -> V0) = %s" % (end - start))
    # print()
    
    ################################
    ### V0/I -> deep coalescence ###
    ### Deep coalescence -> V0/I ###
    ################################

    
    # A pair of sites where the left site is in V0 and the right site is of deep coalescence
    # is represented as ((0, l, L), (i, r, R)), where l is the index of the interval where the 
    # first left coalescent happens, L is the same for the second left coalescent, r is the 
    # same for the first right coalescent and R is the same for the second right coalescent. The index
    # i can take values from 1 to 4, where 1 to 3 represents deep coalescent and l < L, and 4 
    # represents a multiple merger event where l = L. Remember that the probability of 
    # ((0, l, L) -> (i, r, R)) and that of ((i, r, R) -> (0, l, L)) is the same. Also,
    # ((0, l, L) -> (1, r, R)) = ((0, l, L) -> (2, r, R)) = ((0, l, L) -> (3, r, R)), following ILS.
    pool_lst = []
    for L in range(n_int_ABC):
        for r in range(n_int_ABC):
            for R in range(r, n_int_ABC):
                if L < r < R:
                    pool_lst.append((L, r, R))
                elif L == r < R:
                    pool_lst.append((L, r, R))
                elif r < L < R:
                    pool_lst.append((L, r, R))
                elif r < L == R:
                    pool_lst.append((L, r, R))
                elif r < R < L:
                    pool_lst.append((L, r, R))
                elif L < r == R:
                    pool_lst.append((L, r, R))
                elif L == r == R:
                    pool_lst.append((L, r, R))
                elif r == R < L:
                    pool_lst.append((L, r, R))
    # starttim = time.time()
    if (n_int_AB == 1) and (n_int_ABC < 10):
        init_worker_AB(pi_ABC, om, omega_tot_ABC, pr, cut_ABC, dct_num, trans_mat_ABC, n_int_AB, names_tab_AB)
        res_lst = [pool_AB_total(*x) for x in pool_lst]
        for result in res_lst:
            for x in result:
                tab[acc_tot] = x
                acc_tot += 1
    else:
        try:
            ncpus = int(os.environ["SLURM_JOB_CPUS_PER_NODE"])
        except KeyError:
            ncpus = mp.cpu_count()
        pool = Pool(
            ncpus, 
            initializer=init_worker_AB, 
            initargs=(pi_ABC, om, omega_tot_ABC, pr, cut_ABC, dct_num, trans_mat_ABC, n_int_AB, names_tab_AB,)
        )
        for result in pool.starmap_async(pool_AB_total, pool_lst).get():
            for x in result:
                tab[acc_tot] = x
                acc_tot += 1
        pool.close()
    # endtim = time.time()
    # print("First {}".format(endtim - starttim))
    
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
    cond = [i == ('D','D') for i in names_tab_AB]
    pi = pi_ABC[cond]
    # The number of tasks is satistied by n*(n+1)*(n**2+n+2)/8 (A002817)
    pool_lst = []
    for l in range(n_int_ABC):
        for L in range(l, n_int_ABC):
            for r in range(n_int_ABC):
                for R in range(r, n_int_ABC):
                    if l < L < r < R:
                        pool_lst.append((l, L, r, R))
                    elif l < L == r < R:
                        pool_lst.append((l, L, r, R))
                    elif l == r < L < R:
                        pool_lst.append((l, L, r, R)) 
                    elif l < r < L < R:
                        pool_lst.append((l, L, r, R))
                    elif r < l < L < R:
                        pool_lst.append((l, L, r, R))
                    elif l == r < L == R:
                        pool_lst.append((l, L, r, R))
                    elif l < r < L == R:
                        pool_lst.append((l, L, r, R))
                    elif l == r == L == R:
                        pool_lst.append((l, L, r, R))
                    elif l == L < r == R:
                        pool_lst.append((l, L, r, R))
                    elif l == L < r < R:
                        pool_lst.append((l, L, r, R))
                    elif l == L == r < R:
                        pool_lst.append((l, L, r, R))
                    elif l < L == r == R:
                        pool_lst.append((l, L, r, R))
                    elif l < L < r == R:
                        pool_lst.append((l, L, r, R))
                    elif r < l == L < R:
                        pool_lst.append((l, L, r, R))
    # starttim = time.time()
    if n_int_ABC in [1, 2]:
        init_worker_ABC(pi, om, omega_tot_ABC, pr, cut_ABC, dct_num, trans_mat_ABC)
        res_lst = [pool_ABC(*x) for x in pool_lst]
        for result in res_lst:
            for x in result:
                tab[acc_tot] = x
                acc_tot += 1
    else:
        try:
            ncpus = int(os.environ["SLURM_JOB_CPUS_PER_NODE"])
        except KeyError:
            ncpus = mp.cpu_count()
        pool = Pool(
            ncpus, 
            initializer=init_worker_ABC, 
            initargs=(pi, om, omega_tot_ABC, pr, cut_ABC, dct_num, trans_mat_ABC,)
        )
        for result in pool.starmap_async(pool_ABC, pool_lst).get():
            for x in result:
                tab[acc_tot] = x
                acc_tot += 1
        pool.close()
    # endtim = time.time()
    # print("Second {}".format(endtim - starttim))
        
    # print(tab[:, 2].sum())
    return tab



def pool_AB_introgression(L, r, R):
    pi_ABC, om, omega_tot_ABC, pr, cut_ABC, dct_num, trans_mat_ABC, n_int_AB, names_tab_AB = shared_data
    tab = []
    # start = time.time()
    if L < r < R:
        omegas_pre = [omega_tot_ABC]+[om['10']]*L+[om['70']]*(r-L)
        for i in [3, 5, 6]:
            ii = dct_num[i]
            omegas = omegas_pre+[om['7%s'%i]]*(R-r)+[om['77']]
            p_ABC = get_ABC_precomp(pr, omegas, list(range(R+int(cut_ABC[R+1]!=np.inf))))
            for l in range(n_int_AB):
                cond = [i == ((4, l), 'D') for i in names_tab_AB]
                pi = pi_ABC[cond]
                tab.append([(4, l, L), (ii, r, R), (pi@p_ABC).sum()])
                tab.append([(ii, r, R), (4, l, L), tab[-1][2]])
    elif L == r < R:
        omegas_pre = [omega_tot_ABC]+[om['10']]*L
        for i in [3, 5, 6]:
            ii = dct_num[i]
            omegas = omegas_pre+[om['7%s'%i]]*(R-L)+[om['77']]
            p_ABC = get_ABC_precomp(pr, omegas, list(range(R+int(cut_ABC[R+1]!=np.inf))))
            for l in range(n_int_AB):
                cond = [i == ((4, l), 'D') for i in names_tab_AB]
                pi = pi_ABC[cond]
                tab.append([(4, l, L), (ii, r, R), (pi@p_ABC).sum()])
                tab.append([(ii, r, R), (4, l, L), tab[-1][2]])
    elif r < L < R:
        omegas_pre = [omega_tot_ABC]+[om['10']]*r
        for i in [3, 5, 6]:
            ii = dct_num[i]
            omegas = omegas_pre+[om['1%s'%i]]*(L-r)+[om['7%s'%i]]*(R-L)+[om['77']]
            p_ABC = get_ABC_precomp(pr, omegas, list(range(R+int(cut_ABC[R+1]!=np.inf))))
            for l in range(n_int_AB):
                cond = [i == ((4, l), 'D') for i in names_tab_AB]
                pi = pi_ABC[cond]
                tab.append([(4, l, L), (ii, r, R), (pi@p_ABC).sum()])
                tab.append([(ii, r, R), (4, l, L), tab[-1][2]])
    elif r < L == R:
        omegas_pre = [omega_tot_ABC]+[om['10']]*r
        for i in [3, 5, 6]:
            ii = dct_num[i]
            omegas = omegas_pre+[om['1%s'%i]]*(L-r)+[om['77']]
            p_ABC = get_ABC_precomp(pr, omegas, list(range(R+int(cut_ABC[R+1]!=np.inf))))
            for l in range(n_int_AB):
                cond = [i == ((4, l), 'D') for i in names_tab_AB]
                pi = pi_ABC[cond]
                tab.append([(4, l, L), (ii, r, R), (pi@p_ABC).sum()])
                tab.append([(ii, r, R), (4, l, L), tab[-1][2]])
    elif r < R < L:
        omegas_pre = [omega_tot_ABC]+[om['10']]*r
        for i in [3, 5, 6]:
            ii = dct_num[i]
            omegas = omegas_pre+[om['1%s'%i]]*(R-r)+[om['17']]*(L-R)+[om['77']]
            p_ABC = get_ABC_precomp(pr, omegas, list(range(L+int(cut_ABC[L+1]!=np.inf))))
            for l in range(n_int_AB):
                cond = [i == ((4, l), 'D') for i in names_tab_AB]
                pi = pi_ABC[cond]
                tab.append([(4, l, L), (ii, r, R), (pi@p_ABC).sum()])
                tab.append([(ii, r, R), (4, l, L), tab[-1][2]])
    elif L < r == R:
        omegas = [omega_tot_ABC]+[om['10']]*L+[om['70']]*(r-L)
        p_ABC = get_ABC_precomp(pr, omegas, list(range(R)))
        for i in [3, 5, 6]:
            if cut_ABC[r+1] != np.inf:
                res = vanloan_1(
                    trans_mat_ABC, (om['70'], om['7%s'%i]),
                    om['70'], om['77'], cut_ABC[r+1]-cut_ABC[r])
            else:
                A_mat = instant_mat(om['70'], om['7%s'%i], trans_mat_ABC)
                res = (-np.linalg.inv(trans_mat_ABC[:-2,:-2])@(A_mat[:-2,:-2]))[om['70']][:,om['7%s'%i]]
            for l in range(n_int_AB):
                cond = [i == ((4, l), 'D') for i in names_tab_AB]
                pi = pi_ABC[cond]
                ii = dct_num[i]
                tab.append([(4, l, L), (ii, r, R), (pi@p_ABC@res).sum()])
                tab.append([(ii, r, R), (4, l, L), tab[-1][2]])
    elif L == r == R:
        omegas = [omega_tot_ABC]+[om['10']]*R
        p_ABC_pre = get_ABC_precomp(pr, omegas, list(range(R)))
        # Get right shapes for first interval
        p_ABC = p_ABC_pre[:,om['10']] if L==0 else p_ABC_pre
        for i in [3, 5, 6]:
            if cut_ABC[r+1] == np.inf:
                A_mat = instant_mat(om['10'], om['1%s'%i], trans_mat_ABC)
                res_1 = (-np.linalg.inv(trans_mat_ABC[:-2,:-2])@(A_mat[:-2,:-2]))[om['10']][:,om['1%s'%i]]
                A_mat = instant_mat(om['10'], om['7%s'%i], trans_mat_ABC)
                res_2 = (-np.linalg.inv(trans_mat_ABC[:-2,:-2])@(A_mat[:-2,:-2]))[om['10']][:,om['7%s'%i]]
                A_mat_1 = instant_mat(om['10'], om['70'], trans_mat_ABC)
                A_mat_2 = instant_mat(om['70'], om['7%s'%i], trans_mat_ABC)
                C_mat_upper =  np.concatenate((trans_mat_ABC[:-2,:-2], A_mat_1[:-2,:-2]), axis = 1)
                C_mat_lower = np.concatenate((np.zeros((201,201)), trans_mat_ABC[:-2,:-2]), axis = 1)
                C_mat = np.concatenate((C_mat_upper, C_mat_lower), axis = 0)
                res_3 = ((-np.linalg.inv(C_mat)[0:201,-201:])@(A_mat_2[:-2,:-2]))[om['10']][:,om['7%s'%i]]
                for l in range(n_int_AB):
                    cond = [i == ((4, l), 'D') for i in names_tab_AB]
                    pi = pi_ABC[cond]
                    ii = dct_num[i]
                    tab.append([(4, l, L), (ii, r, R), (pi@p_ABC@res_1).sum()+(pi@p_ABC@sum([res_2, res_3])).sum()])
                    tab.append([(ii, r, R), (4, l, L), tab[-1][2]])
            else:
                omega_lst = ['10', '1%s'%i, '17', '70', '7%s'%i, '77']
                iter_lst = []
                for y in range(1, len(omega_lst)):
                    for z in range(y+1, len(omega_lst)):
                        if int(omega_lst[z][0]) < int(omega_lst[y][0]):
                            continue
                        elif int(omega_lst[z][1]) < int(omega_lst[y][1]):
                            continue
                        elif (int(omega_lst[z][1])-int(omega_lst[y][1]))==7:
                            continue
                        elif omega_lst[y][1]=='7':
                            continue
                        tup = (om['%s'%(omega_lst[0],)], 
                                om['%s'%(omega_lst[y],)],
                                om['%s'%(omega_lst[z],)])
                        iter_lst.append(tup)
                iterable = [(trans_mat_ABC, tup, om['10'], om['77'], cut_ABC[r+1]-cut_ABC[r]) for tup in iter_lst]
                res_tot = [vanloan_2(*x) for x in iterable]
                for l in range(n_int_AB):
                    cond = [i == ((4, l), 'D') for i in names_tab_AB]
                    pi = pi_ABC[cond]
                    ii = dct_num[i]
                    tab.append([(4, l, L), (ii, r, R), (pi@p_ABC@sum(res_tot)).sum()])
                    tab.append([(ii, r, R), (4, l, L), tab[-1][2]])
    elif r == R < L:
        omegas = [omega_tot_ABC]+[om['10']]*R
        p_ABC_start_pre = get_ABC_precomp(pr, omegas, list(range(R)))
        # Get right shapes for first interval
        p_ABC_start = p_ABC_start_pre[:,om['10']] if R==0 else p_ABC_start_pre
        omegas = [om['17']]*(L-R)+[om['77']]
        p_ABC_end = get_ABC_precomp(pr, omegas, list(range(R+1, L+int(cut_ABC[L+1]!=np.inf))))
        for i in [3, 5, 6]:
            A_mat = instant_mat(om['10'], om['1%s'%i], trans_mat_ABC)
            C_mat_upper = np.concatenate((trans_mat_ABC, A_mat), axis = 1)
            C_mat_lower = np.concatenate((np.zeros((203,203)), trans_mat_ABC), axis = 1)
            C_mat = np.concatenate((C_mat_upper, C_mat_lower), axis = 0)
            res = (expm(C_mat*(cut_ABC[r+1]-cut_ABC[r]))[:203,-203:])[om['10']][:,om['17']]
            for l in range(n_int_AB):
                cond = [i == ((4, l), 'D') for i in names_tab_AB]
                pi = pi_ABC[cond]
                ii = dct_num[i]
                tab.append([(4, l, L), (ii, r, R), (pi@p_ABC_start@res@p_ABC_end).sum()])
                tab.append([(ii, r, R), (4, l, L), tab[-1][2]])
    # end = time.time()
    # print("((4, {}, {}) -> (i, {}, {})) = {}".format('l', L, r, R, end - start))
    return tab




def pool_AB_total(L, r, R):
    pi_ABC, om, omega_tot_ABC, pr, cut_ABC, dct_num, trans_mat_ABC, n_int_AB, names_tab_AB = shared_data
    tab = []
    # start = time.time()
    if L < r < R:
        omegas_pre = [omega_tot_ABC]+[om['10']]*L+[om['70']]*(r-L)
        for i in [3, 5, 6]:
            ii = dct_num[i]
            omegas = omegas_pre+[om['7%s'%i]]*(R-r)+[om['77']]
            p_ABC = get_ABC_precomp(pr, omegas, list(range(R+int(cut_ABC[R+1]!=np.inf))))
            for l in range(n_int_AB):
                cond = [i == ((0, l), 'D') for i in names_tab_AB]
                pi = pi_ABC[cond]
                tab.append([(0, l, L), (ii, r, R), (pi@p_ABC).sum()])
                tab.append([(ii, r, R), (0, l, L), tab[-1][2]])
                cond = [i == ((4, l), 'D') for i in names_tab_AB]
                pi = pi_ABC[cond]
                tab.append([(4, l, L), (ii, r, R), (pi@p_ABC).sum()])
                tab.append([(ii, r, R), (4, l, L), tab[-1][2]])
    elif L == r < R:
        omegas_pre = [omega_tot_ABC]+[om['10']]*L
        for i in [3, 5, 6]:
            ii = dct_num[i]
            omegas = omegas_pre+[om['7%s'%i]]*(R-L)+[om['77']]
            p_ABC = get_ABC_precomp(pr, omegas, list(range(R+int(cut_ABC[R+1]!=np.inf))))
            for l in range(n_int_AB):
                cond = [i == ((0, l), 'D') for i in names_tab_AB]
                pi = pi_ABC[cond]
                tab.append([(0, l, L), (ii, r, R), (pi@p_ABC).sum()])
                tab.append([(ii, r, R), (0, l, L), tab[-1][2]])
                cond = [i == ((4, l), 'D') for i in names_tab_AB]
                pi = pi_ABC[cond]
                tab.append([(4, l, L), (ii, r, R), (pi@p_ABC).sum()])
                tab.append([(ii, r, R), (4, l, L), tab[-1][2]])
    elif r < L < R:
        omegas_pre = [omega_tot_ABC]+[om['10']]*r
        for i in [3, 5, 6]:
            ii = dct_num[i]
            omegas = omegas_pre+[om['1%s'%i]]*(L-r)+[om['7%s'%i]]*(R-L)+[om['77']]
            p_ABC = get_ABC_precomp(pr, omegas, list(range(R+int(cut_ABC[R+1]!=np.inf))))
            for l in range(n_int_AB):
                cond = [i == ((0, l), 'D') for i in names_tab_AB]
                pi = pi_ABC[cond]
                tab.append([(0, l, L), (ii, r, R), (pi@p_ABC).sum()])
                tab.append([(ii, r, R), (0, l, L), tab[-1][2]])
                cond = [i == ((4, l), 'D') for i in names_tab_AB]
                pi = pi_ABC[cond]
                tab.append([(4, l, L), (ii, r, R), (pi@p_ABC).sum()])
                tab.append([(ii, r, R), (4, l, L), tab[-1][2]])
    elif r < L == R:
        omegas_pre = [omega_tot_ABC]+[om['10']]*r
        for i in [3, 5, 6]:
            ii = dct_num[i]
            omegas = omegas_pre+[om['1%s'%i]]*(L-r)+[om['77']]
            p_ABC = get_ABC_precomp(pr, omegas, list(range(R+int(cut_ABC[R+1]!=np.inf))))
            for l in range(n_int_AB):
                cond = [i == ((0, l), 'D') for i in names_tab_AB]
                pi = pi_ABC[cond]
                tab.append([(0, l, L), (ii, r, R), (pi@p_ABC).sum()])
                tab.append([(ii, r, R), (0, l, L), tab[-1][2]])
                cond = [i == ((4, l), 'D') for i in names_tab_AB]
                pi = pi_ABC[cond]
                tab.append([(4, l, L), (ii, r, R), (pi@p_ABC).sum()])
                tab.append([(ii, r, R), (4, l, L), tab[-1][2]])
    elif r < R < L:
        omegas_pre = [omega_tot_ABC]+[om['10']]*r
        for i in [3, 5, 6]:
            ii = dct_num[i]
            omegas = omegas_pre+[om['1%s'%i]]*(R-r)+[om['17']]*(L-R)+[om['77']]
            p_ABC = get_ABC_precomp(pr, omegas, list(range(L+int(cut_ABC[L+1]!=np.inf))))
            for l in range(n_int_AB):
                cond = [i == ((0, l), 'D') for i in names_tab_AB]
                pi = pi_ABC[cond]
                tab.append([(0, l, L), (ii, r, R), (pi@p_ABC).sum()])
                tab.append([(ii, r, R), (0, l, L), tab[-1][2]])
                cond = [i == ((4, l), 'D') for i in names_tab_AB]
                pi = pi_ABC[cond]
                tab.append([(4, l, L), (ii, r, R), (pi@p_ABC).sum()])
                tab.append([(ii, r, R), (4, l, L), tab[-1][2]])
    elif L < r == R:
        omegas = [omega_tot_ABC]+[om['10']]*L+[om['70']]*(r-L)
        p_ABC = get_ABC_precomp(pr, omegas, list(range(R)))
        for i in [3, 5, 6]:
            if cut_ABC[r+1] != np.inf:
                res = vanloan_1(
                    trans_mat_ABC, (om['70'], om['7%s'%i]),
                    om['70'], om['77'], cut_ABC[r+1]-cut_ABC[r])
            else:
                A_mat = instant_mat(om['70'], om['7%s'%i], trans_mat_ABC)
                res = (-np.linalg.inv(trans_mat_ABC[:-2,:-2])@(A_mat[:-2,:-2]))[om['70']][:,om['7%s'%i]]
            for l in range(n_int_AB):
                cond = [i == ((0, l), 'D') for i in names_tab_AB]
                pi = pi_ABC[cond]
                ii = dct_num[i]
                tab.append([(0, l, L), (ii, r, R), (pi@p_ABC@res).sum()])
                tab.append([(ii, r, R), (0, l, L), tab[-1][2]])
                cond = [i == ((4, l), 'D') for i in names_tab_AB]
                pi = pi_ABC[cond]
                ii = dct_num[i]
                tab.append([(4, l, L), (ii, r, R), (pi@p_ABC@res).sum()])
                tab.append([(ii, r, R), (4, l, L), tab[-1][2]])
    elif L == r == R:
        omegas = [omega_tot_ABC]+[om['10']]*R
        p_ABC_pre = get_ABC_precomp(pr, omegas, list(range(R)))
        # Get right shapes for first interval
        p_ABC = p_ABC_pre[:,om['10']] if L==0 else p_ABC_pre
        for i in [3, 5, 6]:
            if cut_ABC[r+1] == np.inf:
                A_mat = instant_mat(om['10'], om['1%s'%i], trans_mat_ABC)
                res_1 = (-np.linalg.inv(trans_mat_ABC[:-2,:-2])@(A_mat[:-2,:-2]))[om['10']][:,om['1%s'%i]]
                A_mat = instant_mat(om['10'], om['7%s'%i], trans_mat_ABC)
                res_2 = (-np.linalg.inv(trans_mat_ABC[:-2,:-2])@(A_mat[:-2,:-2]))[om['10']][:,om['7%s'%i]]
                A_mat_1 = instant_mat(om['10'], om['70'], trans_mat_ABC)
                A_mat_2 = instant_mat(om['70'], om['7%s'%i], trans_mat_ABC)
                C_mat_upper =  np.concatenate((trans_mat_ABC[:-2,:-2], A_mat_1[:-2,:-2]), axis = 1)
                C_mat_lower = np.concatenate((np.zeros((201,201)), trans_mat_ABC[:-2,:-2]), axis = 1)
                C_mat = np.concatenate((C_mat_upper, C_mat_lower), axis = 0)
                res_3 = ((-np.linalg.inv(C_mat)[0:201,-201:])@(A_mat_2[:-2,:-2]))[om['10']][:,om['7%s'%i]]
                for l in range(n_int_AB):
                    cond = [i == ((0, l), 'D') for i in names_tab_AB]
                    pi = pi_ABC[cond]
                    ii = dct_num[i]
                    tab.append([(0, l, L), (ii, r, R), (pi@p_ABC@res_1).sum()+(pi@p_ABC@sum([res_2, res_3])).sum()])
                    tab.append([(ii, r, R), (0, l, L), tab[-1][2]])
                    cond = [i == ((4, l), 'D') for i in names_tab_AB]
                    pi = pi_ABC[cond]
                    ii = dct_num[i]
                    tab.append([(4, l, L), (ii, r, R), (pi@p_ABC@res_1).sum()+(pi@p_ABC@sum([res_2, res_3])).sum()])
                    tab.append([(ii, r, R), (4, l, L), tab[-1][2]])
            else:
                omega_lst = ['10', '1%s'%i, '17', '70', '7%s'%i, '77']
                iter_lst = []
                for y in range(1, len(omega_lst)):
                    for z in range(y+1, len(omega_lst)):
                        if int(omega_lst[z][0]) < int(omega_lst[y][0]):
                            continue
                        elif int(omega_lst[z][1]) < int(omega_lst[y][1]):
                            continue
                        elif (int(omega_lst[z][1])-int(omega_lst[y][1]))==7:
                            continue
                        elif omega_lst[y][1]=='7':
                            continue
                        tup = (om['%s'%(omega_lst[0],)], 
                                om['%s'%(omega_lst[y],)],
                                om['%s'%(omega_lst[z],)])
                        iter_lst.append(tup)
                iterable = [(trans_mat_ABC, tup, om['10'], om['77'], cut_ABC[r+1]-cut_ABC[r]) for tup in iter_lst]
                res_tot = [vanloan_2(*x) for x in iterable]
                for l in range(n_int_AB):
                    cond = [i == ((0, l), 'D') for i in names_tab_AB]
                    pi = pi_ABC[cond]
                    ii = dct_num[i]
                    tab.append([(0, l, L), (ii, r, R), (pi@p_ABC@sum(res_tot)).sum()])
                    tab.append([(ii, r, R), (0, l, L), tab[-1][2]])
                    cond = [i == ((4, l), 'D') for i in names_tab_AB]
                    pi = pi_ABC[cond]
                    ii = dct_num[i]
                    tab.append([(4, l, L), (ii, r, R), (pi@p_ABC@sum(res_tot)).sum()])
                    tab.append([(ii, r, R), (4, l, L), tab[-1][2]])
    elif r == R < L:
        omegas = [omega_tot_ABC]+[om['10']]*R
        p_ABC_start_pre = get_ABC_precomp(pr, omegas, list(range(R)))
        # Get right shapes for first interval
        p_ABC_start = p_ABC_start_pre[:,om['10']] if R==0 else p_ABC_start_pre
        omegas = [om['17']]*(L-R)+[om['77']]
        p_ABC_end = get_ABC_precomp(pr, omegas, list(range(R+1, L+int(cut_ABC[L+1]!=np.inf))))
        for i in [3, 5, 6]:
            A_mat = instant_mat(om['10'], om['1%s'%i], trans_mat_ABC)
            C_mat_upper = np.concatenate((trans_mat_ABC, A_mat), axis = 1)
            C_mat_lower = np.concatenate((np.zeros((203,203)), trans_mat_ABC), axis = 1)
            C_mat = np.concatenate((C_mat_upper, C_mat_lower), axis = 0)
            res = (expm(C_mat*(cut_ABC[r+1]-cut_ABC[r]))[:203,-203:])[om['10']][:,om['17']]
            for l in range(n_int_AB):
                cond = [i == ((0, l), 'D') for i in names_tab_AB]
                pi = pi_ABC[cond]
                ii = dct_num[i]
                tab.append([(0, l, L), (ii, r, R), (pi@p_ABC_start@res@p_ABC_end).sum()])
                tab.append([(ii, r, R), (0, l, L), tab[-1][2]])
                cond = [i == ((4, l), 'D') for i in names_tab_AB]
                pi = pi_ABC[cond]
                ii = dct_num[i]
                tab.append([(4, l, L), (ii, r, R), (pi@p_ABC_start@res@p_ABC_end).sum()])
                tab.append([(ii, r, R), (4, l, L), tab[-1][2]])
    # end = time.time()
    # print("((0, {}, {}) -> (i, {}, {})) = {}".format('l', L, r, R, end - start))
    return tab


def pool_AB(L, r, R):
    pi_ABC, om, omega_tot_ABC, pr, cut_ABC, dct_num, trans_mat_ABC, n_int_AB, names_tab_AB = shared_data
    tab = []
    # start = time.time()
    if L < r < R:
        omegas_pre = [omega_tot_ABC]+[om['10']]*L+[om['70']]*(r-L)
        for i in [3, 5, 6]:
            ii = dct_num[i]
            omegas = omegas_pre+[om['7%s'%i]]*(R-r)+[om['77']]
            p_ABC = get_ABC_precomp(pr, omegas, list(range(R+int(cut_ABC[R+1]!=np.inf))))
            for l in range(n_int_AB):
                cond = [i == ((0, l), 'D') for i in names_tab_AB]
                pi = pi_ABC[cond]
                tab.append([(0, l, L), (ii, r, R), (pi@p_ABC).sum()])
                tab.append([(ii, r, R), (0, l, L), tab[-1][2]])
    elif L == r < R:
        omegas_pre = [omega_tot_ABC]+[om['10']]*L
        for i in [3, 5, 6]:
            ii = dct_num[i]
            omegas = omegas_pre+[om['7%s'%i]]*(R-L)+[om['77']]
            p_ABC = get_ABC_precomp(pr, omegas, list(range(R+int(cut_ABC[R+1]!=np.inf))))
            for l in range(n_int_AB):
                cond = [i == ((0, l), 'D') for i in names_tab_AB]
                pi = pi_ABC[cond]
                tab.append([(0, l, L), (ii, r, R), (pi@p_ABC).sum()])
                tab.append([(ii, r, R), (0, l, L), tab[-1][2]])
    elif r < L < R:
        omegas_pre = [omega_tot_ABC]+[om['10']]*r
        for i in [3, 5, 6]:
            ii = dct_num[i]
            omegas = omegas_pre+[om['1%s'%i]]*(L-r)+[om['7%s'%i]]*(R-L)+[om['77']]
            p_ABC = get_ABC_precomp(pr, omegas, list(range(R+int(cut_ABC[R+1]!=np.inf))))
            for l in range(n_int_AB):
                cond = [i == ((0, l), 'D') for i in names_tab_AB]
                pi = pi_ABC[cond]
                tab.append([(0, l, L), (ii, r, R), (pi@p_ABC).sum()])
                tab.append([(ii, r, R), (0, l, L), tab[-1][2]])
    elif r < L == R:
        omegas_pre = [omega_tot_ABC]+[om['10']]*r
        for i in [3, 5, 6]:
            ii = dct_num[i]
            omegas = omegas_pre+[om['1%s'%i]]*(L-r)+[om['77']]
            p_ABC = get_ABC_precomp(pr, omegas, list(range(R+int(cut_ABC[R+1]!=np.inf))))
            for l in range(n_int_AB):
                cond = [i == ((0, l), 'D') for i in names_tab_AB]
                pi = pi_ABC[cond]
                tab.append([(0, l, L), (ii, r, R), (pi@p_ABC).sum()])
                tab.append([(ii, r, R), (0, l, L), tab[-1][2]])
    elif r < R < L:
        omegas_pre = [omega_tot_ABC]+[om['10']]*r
        for i in [3, 5, 6]:
            ii = dct_num[i]
            omegas = omegas_pre+[om['1%s'%i]]*(R-r)+[om['17']]*(L-R)+[om['77']]
            p_ABC = get_ABC_precomp(pr, omegas, list(range(L+int(cut_ABC[L+1]!=np.inf))))
            for l in range(n_int_AB):
                cond = [i == ((0, l), 'D') for i in names_tab_AB]
                pi = pi_ABC[cond]
                tab.append([(0, l, L), (ii, r, R), (pi@p_ABC).sum()])
                tab.append([(ii, r, R), (0, l, L), tab[-1][2]])
    elif L < r == R:
        omegas = [omega_tot_ABC]+[om['10']]*L+[om['70']]*(r-L)
        p_ABC = get_ABC_precomp(pr, omegas, list(range(R)))
        for i in [3, 5, 6]:
            if cut_ABC[r+1] != np.inf:
                res = vanloan_1(
                    trans_mat_ABC, (om['70'], om['7%s'%i]),
                    om['70'], om['77'], cut_ABC[r+1]-cut_ABC[r])
            else:
                A_mat = instant_mat(om['70'], om['7%s'%i], trans_mat_ABC)
                res = (-np.linalg.inv(trans_mat_ABC[:-2,:-2])@(A_mat[:-2,:-2]))[om['70']][:,om['7%s'%i]]
            for l in range(n_int_AB):
                cond = [i == ((0, l), 'D') for i in names_tab_AB]
                pi = pi_ABC[cond]
                ii = dct_num[i]
                tab.append([(0, l, L), (ii, r, R), (pi@p_ABC@res).sum()])
                tab.append([(ii, r, R), (0, l, L), tab[-1][2]])
    elif L == r == R:
        omegas = [omega_tot_ABC]+[om['10']]*R
        p_ABC_pre = get_ABC_precomp(pr, omegas, list(range(R)))
        # Get right shapes for first interval
        p_ABC = p_ABC_pre[:,om['10']] if L==0 else p_ABC_pre
        for i in [3, 5, 6]:
            if cut_ABC[r+1] == np.inf:
                A_mat = instant_mat(om['10'], om['1%s'%i], trans_mat_ABC)
                res_1 = (-np.linalg.inv(trans_mat_ABC[:-2,:-2])@(A_mat[:-2,:-2]))[om['10']][:,om['1%s'%i]]
                A_mat = instant_mat(om['10'], om['7%s'%i], trans_mat_ABC)
                res_2 = (-np.linalg.inv(trans_mat_ABC[:-2,:-2])@(A_mat[:-2,:-2]))[om['10']][:,om['7%s'%i]]
                A_mat_1 = instant_mat(om['10'], om['70'], trans_mat_ABC)
                A_mat_2 = instant_mat(om['70'], om['7%s'%i], trans_mat_ABC)
                C_mat_upper =  np.concatenate((trans_mat_ABC[:-2,:-2], A_mat_1[:-2,:-2]), axis = 1)
                C_mat_lower = np.concatenate((np.zeros((201,201)), trans_mat_ABC[:-2,:-2]), axis = 1)
                C_mat = np.concatenate((C_mat_upper, C_mat_lower), axis = 0)
                res_3 = ((-np.linalg.inv(C_mat)[0:201,-201:])@(A_mat_2[:-2,:-2]))[om['10']][:,om['7%s'%i]]
                for l in range(n_int_AB):
                    cond = [i == ((0, l), 'D') for i in names_tab_AB]
                    pi = pi_ABC[cond]
                    ii = dct_num[i]
                    tab.append([(0, l, L), (ii, r, R), (pi@p_ABC@res_1).sum()+(pi@p_ABC@sum([res_2, res_3])).sum()])
                    tab.append([(ii, r, R), (0, l, L), tab[-1][2]])
            else:
                omega_lst = ['10', '1%s'%i, '17', '70', '7%s'%i, '77']
                iter_lst = []
                for y in range(1, len(omega_lst)):
                    for z in range(y+1, len(omega_lst)):
                        if int(omega_lst[z][0]) < int(omega_lst[y][0]):
                            continue
                        elif int(omega_lst[z][1]) < int(omega_lst[y][1]):
                            continue
                        elif (int(omega_lst[z][1])-int(omega_lst[y][1]))==7:
                            continue
                        elif omega_lst[y][1]=='7':
                            continue
                        tup = (om['%s'%(omega_lst[0],)], 
                                om['%s'%(omega_lst[y],)],
                                om['%s'%(omega_lst[z],)])
                        iter_lst.append(tup)
                iterable = [(trans_mat_ABC, tup, om['10'], om['77'], cut_ABC[r+1]-cut_ABC[r]) for tup in iter_lst]
                res_tot = [vanloan_2(*x) for x in iterable]
                for l in range(n_int_AB):
                    cond = [i == ((0, l), 'D') for i in names_tab_AB]
                    pi = pi_ABC[cond]
                    ii = dct_num[i]
                    tab.append([(0, l, L), (ii, r, R), (pi@p_ABC@sum(res_tot)).sum()])
                    tab.append([(ii, r, R), (0, l, L), tab[-1][2]])
    elif r == R < L:
        omegas = [omega_tot_ABC]+[om['10']]*R
        p_ABC_start_pre = get_ABC_precomp(pr, omegas, list(range(R)))
        # Get right shapes for first interval
        p_ABC_start = p_ABC_start_pre[:,om['10']] if R==0 else p_ABC_start_pre
        omegas = [om['17']]*(L-R)+[om['77']]
        p_ABC_end = get_ABC_precomp(pr, omegas, list(range(R+1, L+int(cut_ABC[L+1]!=np.inf))))
        for i in [3, 5, 6]:
            A_mat = instant_mat(om['10'], om['1%s'%i], trans_mat_ABC)
            C_mat_upper = np.concatenate((trans_mat_ABC, A_mat), axis = 1)
            C_mat_lower = np.concatenate((np.zeros((203,203)), trans_mat_ABC), axis = 1)
            C_mat = np.concatenate((C_mat_upper, C_mat_lower), axis = 0)
            res = (expm(C_mat*(cut_ABC[r+1]-cut_ABC[r]))[:203,-203:])[om['10']][:,om['17']]
            for l in range(n_int_AB):
                cond = [i == ((0, l), 'D') for i in names_tab_AB]
                pi = pi_ABC[cond]
                ii = dct_num[i]
                tab.append([(0, l, L), (ii, r, R), (pi@p_ABC_start@res@p_ABC_end).sum()])
                tab.append([(ii, r, R), (0, l, L), tab[-1][2]])
    # end = time.time()
    # print("((0, {}, {}) -> (i, {}, {})) = {}".format('l', L, r, R, end - start))
    return tab

def pool_ABC(l, L, r, R):
    pi, om, omega_tot_ABC, pr, cut_ABC, dct_num, trans_mat_ABC = shared_data
    tab = []
    # starttim =  time.time()
    if l < L < r < R:
        times_ABC = get_times(cut_ABC, [0, l, l+1, L, L+1, r, r+1, R, R+1])
        for i in [3, 5, 6]:
            for j in [3, 5, 6]:
                # 00 -> i0 -> 70 -> 7j -> 77
                omegas = [omega_tot_ABC]+[om['00']]*l+[om['%s0'%i]]*(L-l)+[om['70']]*(r-L)+[om['7%s'%j]]*(R-r)+[om['77']]
                p_ABC = get_ABC_precomp(pr, omegas, list(range(R+int(cut_ABC[R+1]!=np.inf))))
                ii = dct_num[i]
                jj = dct_num[j]
                tab.append([(ii, l, L), (jj, r, R), (pi@p_ABC).sum()])
                tab.append([(jj, r, R), (ii, l, L), tab[-1][2]])
    elif l < L == r < R:
        times_ABC = get_times(cut_ABC, [0, l, l+1, L, L+1, R, R+1])
        for i in [3, 5, 6]:
            for j in [3, 5, 6]:
                # 00 -> i0 -> 7j -> 77
                omegas = [omega_tot_ABC]+[om['00']]*l+[om['%s0'%i]]*(L-l)+[om['7%s'%j]]*(R-L)+[om['77']]
                p_ABC = get_ABC_precomp(pr, omegas, list(range(R+int(cut_ABC[R+1]!=np.inf))))
                ii = dct_num[i]
                jj = dct_num[j]
                tab.append([(ii, l, L), (jj, r, R), (pi@p_ABC).sum()])
                tab.append([(jj, r, R), (ii, l, L), tab[-1][2]])
    elif l == r < L < R:
        times_ABC = get_times(cut_ABC, [0, l, l+1, L, L+1, R, R+1])
        for i in [3, 5, 6]:
            for j in [3, 5, 6]:
                # 00 -> ij -> 7j -> 77
                omegas = [omega_tot_ABC]+[om['00']]*l+[om['%s%s'%(i,j)]]*(L-l)+[om['7%s'%j]]*(R-L)+[om['77']]
                p_ABC = get_ABC_precomp(pr, omegas, list(range(R+int(cut_ABC[R+1]!=np.inf))))
                ii = dct_num[i]
                jj = dct_num[j]
                tab.append([(ii, l, L), (jj, r, R), (pi@p_ABC).sum()])
                tab.append([(jj, r, R), (ii, l, L), tab[-1][2]])
    elif l < r < L < R:
        times_ABC = get_times(cut_ABC, [0, l, l+1, r, r+1, L, L+1, R, R+1])
        for i in [3, 5, 6]:
            for j in [3, 5, 6]:
                # 00 -> i0 -> ij -> 7j -> 77
                omegas = [omega_tot_ABC]+[om['00']]*l+[om['%s0'%i]]*(r-l)+[om['%s%s'%(i,j)]]*(L-r)+[om['7%s'%j]]*(R-L)+[om['77']]
                p_ABC = get_ABC_precomp(pr, omegas, list(range(R+int(cut_ABC[R+1]!=np.inf))))
                ii = dct_num[i]
                jj = dct_num[j]
                tab.append([(ii, l, L), (jj, r, R), (pi@p_ABC).sum()])
                tab.append([(jj, r, R), (ii, l, L), tab[-1][2]])
    elif r < l < L < R:
        times_ABC = get_times(cut_ABC, [0, r, r+1, l, l+1, L, L+1, R, R+1])
        for i in [3, 5, 6]:
            for j in [3, 5, 6]:
                # 00 -> 0j -> ij -> 7j -> 77
                omegas = [omega_tot_ABC]+[om['00']]*r+[om['0%s'%j]]*(l-r)+[om['%s%s'%(i,j)]]*(L-l)+[om['7%s'%j]]*(R-L)+[om['77']]
                p_ABC = get_ABC_precomp(pr, omegas, list(range(R+int(cut_ABC[R+1]!=np.inf))))
                ii = dct_num[i]
                jj = dct_num[j]
                tab.append([(ii, l, L), (jj, r, R), (pi@p_ABC).sum()])
                tab.append([(jj, r, R), (ii, l, L), tab[-1][2]])
    elif l == r < L == R:
        times_ABC = get_times(cut_ABC, [0, l, l+1, L, L+1])
        for i in [3, 5, 6]:
            for j in [3, 5, 6]:
                # 00 -> ij -> 77
                omegas = [omega_tot_ABC]+[om['00']]*l+[om['%s%s'%(i,j)]]*(L-l)+[om['77']]
                p_ABC = get_ABC_precomp(pr, omegas, list(range(R+int(cut_ABC[R+1]!=np.inf))))
                ii = dct_num[i]
                jj = dct_num[j] 
                tab.append([(ii, l, L), (jj, r, R), (pi@p_ABC).sum()])
    elif l < r < L == R:
        times_ABC = get_times(cut_ABC, [0, l, l+1, r, r+1, L, L+1])
        for i in [3, 5, 6]:
            for j in [3, 5, 6]:
                # 00 -> i0 -> ij -> 77
                omegas = [omega_tot_ABC]+[om['00']]*l+[om['%s0'%i]]*(r-l)+[om['%s%s'%(i,j)]]*(R-r)+[om['77']]
                p_ABC = get_ABC_precomp(pr, omegas, list(range(R+int(cut_ABC[R+1]!=np.inf))))
                ii = dct_num[i]
                jj = dct_num[j]
                tab.append([(ii, l, L), (jj, r, R), (pi@p_ABC).sum()])
                tab.append([(jj, r, R), (ii, l, L), tab[-1][2]])
    elif l == r == L == R:
        omegas = [omega_tot_ABC]+[om['00']]*R
        p_ABC_pre = get_ABC_precomp(pr, omegas, list(range(R)))
        start = p_ABC_pre[:,om['00']] if L==0 else p_ABC_pre
        for i in [3, 5, 6]:
            for j in [3, 5, 6]:
                res_tot = 0
                if cut_ABC[r+1] == np.inf:
                    # 00 -> ij
                    A_mat = instant_mat(om['00'], om['%s%s'%(i,j)], trans_mat_ABC)
                    res_1 = (-np.linalg.inv(trans_mat_ABC[:-2,:-2])@(A_mat[:-2,:-2]))[om['00']][:,om['%s%s'%(i,j)]]
                    # 00 -> i0 -> ij
                    A_mat_1 = instant_mat(om['00'], om['%s0'%i], trans_mat_ABC)
                    A_mat_2 = instant_mat(om['%s0'%i], om['%s%s'%(i,j)], trans_mat_ABC)
                    C_mat_upper =  np.concatenate((trans_mat_ABC[:-2,:-2], A_mat_1[:-2,:-2]), axis = 1)
                    C_mat_lower = np.concatenate((np.zeros((201,201)), trans_mat_ABC[:-2,:-2]), axis = 1)
                    C_mat = np.concatenate((C_mat_upper, C_mat_lower), axis = 0)
                    res_2 = ((-np.linalg.inv(C_mat)[0:201,-201:])@(A_mat_2[:-2,:-2]))[om['00']][:,om['%s%s'%(i,j)]]
                    # 00 -> 0j -> ij
                    A_mat_1 = instant_mat(om['00'], om['0%s'%j], trans_mat_ABC)
                    A_mat_2 = instant_mat(om['0%s'%j], om['%s%s'%(i,j)], trans_mat_ABC)
                    C_mat_upper =  np.concatenate((trans_mat_ABC[:-2,:-2], A_mat_1[:-2,:-2]), axis = 1)
                    C_mat_lower = np.concatenate((np.zeros((201,201)), trans_mat_ABC[:-2,:-2]), axis = 1)
                    C_mat = np.concatenate((C_mat_upper, C_mat_lower), axis = 0)
                    res_3 = ((-np.linalg.inv(C_mat)[0:201,-201:])@(A_mat_2[:-2,:-2]))[om['00']][:,om['%s%s'%(i,j)]]
                    # 00 -> 0j -> 07 -> i7
                    A_mat_1 = instant_mat(om['00'], om['0%s'%j], trans_mat_ABC)
                    A_mat_2 = instant_mat(om['0%s'%j], om['07'], trans_mat_ABC)
                    A_mat_3 = instant_mat(om['07'], om['%s7'%i], trans_mat_ABC)
                    C_mat_upper  =  np.concatenate((trans_mat_ABC[:-2,:-2], A_mat_1[:-2,:-2], np.zeros((201,201))), axis = 1)
                    C_mat_middle =  np.concatenate((np.zeros((201,201)), trans_mat_ABC[:-2,:-2], A_mat_2[:-2,:-2]), axis = 1)
                    C_mat_lower  =  np.concatenate((np.zeros((201,201)),np.zeros((201,201)), trans_mat_ABC[:-2,:-2]), axis = 1)
                    C_mat = np.concatenate((C_mat_upper, C_mat_middle, C_mat_lower), axis = 0)
                    res_4 = ((-np.linalg.inv(C_mat)[0:201,-201:])@(A_mat_3[:-2,:-2]))[om['00']][:,om['%s7'%i]]
                    # 00 -> i0 -> 70 -> 7j
                    A_mat_1 = instant_mat(om['00'], om['%s0'%i], trans_mat_ABC)
                    A_mat_2 = instant_mat(om['%s0'%i], om['70'], trans_mat_ABC)
                    A_mat_3 = instant_mat(om['70'], om['7%s'%j], trans_mat_ABC)
                    C_mat_upper  =  np.concatenate((trans_mat_ABC[:-2,:-2], A_mat_1[:-2,:-2], np.zeros((201,201))), axis = 1)
                    C_mat_middle =  np.concatenate((np.zeros((201,201)), trans_mat_ABC[:-2,:-2], A_mat_2[:-2,:-2]), axis = 1)
                    C_mat_lower  =  np.concatenate((np.zeros((201,201)), np.zeros((201,201)), trans_mat_ABC[:-2,:-2]), axis = 1)
                    C_mat = np.concatenate((C_mat_upper, C_mat_middle, C_mat_lower), axis = 0)
                    res_5 = ((-np.linalg.inv(C_mat)[0:201,-201:])@(A_mat_3[:-2,:-2]))[om['00']][:,om['7%s'%j]]
                    # 00 -> 0j -> j7
                    A_mat_1 = instant_mat(om['00'], om['0%s'%j], trans_mat_ABC)
                    A_mat_2 = instant_mat(om['0%s'%j], om['%s7'%i], trans_mat_ABC)
                    C_mat_upper =  np.concatenate((trans_mat_ABC[:-2,:-2], A_mat_1[:-2,:-2]), axis = 1)
                    C_mat_lower = np.concatenate((np.zeros((201,201)), trans_mat_ABC[:-2,:-2]), axis = 1)
                    C_mat = np.concatenate((C_mat_upper, C_mat_lower), axis = 0)
                    res_6 = ((-np.linalg.inv(C_mat)[0:201,-201:])@(A_mat_2[:-2,:-2]))[om['00']][:,om['%s7'%i]]
                    # 00 -> i0 -> 7j
                    A_mat_1 = instant_mat(om['00'], om['%s0'%i], trans_mat_ABC)
                    A_mat_2 = instant_mat(om['%s0'%i], om['7%s'%j], trans_mat_ABC)
                    C_mat_upper =  np.concatenate((trans_mat_ABC[:-2,:-2], A_mat_1[:-2,:-2]), axis = 1)
                    C_mat_lower = np.concatenate((np.zeros((201,201)), trans_mat_ABC[:-2,:-2]), axis = 1)
                    C_mat = np.concatenate((C_mat_upper, C_mat_lower), axis = 0)
                    res_7 = ((-np.linalg.inv(C_mat)[0:201,-201:])@(A_mat_2[:-2,:-2]))[om['00']][:,om['7%s'%j]]
                    # Sum results
                    res_tot += (pi@start@sum([res_1, res_2, res_3])).sum()
                    res_tot += (pi@start@sum([res_4, res_6])).sum()
                    res_tot += (pi@start@sum([res_5, res_7])).sum()
                    ii = dct_num[i]
                    jj = dct_num[j]
                    tab.append([(ii, l, L), (jj, r, R), res_tot])
                else:
                    iter_lst = []
                    tup = (om['00'], om['%s0'%i], om['7%s'%j], om['77'])
                    iter_lst.append(tup)
                    tup = (om['00'], om['0%s'%j], om['%s7'%i], om['77'])
                    iter_lst.append(tup)
                    for y in ['%s%s'%(i,j)]:
                        for z in ['%s7'%i, '7%s'%j]:
                            tup = (om['00'], 
                               om['%s'%(y,)],
                               om['%s'%(z,)],
                               om['77'])
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
                                tup = (om['00'], 
                                       om['%s'%(y,)],
                                       om['%s'%(z,)],
                                       om['%s'%(v,)])
                                iter_lst.append(tup)
                    iterable = [(trans_mat_ABC, tup, om['00'], om['77'],cut_ABC[r+1]-cut_ABC[r]) for tup in iter_lst]
                    res_iter = [vanloan_3(*x) for x in iterable]
                    res_tot += (pi@start@sum(res_iter)).sum()
                    res_test = vanloan_2(
                        trans_mat_ABC, 
                        (om['00'], om['%s%s'%(i,j)], om['77']),
                        om['00'], om['77'],
                        cut_ABC[r+1]-cut_ABC[r]
                    )
                    res_tot += (pi@start@res_test).sum()
                    ii = dct_num[i]
                    jj = dct_num[j]
                    tab.append([(ii, l, L), (jj, r, R), res_tot])
    elif l == L < r == R:
        omegas = [omega_tot_ABC]+[om['00']]*L
        p_ABC_pre = get_ABC_precomp(pr, omegas, list(range(L)))
        start = p_ABC_pre[:,om['00']] if L==0 else p_ABC_pre
        omegas = [om['70']]*(R-L)
        end = get_ABC_precomp(pr, omegas, list(range(L+1, R)))
        for i in [3, 5, 6]:
            res_1 = vanloan_1(
                trans_mat_ABC, 
                (om['00'], om['%s0'%i]),
                om['00'], om['70'], cut_ABC[l+1]-cut_ABC[l])
            for j in [3, 5, 6]:
                # 00 -> 0i -> 70 -> 7j
                if cut_ABC[r+1] == np.inf:
                    A_mat = instant_mat(om['70'], om['7%s'%j], trans_mat_ABC)
                    res_2 = (-np.linalg.inv(trans_mat_ABC[:-2,:-2])@(A_mat[:-2,:-2]))[om['70']][:,om['7%s'%j]]
                    ii = dct_num[i]
                    jj = dct_num[j]
                    tab.append([(ii, l, L), (jj, r, R), (pi@start@res_1@end@res_2).sum()])
                    tab.append([(jj, r, R), (ii, l, L), tab[-1][2]])
                else:
                    res_2 = vanloan_1(
                        trans_mat_ABC,
                        (om['70'], om['7%s'%j]),
                        om['70'], om['77'], cut_ABC[r+1]-cut_ABC[r])
                    ii = dct_num[i]
                    jj = dct_num[j]
                    tab.append([(ii, l, L), (jj, r, R), (pi@start@res_1@end@res_2).sum()])
                    tab.append([(jj, r, R), (ii, l, L), tab[-1][2]])
    elif l == L < r < R:
        for j in [3, 5, 6]:
            omegas = [omega_tot_ABC]+[om['00']]*L
            p_ABC_pre = get_ABC_precomp(pr, omegas, list(range(L)))
            start = p_ABC_pre[:,om['00']] if L==0 else p_ABC_pre
            omegas = [om['70']]*(r-L)+[om['7%s'%j]]*(R-r)+[om['77']]
            end = get_ABC_precomp(pr, omegas, list(range(L+1, R+int(cut_ABC[R+1]!=np.inf))))
            for i in [3, 5, 6]:
                # 00 -> i0 -> 70 -> 7j
                res = vanloan_1(
                        trans_mat_ABC,
                        (om['00'], om['%s0'%i]),
                        om['00'], om['70'],
                        cut_ABC[l+1]-cut_ABC[l])
                ii = dct_num[i]
                jj = dct_num[j]
                tab.append([(ii, l, L), (jj, r, R), (pi@start@res@end).sum()])
                tab.append([(jj, r, R), (ii, l, L), tab[-1][2]])
    elif l == L == r < R:
        omegas = [omega_tot_ABC]+[om['00']]*L
        p_ABC_pre = get_ABC_precomp(pr, omegas, list(range(L)))
        start = p_ABC_pre[:,om['00']] if L==0 else p_ABC_pre
        for j in [3, 5, 6]:
            for i in [3, 5, 6]:
                omegas = [om['7%s'%j]]*(R-L)+[om['77']]
                end = get_ABC_precomp(pr, omegas, list(range(L+1, R+int(cut_ABC[R+1]!=np.inf))))
                omega_lst = ['00', '%s0'%i, '0%s'%j, '%s%s'%(i,j), '70', '7%s'%j]
                iter_lst = []
                for y in range(1, len(omega_lst)):
                    for z in range(y+1, len(omega_lst)):
                        if int(omega_lst[z][0]) < int(omega_lst[y][0]):
                            continue
                        elif int(omega_lst[z][1]) < int(omega_lst[y][1]):
                            continue
                        elif (int(omega_lst[z][0])-int(omega_lst[y][0]))==7:
                            continue
                        elif omega_lst[y][0]=='7':
                            continue
                        tup = (om['%s'%(omega_lst[0],)], 
                               om['%s'%(omega_lst[y],)],
                               om['%s'%(omega_lst[z],)])
                        iter_lst.append(tup)
                iterable = [(trans_mat_ABC, tup, om['00'], om['7%s'%j], cut_ABC[r+1]-cut_ABC[r]) for tup in iter_lst]
                res_tot = [vanloan_2(*x) for x in iterable]
                ii = dct_num[i]
                jj = dct_num[j]
                res_tot = (pi@start@sum(res_tot)@end).sum()
                tab.append([(ii, l, L), (jj, r, R), res_tot])
                tab.append([(jj, r, R), (ii, l, L), res_tot])
    elif l < L == r == R:
        for i in [3, 5, 6]:
            omegas = [omega_tot_ABC]+[om['00']]*l+[om['%s0'%i]]*(L-l)
            p_ABC = get_ABC_precomp(pr, omegas, list(range(L)))
            for j in [3, 5, 6]:
                if cut_ABC[L+1] == np.inf:
                    # 00 -> i0 -> 1j
                    A_mat = instant_mat(om['%s0'%i], om['1%s'%j], trans_mat_ABC)
                    res_1 = (-np.linalg.inv(trans_mat_ABC[:-2,:-2])@(A_mat[:-2,:-2]))[om['%s0'%i]][:,om['1%s'%j]]
                    # 00 -> i0 -> 7j
                    A_mat = instant_mat(om['%s0'%i], om['7%s'%j], trans_mat_ABC)
                    res_2 = (-np.linalg.inv(trans_mat_ABC[:-2,:-2])@(A_mat[:-2,:-2]))[om['%s0'%i]][:,om['7%s'%j]]
                    # 00 -> i0 -> 70 -> 7j
                    A_mat_1 = instant_mat(om['%s0'%i], om['70'], trans_mat_ABC)
                    A_mat_2 = instant_mat(om['70'], om['7%s'%j], trans_mat_ABC)
                    C_mat_upper =  np.concatenate((trans_mat_ABC[:-2,:-2], A_mat_1[:-2,:-2]), axis = 1)
                    C_mat_lower = np.concatenate((np.zeros((201,201)), trans_mat_ABC[:-2,:-2]), axis = 1)
                    C_mat = np.concatenate((C_mat_upper, C_mat_lower), axis = 0)
                    res_3 = ((-np.linalg.inv(C_mat)[0:201,-201:])@(A_mat_2[:-2,:-2]))[om['%s0'%i]][:,om['7%s'%j]]
                    # Sum of results
                    ii = dct_num[i]
                    jj = dct_num[j]
                    tab.append([(ii, l, L), (jj, r, R), (pi@p_ABC@res_1).sum()+(pi@p_ABC@sum([res_2, res_3])).sum()])
                    tab.append([(jj, r, R), (ii, l, L), tab[-1][2]])
                else:
                    omega_lst = ['%s0'%i, '%s%s'%(i,j), '%s7'%i, '70', '7%s'%j, '77']
                    iter_lst = []
                    for y in range(1, len(omega_lst)):
                        for z in range(y+1, len(omega_lst)):
                            tup = (om['%s'%(omega_lst[0],)], 
                                   om['%s'%(omega_lst[y],)],
                                   om['%s'%(omega_lst[z],)])
                            iter_lst.append(tup)
                    iterable = [(trans_mat_ABC, tup, om['%s0'%i], om['77'], cut_ABC[r+1]-cut_ABC[r]) for tup in iter_lst]
                    res_tot = [vanloan_2(*x) for x in iterable]
                    res_tot = (pi@p_ABC@sum(res_tot)).sum()
                    ii = dct_num[i]
                    jj = dct_num[j]
                    tab.append([(ii, l, L), (jj, r, R), res_tot])
                    tab.append([(jj, r, R), (ii, l, L), res_tot])
    elif l < L < r == R:
        for i in [3, 5, 6]:
            omegas = [omega_tot_ABC]+[om['00']]*l+[om['%s0'%i]]*(L-l)+[om['70']]*(r-L)
            p_ABC = get_ABC_precomp(pr, omegas, list(range(r)))
            for j in [3, 5, 6]:
                if cut_ABC[r+1] != np.inf:
                    res = vanloan_1(
                        trans_mat_ABC,
                        (om['70'], om['7%s'%j]),
                        om['70'], om['77'],
                        cut_ABC[r+1]-cut_ABC[r])
                    ii = dct_num[i]
                    jj = dct_num[j]
                    tab.append([(ii, l, L), (jj, r, R), (pi@p_ABC@res).sum()])
                    tab.append([(jj, r, R), (ii, l, L), tab[-1][2]])
                else:
                    A_mat = instant_mat(om['70'], om['7%s'%j], trans_mat_ABC)
                    res = (-np.linalg.inv(trans_mat_ABC[:-2,:-2])@(A_mat[:-2,:-2]))[om['70']][:,om['7%s'%j]]
                    ii = dct_num[i]
                    jj = dct_num[j]
                    tab.append([(ii, l, L), (jj, r, R), (pi@p_ABC@res).sum()])
                    tab.append([(jj, r, R), (ii, l, L), tab[-1][2]])
    elif r < l == L < R:
        for j in [3, 5, 6]:
            omegas = [omega_tot_ABC]+[om['00']]*r+[om['0%s'%j]]*(l-r)
            start = get_ABC_precomp(pr, omegas, list(range(l)))
            omegas = [om['7%s'%j]]*(R-l)+[om['77']]
            end = get_ABC_precomp(pr, omegas, list(range(l+1, R+int(cut_ABC[R+1]!=np.inf))))
            for i in [3, 5, 6]:
                # 00 -> 0j -> ij -> 7j -> 77
                res = vanloan_1(
                    trans_mat_ABC,
                    (om['0%s'%j], om['%s%s'%(i, j)]),
                    om['0%s'%j], om['7%s'%j],
                    cut_ABC[l+1]-cut_ABC[l])
                ii = dct_num[i]
                jj = dct_num[j]
                tab.append([(ii, l, L), (jj, r, R), (pi@start@res@end).sum()])
                tab.append([(jj, r, R), (ii, l, L), tab[-1][2]])
    # endtim = time.time()
    # print("((i, {}, {}) -> (j, {}, {})) = {}".format(l, L, r, R, endtim - starttim))
    return tab

def init_worker_AB(pi_ABC, om, omega_tot_ABC, pr, cut_ABC, dct_num, trans_mat_ABC, n_int_AB, names_tab_AB):
    global shared_data
    shared_data = (pi_ABC, om, omega_tot_ABC, pr, cut_ABC, dct_num, trans_mat_ABC, n_int_AB, names_tab_AB)

def init_worker_ABC(pi, om, omega_tot_ABC, pr, cut_ABC, dct_num, trans_mat_ABC):
    global shared_data
    shared_data = (pi, om, omega_tot_ABC, pr, cut_ABC, dct_num, trans_mat_ABC)