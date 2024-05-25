import numpy as np
from scipy.linalg import expm
from ast import literal_eval
from trails.cutpoints import cutpoints_AB, cutpoints_ABC
from trails.load_trans_mat import load_trans_mat, trans_mat_num
from trails.combine_states import combine_states
from trails.get_tab_introgression import get_tab_AB_introgression, get_tab_ABC_introgression
from trails.get_times import get_times
from trails.get_tab import precomp



def get_joint_prob_mat_introgression(
        t_A,    t_B,    t_AB,    t_C,    t_m,
        rho_A,  rho_B,  rho_AB,  rho_C,  rho_ABC, 
        coal_A, coal_B, coal_AB, coal_BC, coal_C, coal_ABC,
        m,
        n_int_AB, n_int_ABC,
        cut_AB = 'standard', cut_ABC = 'standard', 
        tmp_path = './'):
    """
    This is a wrapper function that unifies all the CTMCs to 
    get a matrix of joint probabilities for the HMM. 
    
    Parameters
    ----------
    t_A : float
        Time for the one-sequence CTMC of the first sequence (A)
    t_B : float
        Time for the one-sequence CTMC of the second sequence (B)
    t_AB : float
        Time for the two-sequence CTMC (AB)
    t_C : float
        Time for the one-sequence CTMC of the third sequence (C)
    t_m : float
        Time from the migration event to the first speciation event
    rho_A : float
        Recombination rate for the one-sequence CTMC of the first sequence (A)  
    rho_B : float
        Recombination rate for the one-sequence CTMC of the second sequence (B)  
    rho_AB : float
        Recombination rate for the two-sequence CTMC (AB)  
    rho_C : float
        Recombination rate for the one-sequence CTMC of the third sequence (C)    
    rho_ABC : float
        Recombination rate for the three-sequence CTMC (ABC)  
    coal_A : float
        Coalescent rate for the one-sequence CTMC of the first sequence (A)  
    coal_B : float
        Coalescent rate for the one-sequence CTMC of the second sequence (B)  
    coal_AB : float
        Coalescent rate for the two-sequence CTMC (AB)  
    coal_BC : float
        Coalescent rate for the two-sequence CTMC (BC)  
    coal_C : float
        Coalescent rate for the one-sequence CTMC of the third sequence (C)    
    coal_ABC : float
        Coalescent rate for the three-sequence CTMC (ABC)  
    m : float
        Probability that an individual migrates from B to C (backwards in time)
    n_int_AB : integer
        Number of intervals of the two-sequence CTMC (AB)
    n_int_ABC : integer
        Number of intervals of the three-sequence CTMC (ABC)
    """

    # Calculate cutpoints
    if isinstance(cut_AB, str):
        cut_AB = cutpoints_AB(n_int_AB, t_AB, coal_AB)
    intervals_AB = get_times(cut_AB, list(range(len(cut_AB))))
    intervals_BC = [intervals_AB[0]+t_m]+intervals_AB[1:]

    ####################################
    ### Load state-space information ###
    ####################################

    # Load string transition rate matrix and convert string names to actual lists
    (trans_mat_1, state_space_1) = load_trans_mat(1)
    state_space_A = [literal_eval(i) for i in state_space_1]
    (trans_mat_2, state_space_2) = load_trans_mat(2)
    state_space_AB = [literal_eval(i) for i in state_space_2]
    (trans_mat_3, state_space_3) = load_trans_mat(3)
    state_space_ABC = [literal_eval(i) for i in state_space_3]

    state_space_B = []
    for j in state_space_A:
        lst = []
        for i in j:
            one = 2 if i[0] == 1 else i[0]
            two = 2 if i[1] == 1 else i[1]
            lst.append((one, two))
        state_space_B.append(lst)

    state_space_C = []
    for j in state_space_A:
        lst = []
        for i in j:
            one = 4 if i[0] == 1 else i[0]
            two = 4 if i[1] == 1 else i[1]
            lst.append((one, two))
        state_space_C.append(lst)

    trans_mat_A = trans_mat_num(trans_mat_1, coal_A, rho_A)
    trans_mat_B = trans_mat_num(trans_mat_1, coal_B, rho_B)
    trans_mat_C = trans_mat_num(trans_mat_1, coal_C, rho_C)
    trans_mat_AB = trans_mat_num(trans_mat_2, coal_AB, rho_AB)
    pr_AB = precomp(trans_mat_AB, intervals_AB)
    trans_mat_BC = trans_mat_num(trans_mat_2, coal_BC, rho_AB)
    pr_BC = precomp(trans_mat_BC, intervals_BC)


    ##########################
    ### One-sequence CTMCs ###
    ##########################

    # Species A, from present to first speciation event
    final_A = expm(trans_mat_A*t_A)[0]
    # Species B, from present to migration event
    final_B = expm(trans_mat_B*t_B)[0]
    # Species C, from present to migration event
    final_C = expm(trans_mat_C*t_C)[0]
    # Species A, from present to second speciation event
    final_A_bis = expm(trans_mat_A*(t_A+t_AB))[0]
    # Species C, from present to second speciation event
    final_C_bis = expm(trans_mat_C*(t_C+t_m+t_AB))[0]

    # Split probabilities for the left and the right path
    # Left path: lineages do not migrate and will later mix with species A
    (state_space_B_left, final_B_left) = split_migration(state_space_B, final_B, m, 'left')
    # Right path: lineages migrate and are instantly mixed with species C
    (state_space_B_right, final_B_right) = split_migration(state_space_B, final_B, m, 'right')

    ##########################
    ### Two-sequence CTMCs ###
    ##########################

    #-#-# Right path, from migration event to second speciation event #-#-#

    # # Full lineages # #
    # This is when both the left and the right sites have B lineages

    # Combine the state space of the migrated B lineages with the state space of C
    (comb_BC_name_full, comb_BC_value_full) = combine_states(state_space_A, state_space_B_right[0:2], final_C, final_B_right[0:2])
    # Re-order states
    pi_BC_full = [comb_BC_value_full[comb_BC_name_full.index(i)] if i in comb_BC_name_full else 0 for i in state_space_2]
    # Obtain the correct state space for BC
    state_space_BC = [j.replace('1', '4') for j in state_space_2]
    state_space_BC = [j.replace('3', '6') for j in state_space_BC]
    state_space_BC = [literal_eval(i) for i in state_space_BC]
    state_space_BC = [sorted(i) for i in state_space_BC]

    # # Missing lineages # #
    # This is when at least one of either the left or the right sites are missing a B lineage

    # This is the transition matrix and state space for when lineages are missing
    (trans_mat_2_miss, state_space_2_miss) = load_trans_mat_miss()
    state_space_BC_miss = [literal_eval(i) for i in state_space_2_miss]
    trans_mat_BC_miss = trans_mat_num(trans_mat_2_miss, coal_BC, rho_AB)
    pr_BC_miss = precomp(trans_mat_BC_miss, intervals_BC)
    # Combine the state space of the migrated B lineages with the state space of C
    (comb_BC_name_miss, comb_BC_value_miss) = combine_states(state_space_C, state_space_B_right[2::], final_C, final_B_right[2::])
    # Re-order states
    pi_BC_miss = [comb_BC_value_miss[comb_BC_name_miss.index(i)] if i in comb_BC_name_miss else 0 for i in state_space_2_miss]


    #-#-# Left path, from migration event to second speciation event #-#-#

    # # Full lineages # #
    # This is when both the left and the right sites have B lineages

    # One-sequence CTMC from migration to first speciation event
    final_B_left_full = final_B_left[0:2] @ expm(trans_mat_B*t_m)
    # Combine the state space of the non-migrated B lineages with the state space of A
    (comb_AB_name_full, comb_AB_value_full) = combine_states(state_space_A, state_space_B_left[0:2], final_A, final_B_left_full)
    # Re-order states
    pi_AB_full = [comb_AB_value_full[comb_AB_name_full.index(i)] if i in comb_AB_name_full else 0 for i in state_space_2]
    state_space_AB = [literal_eval(i) for i in state_space_2]

    # # Missing lineages # #
    # This is when at least one of either the left or the right sites are missing a B lineage

    # This is the transition matrix and state space for when lineages are missing.
    # It is the same as for the right path, but with A instead of C.
    (trans_mat_4, state_space_4) = load_trans_mat_miss()
    trans_mat_AB_miss = trans_mat_num(trans_mat_4, coal_AB, rho_AB)
    pr_AB_miss = precomp(trans_mat_AB_miss, intervals_AB)
    state_space_4 = [i.replace('4', '1') for i in state_space_4]
    state_space_4 = [i.replace('6', '3') for i in state_space_4]
    state_space_AB_miss = [sorted(literal_eval(i)) for i in state_space_4]
    state_space_4 = [str(i) for i in state_space_AB_miss]
    # Combine the state space of the non-migrated B lineages with the state space of A
    (comb_AB_name_miss, comb_AB_value_miss) = combine_states(state_space_A, state_space_B_left[2::], final_A, final_B_left[2::])
    # Re-order states
    pi_AB_miss = [comb_AB_value_miss[comb_AB_name_miss.index(i)] if i in comb_AB_name_miss else 0 for i in state_space_4]

    tab, tab_names = get_tab_AB_introgression(
        state_space_AB, state_space_AB_miss, state_space_BC, state_space_BC_miss,
        state_space_A, state_space_C, state_space_ABC,
        pi_AB_full, pi_AB_miss, pi_BC_full, pi_BC_miss,
        final_A_bis, final_C_bis,
        pr_AB, pr_AB_miss, pr_BC, pr_BC_miss,
        n_int_AB
    )

    trans_mat_ABC = trans_mat_num(trans_mat_3, coal_ABC, rho_ABC)
    # Calculate cutpoints
    if isinstance(cut_ABC, str):
        cut_ABC = cutpoints_ABC(n_int_ABC, coal_ABC)
    joint_mat = get_tab_ABC_introgression(state_space_ABC, trans_mat_ABC, cut_ABC, tab, tab_names, n_int_AB, tmp_path)


    return joint_mat


def split_migration(state_space, prob_vec, m, direction):
    """
    This function splits the vector of final probabilities for species B given a certain 
    migration rate m. Based on the direction of choice, the resulting probabilities will 
    correspond to the left path (where lineages do not migrate and will later mix with 
    species A), or the right path (where lineages migrate and are instantly mixed with 
    species C).

    Parameters
    ----------
    state_space : list of lists of tuples
        The state space for a one-sequence CTMC
    prob_vec : numpy array
        The end probabilities of the one-sequence CTMC at the time of migration
    m : float
        Probability that a sequence migrates
    direction : string
        Left or right path
    """
    # Pr(O--  --O) = 1 - Pr(O--O)
    x = prob_vec[1]
    # Define new state space
    #         O--O          O--    --O           O--                    --O
    st = [state_space[0], state_space[1], [state_space[1][0]], [state_space[1][1]]]
    if direction == 'left':
        pr = np.array([(1-x)*(1-m), (1-m)**2*x, 1/2*(1-m)*m*x, 1/2*(1-m)*m*x])
    if direction == 'right':
        pr = np.array([(1-x)*m, x*m**2, 1/2*(1-m)*m*x, 1/2*(1-m)*m*x])
    return (st, pr)


def load_trans_mat_miss():
    """
    This function defines the state space and transition rate matrix for
    the right path when lineages are missing, i.e., when only the right
    or the left site of species B are mixed with the two sites of species C.  
    """
    mat = np.array(
        [
            ['0', 'R', '0', '0', 'C', '0', '0', '0', '0', '0'],
            ['C', '0', 'C', 'C', '0', '0', '0', '0', '0', '0'],
            ['0', 'R', '0', '0', 'C', '0', '0', '0', '0', '0'],
            ['0', '0', '0', '0', 'C', '0', '0', '0', '0', '0'],
            ['0', '0', '0', 'R', '0', '0', '0', '0', '0', '0'],
            ['0', '0', '0', '0', '0', '0', 'R', '0', '0', 'C'],
            ['0', '0', '0', '0', '0', 'C', '0', 'C', 'C', '0'],
            ['0', '0', '0', '0', '0', '0', 'R', '0', '0', 'C'],
            ['0', '0', '0', '0', '0', '0', '0', '0', '0', 'C'],
            ['0', '0', '0', '0', '0', '0', '0', '0', 'R', '0']
        ], 
        dtype=object)
    st = [
        '[(2, 0), (4, 4)]', 
        '[(0, 4), (2, 0), (4, 0)]', 
        '[(2, 4), (4, 0)]', 
        '[(0, 4), (6, 0)]',
        '[(6, 4)]',
        '[(0, 2), (4, 4)]', 
        '[(0, 2), (0, 4), (4, 0)]', 
        '[(0, 4), (4, 2)]', 
        '[(0, 6), (4, 0)]',
        '[(4, 6)]']
    return (mat, st)

def divide_starting_probs(ordered_pi_ABC, state_space_ABC):
    """
    This function divides the starting probabilities of the three-sequence
    CTMC according to the path taken, i.e., it identifies the states where sequences 
    have already coalesced at the left and/or the right sites, and it divides
    their starting probabilities based on this.  

    Parameters
    ----------
    ordered_pi_ABC : list of floats
        The starting probabilities of the three-sequence CTMC
    state_space_ABC : list of lists of tuples
        The ordered state space of the three-sequence CTMC
    """

    # Obtain all of the omegas
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

    # Create empty table for the joint probabilities
    tab = np.zeros((9, 203))
    # Create empty vector for the names of the states
    tab_names = []
    # Create accumulator for keeping track of the indices for the table
    acc = 0

    # Uncoalesced states (destined for deep coalesce at both sites)
    tab_names.append(('D', 'D')) 
    tmp_lst = om['00']
    tab[acc] = [ordered_pi_ABC[i] if i in tmp_lst else 0 for i in range(len(state_space_ABC))]
    acc += 1
    # Introgression states (already coalesced at both sites through right path)
    tab_names.append(((4, 0), (4, 0))) 
    tmp_lst = om['66']
    tab[acc] = [ordered_pi_ABC[i] if i in tmp_lst else 0 for i in range(len(state_space_ABC))]
    acc += 1
    # Shallow coalescence states (already coalesced at both sites through left path)
    tab_names.append(((0, 0), (0, 0))) 
    tmp_lst = om['33']
    tab[acc] = [ordered_pi_ABC[i] if i in tmp_lst else 0 for i in range(len(state_space_ABC))]
    acc += 1
    # Uncoalesced on left site, introgressed on right site
    tab_names.append(('D', (4, 0)))
    tmp_lst = om['06']
    tab[acc] = [ordered_pi_ABC[i] if i in tmp_lst else 0 for i in range(len(state_space_ABC))]
    acc += 1
    # Introgressed on left site, uncoalesced on right site
    tab_names.append(((4, 0), 'D'))
    tmp_lst = om['60']
    tab[acc] = [ordered_pi_ABC[i] if i in tmp_lst else 0 for i in range(len(state_space_ABC))]
    acc += 1
    # Uncoalesced on left site, shallow coalescence on right site
    tab_names.append(('D', (0, 0)))
    tmp_lst = om['03']
    tab[acc] = [ordered_pi_ABC[i] if i in tmp_lst else 0 for i in range(len(state_space_ABC))]
    acc += 1
    # Shallow coalescence on left site, uncoalesced on right site
    tab_names.append(((0, 0), 'D'))
    tmp_lst = om['30']
    tab[acc] = [ordered_pi_ABC[i] if i in tmp_lst else 0 for i in range(len(state_space_ABC))]
    acc += 1
    # Introgressed on left site, uncoalesced on right site
    tab_names.append(((4, 0), (0, 0)))
    tmp_lst = om['63']
    tab[acc] = [ordered_pi_ABC[i] if i in tmp_lst else 0 for i in range(len(state_space_ABC))]
    acc += 1
    # Uncoalesced on left site, introgressed on right site
    tab_names.append(((0, 0), (4, 0)))
    tmp_lst = om['36']
    tab[acc] = [ordered_pi_ABC[i] if i in tmp_lst else 0 for i in range(len(state_space_ABC))]
    acc += 1

    return (tab, tab_names)