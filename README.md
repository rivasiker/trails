# TRAILS

TRAILS is a coalescent hidden Markov model (HMM) that reconstructs demographic parameters (ancestral Ne and split times) for three species and an outgroup. After model fitting, TRAILS can also perform posterior decoding to identify incomplete lineage sorting fragments. The original method is described in [Rivas-González et al. 2024](https://doi.org/10.1371/journal.pgen.1010836).

## Instalation

The necessary packages for running trails can be installed through conda:

```bash
conda create -n trails -c conda-forge python=3.8 "ray-default" pandas numba numpy scipy biopython
conda activate trails
```

The TRAILS python package can be installed from pip:

```bash
pip install trails-rivasiker
```

The development version can be installed by cloning this repository and installing the python package locally:

```bash
git clone https://github.com/rivasiker/trails.git
cd trails
pip3 install -e .
```

## Model optimization

Demographic parameters can be optimized using the `optimizer` function:

```python
# Load functions
from trails.cutpoints import cutpoints_ABC
from trails.optimizer import optimizer
from trails.read_data import maf_parser

# Define fixed parameters
n_int_AB = 3
n_int_ABC = 3
mu = 2e-8
method = 'Nelder-Mead'

# Define optimized parameters
N_AB = 25000*2*mu
N_ABC = 25000*2*mu
t_1 = 240000*mu
t_A = t_1
t_B = t_1
t_2 = 40000*mu
t_C = t_1+t_2
t_3 = 800000*mu
t_upper = t_3-cutpoints_ABC(n_int_ABC,  1/N_ABC)[-2]
t_out = t_1+t_2+t_3+2*N_ABC
r = 1e-8/mu

# Define initial parameters
t_init_A = t_A
t_init_B = t_B
t_init_C = t_C
t_init_2 = t_2
t_init_upper = t_upper
N_init_AB = N_AB
N_init_ABC = N_ABC
r_init = r

# Define parameter boundaries as dictionary, with entries being
# 'param_name': [initial_value, lower_bound, upper_bound]
dct = {
    't_A':     [t_init_A,     t_A/10, t_A*10], 
    't_B':     [t_init_B,     t_B/10, t_B*10], 
    't_C':     [t_init_C,     t_C/10, t_C*10], 
    't_2':     [t_init_2,     t_2/10, t_2*10], 
    't_upper': [t_init_upper, t_upper/10, t_upper*10], 
    'N_AB':    [N_init_AB,    N_AB/10,  N_AB*10], 
    'N_ABC':   [N_init_ABC,   N_ABC/10,  N_ABC*10], 
    'r':       [r_init,       r/10,  r*10]
    }

# Define fixed parameter values
dct2 = {'n_int_AB':n_int_AB, 'n_int_ABC':n_int_ABC}

# Define list of species
sp_lst = ['species1','species2','species3','outgroup']
# Read MAF alignment
alignment = maf_parser('path_to_alignment/alignment.maf', sp_lst)

# Run optimization
res = optimizer(
    optim_params = dct, 
    fixed_params = dct2, 
    V_lst = alignment, 
    res_name = 'optimization.csv',
    method = method, 
    header = True
    )
```

## Transition and emission probabilities

Transition and emission probabilities can be defined after fixing the demographic parameters using the `trans_emiss_calc` function:

```python
from trails.optimizer import trans_emiss_calc
import pandas as pd

transitions, emissions, starting, hidden_states, observed_states = trans_emiss_calc(
    t_1, t_1, t_C, t_2, t_upper, t_out,
    N_AB, N_ABC, r, n_int_AB, n_int_ABC)
```

Print transition probability matrix:

```python
df_transitions = pd.DataFrame(transitions).melt(ignore_index = False).reset_index(level=0)
df_transitions.columns = ['from', 'to', 'value']
df_transitions['from'] = [hidden_states[i] for i in df_transitions['from']]
df_transitions['to'] = [hidden_states[i] for i in df_transitions['to']]
df_transitions = df_transitions.sort_values(['from', 'to']).reset_index(drop=True)
print(df_transitions)
```

```
          from         to         value
0    (0, 0, 0)  (0, 0, 0)  9.986414e-01
1    (0, 0, 0)  (0, 0, 1)  2.717854e-04
2    (0, 0, 0)  (0, 0, 2)  2.717854e-04
3    (0, 0, 0)  (0, 1, 0)  1.831404e-04
4    (0, 0, 0)  (0, 1, 1)  5.365867e-08
..         ...        ...           ...
724  (3, 2, 2)  (3, 0, 1)  5.593765e-08
725  (3, 2, 2)  (3, 0, 2)  2.437075e-04
726  (3, 2, 2)  (3, 1, 1)  2.767875e-08
727  (3, 2, 2)  (3, 1, 2)  2.706694e-04
728  (3, 2, 2)  (3, 2, 2)  9.974707e-01

[729 rows x 3 columns]
```

Print emission probability matrix:

```python
df_emissions = pd.DataFrame(emissions).melt(ignore_index = False).reset_index(level=0)
df_emissions.columns = ['hidden', 'observed', 'value']
df_emissions['hidden'] = [hidden_states[i] for i in df_emissions['hidden']]
df_emissions['observed'] = [observed_states[i] for i in df_emissions['observed']]
df_emissions = df_emissions.sort_values(['hidden', 'observed']).reset_index(drop=True)
print(df_emissions)
```

```
         hidden observed     value
0     (0, 0, 0)     AAAA  0.236476
1     (0, 0, 0)     AAAC  0.003147
2     (0, 0, 0)     AAAG  0.003147
3     (0, 0, 0)     AAAT  0.003147
4     (0, 0, 0)     AACA  0.000458
...         ...      ...       ...
6907  (3, 2, 2)     TTGT  0.000553
6908  (3, 2, 2)     TTTA  0.002953
6909  (3, 2, 2)     TTTC  0.002953
6910  (3, 2, 2)     TTTG  0.002953
6911  (3, 2, 2)     TTTT  0.235440

[6912 rows x 3 columns]
```

## Posterior probability

Posterior probabilities can be calculated using the `post_prob_wrapper` function:

```python
from trails.optimizer import post_prob_wrapper

post_prob = post_prob_wrapper(transitions, emissions, starting, alignment)
```
