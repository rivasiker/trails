# TRAILS

Define demographic model and calculate transition and emission probabilities:

```python
from trails.optimizer import trans_emiss_calc
from trails.cutpoints import cutpoints_ABC
import pandas as pd

n_int_AB = 3
n_int_ABC = 3

N_AB = 50000
N_ABC = 50000
N_ref = N_ABC
t_1 = 160000
t_2 = 40000
t_3 = 800000
t_upper = t_3-cutpoints_ABC(n_int_ABC,  N_ref/N_ABC)[-2]*N_ref
r = 1e-8
mu = 2e-8

transitions, emissions, starting, hidden_states, observed_states = trans_emiss_calc(
    t_1, t_1, t_1+t_2, t_2, t_upper, t_1+t_2+t_3+2*N_ABC,
    N_AB, N_ABC,
    r, mu, mu, mu, mu, mu, mu, n_int_AB, n_int_ABC)
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
0    (0, 0, 0)  (0, 0, 0)  9.973359e-01
1    (0, 0, 0)  (0, 0, 1)  5.416001e-04
2    (0, 0, 0)  (0, 0, 2)  5.416001e-04
3    (0, 0, 0)  (0, 1, 0)  3.550184e-04
4    (0, 0, 0)  (0, 1, 1)  2.071043e-07
..         ...        ...           ...
724  (3, 2, 2)  (3, 0, 1)  2.218842e-07
725  (3, 2, 2)  (3, 0, 2)  4.840551e-04
726  (3, 2, 2)  (3, 1, 1)  1.101587e-07
727  (3, 2, 2)  (3, 1, 2)  5.393714e-04
728  (3, 2, 2)  (3, 2, 2)  9.949861e-01

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
0     (0, 0, 0)     AAAA  0.226306
1     (0, 0, 0)     AAAC  0.005847
2     (0, 0, 0)     AAAG  0.005847
3     (0, 0, 0)     AAAT  0.005847
4     (0, 0, 0)     AACA  0.000759
...         ...      ...       ...
6907  (3, 2, 2)     TTGT  0.000936
6908  (3, 2, 2)     TTTA  0.005445
6909  (3, 2, 2)     TTTC  0.005445
6910  (3, 2, 2)     TTTG  0.005445
6911  (3, 2, 2)     TTTT  0.224319

[6912 rows x 3 columns]
```
