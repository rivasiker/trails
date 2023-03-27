# TRAILS

Define demographic model and calculate transition and emission probabilities:

```python
from trails.optimizer import trans_emiss_calc
from trails.cutpoints import cutpoints_ABC
import pandas as pd

n_int_AB = 3
n_int_ABC = 3
mu = 2e-8

N_AB = 25000*2*mu
N_ABC = 25000*2*mu
t_1 = 240000*mu
t_2 = 40000*mu
t_3 = 800000*mu
t_upper = t_3-cutpoints_ABC(n_int_ABC,  1/N_ABC)[-2]
t_out = t_1+t_2+t_3+2*N_ABC
r = 1e-8/mu

transitions, emissions, starting, hidden_states, observed_states = trans_emiss_calc(
    t_1, t_1, t_1+t_2, t_2, t_upper, t_out,
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
0     (0, 0, 0)     AAAA  0.236472
1     (0, 0, 0)     AAAC  0.003146
2     (0, 0, 0)     AAAG  0.003146
3     (0, 0, 0)     AAAT  0.003146
4     (0, 0, 0)     AACA  0.000459
...         ...      ...       ...
6907  (3, 2, 2)     TTGT  0.000905
6908  (3, 2, 2)     TTTA  0.002240
6909  (3, 2, 2)     TTTC  0.002240
6910  (3, 2, 2)     TTTG  0.002240
6911  (3, 2, 2)     TTTT  0.232358

[6912 rows x 3 columns]
```
