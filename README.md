# ILSMC: basic example

```python
from ilsmc.get_joint_prob_mat import get_joint_prob_mat
import pandas as pd

arr = get_joint_prob_mat(
    t_A = 0.1,    t_B = 0.1,    t_AB = 1,    t_C = 2, 
    rho_A = 2,    rho_B = 1,    rho_AB = 3,  rho_C = 1,  rho_ABC = 1, 
    coal_A = 1,   coal_B = 0.5, coal_AB = 1, coal_C = 1, coal_ABC = 1,
    n_int_AB = 3, n_int_ABC = 3 
)

pd.DataFrame(arr, columns=['From', 'To', 'Prob'])
```

```
          From         To      Prob
0    (0, 0, 0)  (0, 0, 0)  0.017103
1    (0, 0, 0)  (0, 0, 1)  0.010521
2    (0, 0, 1)  (0, 0, 0)  0.010521
3    (0, 0, 0)  (0, 0, 2)  0.010521
4    (0, 0, 2)  (0, 0, 0)  0.010521
..         ...        ...       ...
724  (2, 2, 2)  (2, 2, 2)  0.000099
725  (2, 2, 2)  (3, 2, 2)  0.000089
726  (3, 2, 2)  (1, 2, 2)  0.000075
727  (3, 2, 2)  (2, 2, 2)  0.000089
728  (3, 2, 2)  (3, 2, 2)  0.000099

[729 rows x 3 columns]
```