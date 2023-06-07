import random
from hungarian_algorithm import algorithm
import numpy as np
import pandas as pd
import h5py

def create_sample(n=4, bound=100):
    G = {}
    for j in range(n):
        G[f'j{j}'] = {}
        for i in range(n):
            G[f'j{j}'][f'i{i}'] = random.randrange(bound)
    try:
        m = algorithm.find_matching(G, matching_type='min', return_type='list')
    
        g = np.array(pd.DataFrame.from_dict(G))
        g = g.tolist()
    
        if m is False:
            raise Exception("try again")

        m = [[int(j[1:]), int(i[1:]), c] for (j, i), c in m]
    except:
        g, m = create_sample(n, bound)

    return g, m

n = 4
count = 50000

with h5py.File('data4.h5', 'w') as f:
    x_data = f.create_dataset('x', (count, n, n))
    y_data = f.create_dataset('y', (count, n, 3))

    for i in range(count):
        if i % 100 == 0:
            print(f'generated: {i}')
        x, y = create_sample(n)
        x_data[i] = np.array(x, dtype=int)
        y_data[i] = np.array(y, dtype=int)
