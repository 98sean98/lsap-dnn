import h5py
import numpy as np

n = 4
count = 1000

with h5py.File('data4.h5', 'r') as f:
    x_data = f['x']
    y_data = f['y']

    print(x_data[0], y_data[0])
    print(x_data[-1], y_data[-1])
    
    print(isinstance(x_data[0], np.ndarray))
    print(isinstance(y_data[0], np.ndarray))
    
    for j, x in enumerate(x_data):
        if not isinstance(x, np.ndarray):
            print(j, x)
    
    for j, y in enumerate(y_data):
        if not isinstance(y, np.ndarray):
            print(j, y)
    
    print("done checking")