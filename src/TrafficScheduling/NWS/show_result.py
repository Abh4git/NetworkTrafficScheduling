import numpy as np

with open('solution.npy', 'rb') as f:
    a = np.load(f)
    print(a)
