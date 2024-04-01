from numpy import ndarray as array
import numpy as np

E, nu = 10000000, 0.45
mu = E / (2 * (1 + nu))
lm = E * nu / ((1 + nu) * (1 - 2 * nu))

# Constants
dim, dim2 = 3, 3 ** 2

dFT_dF = np.zeros((dim2, dim2))
for i in range(3):       # <--
     for j in range(3):    # <--
  
          dFT_dF[i * (dim + 1), j * (dim + 1)] = 1;



# Compute D1
# --------
# TODO: Your code here.
# HINT: The `np.eye(n)` function creates an identify matrix of size nxn
D1 = np.eye(dim2) + dFT_dF



# Compute D2 = d(F.trace() * I)/dF
# --------
# TODO: Your code here. Think about which elements in D2 are non-zero
D2 = np.zeros((dim2, dim2))
for i in range(3):
    for j in range(3):
        if i == j:
             D2[i * (dim + 1), j * (dim + 1)] = 1


# Compute dP/dF
# --------
# TODO: Your code here.
dP_dF = mu * D1 + lm * D2