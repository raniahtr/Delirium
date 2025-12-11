import numpy as np
from numpy.linalg import eigh
import scipy.io as sio

mat = sio.loadmat("/media/RCPNAS/Data/Delirium/Delirium_Rania/SDI/data/SC_A.mat")
A = np.asarray(mat["A_mean"], float)
# Replace NaN/Inf with zeros for stability
A = np.nan_to_num(A, nan=0.0, posinf=0.0, neginf=0.0)
# Guard against negatives / self-loops if needed:
A = np.maximum(A, 0)
np.fill_diagonal(A, 0)

deg = A.sum(axis=1)
# Avoid div-by-zero
deg[deg == 0] = 1e-12
D_inv_sqrt = np.diag(1.0 / np.sqrt(deg))
A_sym = D_inv_sqrt @ A @ D_inv_sqrt
L = np.eye(A.shape[0]) - A_sym

# Eigen-decomposition (symmetric → eigh)
evals, evecs = eigh(L)
# evecs are the structural harmonics ordered by increasing eigenvalue

sio.savemat("/media/RCPNAS/Data/Delirium/Delirium_Rania/SDI/data/harmonics.mat", {
    "A_unorm": A,
    "A_norm": A_sym,
    "evals": evals,
    "evecs": evecs,
    "L_norm": L,
})