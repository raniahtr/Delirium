import sys, os
import numpy as np, pandas as pd

if len(sys.argv) < 3:
    print("Usage: python sizecorr.py <SUBJ> <MATRIX_BASENAME>")
    print("Example: python sizecorr.py sub-AF SC_sift2")
    sys.exit(1)

subj, base = sys.argv[1], sys.argv[2]
d = "connectome_local"

A = np.loadtxt(f"{d}/{subj}_{base}.csv", delimiter=",")
vol = pd.read_csv(f"{d}/{subj}_node_volumes.csv")["voxels"].to_numpy(float)

n = A.shape[0]
if len(vol) != n:
    raise RuntimeError(f"Volume length {len(vol)} != matrix size {n}. "
                       "Make sure volumes come from the SAME atlas used for this connectome.")

norm = np.sqrt(np.outer(vol, vol))
norm[norm==0] = np.nan

A_size = A / norm
np.savetxt(f"{d}/{subj}_{base}_sizecorr.csv", A_size, delimiter=",", fmt="%.6g")
print(f"[OK] Wrote {d}/{subj}_{base}_sizecorr.csv")
