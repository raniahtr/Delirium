import argparse
from pathlib import Path

import numpy as np
from numpy.linalg import eigh
import scipy.io as sio


def compute_harmonics(A):
    A = np.asarray(A, float)
    A = np.nan_to_num(A, nan=0.0, posinf=0.0, neginf=0.0)
    A = np.maximum(A, 0)
    np.fill_diagonal(A, 0)

    deg = A.sum(axis=1)
    deg[deg == 0] = 1e-12
    D_inv_sqrt = np.diag(1.0 / np.sqrt(deg))
    A_sym = D_inv_sqrt @ A @ D_inv_sqrt
    L = np.eye(A.shape[0]) - A_sym

    evals, evecs = eigh(L)
    return A, A_sym, L, evals, evecs


def process_file(in_path: Path, out_path: Path):
    mat = sio.loadmat(in_path)
    if "A_mean" not in mat:
        raise KeyError(f"'A_mean' not found in {in_path}")
    A = mat["A_mean"]
    A_unorm, A_norm, L, evals, evecs = compute_harmonics(A)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sio.savemat(
        out_path,
        {
            "A_unorm": A_unorm,
            "A_norm": A_norm,
            "evals": evals,
            "evecs": evecs,
            "L_norm": L,
        },
    )
    print(f"Saved harmonics to {out_path} (shape {A_unorm.shape})")


def main():
    parser = argparse.ArgumentParser(description="Compute Laplacian harmonics for SC matrices")
    parser.add_argument(
        "--inputs",
        nargs="*",
        help="Input MAT files (default: SDI/data/SC_A*.mat)",
    )
    args = parser.parse_args()

    base_dir = Path("/media/RCPNAS/Data/Delirium/Delirium_Rania/SDI/data")

    if args.inputs:
        input_paths = [Path(p) for p in args.inputs]
    else:
        input_paths = sorted(base_dir.glob("SC_A*.mat"))

    if not input_paths:
        raise SystemExit("No input SC_A*.mat files found.")

    for in_path in input_paths:
        suffix = in_path.stem.replace("SC_A", "").strip("_")
        if suffix:
            out_name = f"harmonics_{suffix}.mat"
        else:
            out_name = "harmonics.mat"
        out_path = base_dir / out_name
        process_file(in_path, out_path)


if __name__ == "__main__":
    main()