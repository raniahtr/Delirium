#!/usr/bin/env python3
import numpy as np
import nibabel as nib
import scipy.io as sio
import numpy as np, matplotlib.pyplot as plt, matplotlib.colors as mcolors
from nilearn import plotting


# Inputs 
HARM_PATH = "/media/RCPNAS/Data/Delirium/Delirium_Rania/SDI/data/harmonics_ICU_combined.mat"
ATLAS_PATH = "/media/RCPNAS/Data/Delirium/Delirium_Rania/atlas/Final_Combined_atlas_MNI2009c_1mm.nii.gz"

# Parcels removed from the connectomes (1-indexed labels)
EXCLUDED_LABELS = {390, 423}

def main():
    # Load harmonics
    mat = sio.loadmat(HARM_PATH)
    evecs = np.asarray(mat["evecs"])
    evals = np.asarray(mat["evals"]).ravel()

    # Load atlas
    img = nib.load(ATLAS_PATH)
    data = img.get_fdata()
    aff = img.affine
    print("Atlas loaded, handling it")
    # Collect labels, drop excluded, sort to match original order minus removed
    labels = np.unique(data[data > 0]).astype(int)
    labels = [lbl for lbl in labels if lbl not in EXCLUDED_LABELS]
    labels = sorted(labels)

    # Sanity check: label count must match harmonics dimension
    if len(labels) != evecs.shape[0]:
        raise ValueError(f"Label count {len(labels)} != harmonics size {evecs.shape[0]}")
    print("computing centroids for each label")
    # Compute centroids for each label
    coords = []
    for lbl in labels:
        ijk = np.argwhere(data == lbl)
        xyz = nib.affines.apply_affine(aff, ijk).mean(axis=0)
        coords.append(xyz)
    coords = np.array(coords)  # shape (N, 3)

    # Choose eigenmode k (0 is the trivial constant mode; start at 1)
    k = 3
    if k >= evecs.shape[1]:
        raise ValueError(f"Requested mode k={k} but only {evecs.shape[1]} modes available")
    vals = evecs[:, k]
    print(f"Mode k={k}: vals range [{vals.min():.4f}, {vals.max():.4f}]")
    print(f"Coords range x[{coords[:,0].min():.1f},{coords[:,0].max():.1f}] "
          f"y[{coords[:,1].min():.1f},{coords[:,1].max():.1f}] "
          f"z[{coords[:,2].min():.1f},{coords[:,2].max():.1f}]")


    finite_vals = vals[np.isfinite(vals)]
    if finite_vals.size == 0:
        raise ValueError("All eigenvector values are NaN/inf; cannot plot.")
    max_abs = np.max(np.abs(finite_vals))
    if max_abs == 0:
        max_abs = 1e-6  # avoid zero range
    norm = mcolors.TwoSlopeNorm(vmin=-max_abs, vcenter=0.0, vmax=max_abs)
    cmap = plt.get_cmap("coolwarm")
    node_color = cmap(norm(vals))  # shape (N,4)
    print("displaying connectome")
    # Nilearn expects an adjacency matrix
    zero_adj = np.zeros((coords.shape[0], coords.shape[0]))
    display = plotting.plot_connectome(
        adjacency_matrix=zero_adj,
        node_coords=coords,
        node_color=node_color,  
        node_size=20,
        display_mode="ortho",
        colorbar=False,  
        annotate=False,
    )

    # Save connectome
    print("saving connectome")
    out_png = "/media/RCPNAS/Data/Delirium/Delirium_Rania/SDI/data/harmonics_combined_k3.png"
    display.savefig(out_png)
    display.close()  # free resources
    print(f"Saved to {out_png}")

if __name__ == "__main__":
    main()