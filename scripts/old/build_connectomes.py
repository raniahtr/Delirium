#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Structural connectome builder (MRtrix-first, Python 3.8 compatible)

Pipeline per subject:
  1) Ensure Glasser atlas is a 3D integer labelmap (auto: 4D -> argmax+1 -> uint16).
  2) Register INTENSITY→INTENSITY: MNI T1 -> subject T1 (rigid -> affine).
  3) Apply that transform to the LABEL atlas with nearest-neighbour (preserve integers).
  4) Map atlas T1 -> DWI using inverse of DWI->T1; regrid onto mean_b0.
  5) Build connectomes (SIFT2-weighted if available, else counts), mean length, and assignments CSV.

"""

from __future__ import annotations

import os
import shlex
import subprocess
from pathlib import Path
from typing import Optional, List, Dict

# ============================ PATHS ============================

ATLAS_IN = Path("/media/RCPNAS/Data/Delirium/Delirium_Rania/atlas/HCPMMP1_labels_3D.mif")
# If you instead point to the original 4D NIfTI (e.g., glasser_MNI152NLin6Asym_atlas.nii.gz),
# the script will require FSL to argmax it to 3D.

FSLDIR = os.environ.get("FSLDIR", "/usr/local/fsl")
MNI_T1_1MM = Path(FSLDIR) / "data/standard/MNI152_T1_1mm_brain.nii.gz"
MNI_T1_2MM = Path(FSLDIR) / "data/standard/MNI152_T1_2mm_brain.nii.gz"
MNI_T1_FALLBACK: Optional[Path] = None


BASE_DIR = Path("/media/RCPNAS/Data/Delirium/Delirium_Rania/Preproc_new")
SUBJECTS = {p.name.replace("sub-", ""): p for p in BASE_DIR.glob("sub-*") if p.is_dir()}
print("Detected subjects:", list(SUBJECTS.keys()))

#SUBJECTS: Dict[str, Path] = {
#    "AF": Path("/media/RCPNAS/Data/Delirium/Delirium_Rania/Preproc_new/sub-AF"),
#    # "XX": Path("/media/RCPNAS/Data/Delirium/Delirium_Rania/Preproc/sub-XX"),
#}

os.environ.setdefault("MRTRIX_NTHREADS", "6")
os.environ.setdefault("OMP_NUM_THREADS", "6")

ASSIGN_ARGS = "-assignment_radial_search 5"  


def run(cmd: str, cwd: Optional[Path] = None) -> None:
    """Run shell command, echo output, fail fast."""
    print("\n$ " + cmd)
    res = subprocess.run(
        cmd,
        shell=True,
        cwd=str(cwd) if cwd else None,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    print(res.stdout)
    if res.returncode != 0:
        raise RuntimeError("Command failed: {}".format(cmd))


def which(cmd: str) -> bool:
    """Return True if an executable is found in PATH."""
    return subprocess.run(
        "command -v {} >/dev/null 2>&1".format(shlex.quote(cmd)), shell=True
    ).returncode == 0


def mrtrix_size(p: Path) -> List[int]:
    """Return image dimensions via mrinfo -size (len=3 for 3D, len=4 for 4D)."""
    out = subprocess.check_output(
        "mrinfo -quiet -size {}".format(shlex.quote(str(p))), shell=True, text=True
    ).strip()
    return [int(x) for x in out.split()]


def check_exists(p: Path, msg: str = "") -> None:
    if not p.exists():
        raise FileNotFoundError("Missing: {} {}".format(p, msg))


def ensure_3d_integer_atlas(atlas_in: Path, workdir: Path) -> Path:
    """
    Ensure the atlas is a 3D, uint16 .mif labelmap.
    If input is a 4D stack, uses FSL to compute argmax across the 4th dim and +1 (1..N).
    """
    workdir.mkdir(parents=True, exist_ok=True)
    atlas_in = atlas_in.resolve()
    atlas3d_mif = workdir / "atlas3D.mif"

    # Case 1: already a .mif
    if atlas_in.suffix == ".mif":
        dims = mrtrix_size(atlas_in)
        if len(dims) == 3:
            run(
                "mrconvert {} {} -datatype uint16 -stride -1,2,3".format(
                    shlex.quote(str(atlas_in)), shlex.quote(str(atlas3d_mif))
                )
            )
            return atlas3d_mif
        # .mif but 4D -> convert to NIfTI for argmax path
        tmp_nifti = workdir / "atlas_in.nii.gz"
        run("mrconvert {} {}".format(shlex.quote(str(atlas_in)), shlex.quote(str(tmp_nifti))))
        atlas_in = tmp_nifti  # continue below

    # Case 2: NIfTI path (3D or 4D)
    dims = mrtrix_size(atlas_in)
    if len(dims) == 4 and dims[3] > 1:
        if not which("fslmaths"):
            raise RuntimeError(
                "Atlas appears 4D ({} vols), but FSL is not available to compute argmax.\n"
                "Install FSL or precompute a 3D atlas (argmax+1) and set ATLAS_IN to it."
                .format(dims[3])
            )
        argmax0 = workdir / "atlas_argmax0.nii.gz"
        labels3d = workdir / "atlas_labels_3D.nii.gz"
        run("fslmaths {} -Tmaxn {}".format(shlex.quote(str(atlas_in)), shlex.quote(str(argmax0))))
        run("fslmaths {} -add 1 {}".format(shlex.quote(str(argmax0)), shlex.quote(str(labels3d))))
        run(
            "mrconvert {} {} -datatype uint16 -stride -1,2,3".format(
                shlex.quote(str(labels3d)), shlex.quote(str(atlas3d_mif))
            )
        )
    else:
        # Already 3D NIfTI
        run(
            "mrconvert {} {} -datatype uint16 -stride -1,2,3".format(
                shlex.quote(str(atlas_in)), shlex.quote(str(atlas3d_mif))
            )
        )

    return atlas3d_mif


def prepare_brain_T1_for_registration(outdir: Path, t1_mif: Path):
    five_tt    = outdir.parent / "5tt_nocoreg.mif"
    brain_prob = outdir / "brain_prob.mif"              # native 5TT grid
    brain_prob_T1 = outdir / "brain_prob_in_T1.mif"     # resampled to T1 grid
    t1_mask    = outdir / "T1_mask.mif"
    t1_brain   = outdir / "T1_brain.mif"

    if not five_tt.exists():
        # Fallback: simple mask
        run(f"mrthreshold -abs 1e-6 {shlex.quote(str(t1_mif))} {shlex.quote(str(t1_mask))} -force")
        return t1_mif, t1_mask

    # Sum tissues (GM+WM+CSF) in 5TT space
    run(f"mrmath {shlex.quote(str(five_tt))} sum -axis 3 {shlex.quote(str(brain_prob))} -force")

    # **Resample to T1 grid** so dimensions/voxels match
    run(
        "mrtransform {} -template {} -interp linear {} -force".format(
            shlex.quote(str(brain_prob)),
            shlex.quote(str(t1_mif)),
            shlex.quote(str(brain_prob_T1)),
        )
    )

    # Threshold mask (now in T1 grid)
    run(f"mrthreshold -abs 0.1 {shlex.quote(str(brain_prob_T1))} {shlex.quote(str(t1_mask))} -force")

    # Apply mask → brain-only T1 (grids match, so no error)
    run(
        "mrcalc {} {} -mult {} -force".format(
            shlex.quote(str(t1_mif)),
            shlex.quote(str(t1_mask)),
            shlex.quote(str(t1_brain)),
        )
    )

    return t1_brain, t1_mask

def pick_mni_t1() -> Path:
    """Pick an MNI T1 intensity image. Prefer 1mm, else 2mm, else fallback."""
    if MNI_T1_1MM.exists():
        return MNI_T1_1MM
    if MNI_T1_2MM.exists():
        return MNI_T1_2MM
    if MNI_T1_FALLBACK and MNI_T1_FALLBACK.exists():
        return MNI_T1_FALLBACK
    raise FileNotFoundError(
        "Cannot find MNI T1 at:\n  {}\n  {}\nSet FSLDIR or MNI_T1_FALLBACK at top of script."
        .format(MNI_T1_1MM, MNI_T1_2MM)
    )


def register_mniT1_to_subjectT1_apply_to_atlas(atlas3d_mif: Path, t1_nii: Path, outdir: Path) -> Path:
    """
    1) Convert inputs.
    2) Register intensity→intensity: MNI_T1 -> subject T1 (rigid -> affine).
       (MRtrix 3.0.2 requires -rigid_init_matrix on the affine stage)
    3) Apply that transform to the LABEL atlas (nearest).
    """
    outdir.mkdir(parents=True, exist_ok=True)

    # Subject T1
    t1_mif = outdir / "T1.mif"
    run("mrconvert {} {}".format(shlex.quote(str(t1_nii)), shlex.quote(str(t1_mif))))

    # MNI T1
    mni_t1 = pick_mni_t1()
    mni_t1_mif = outdir / "MNI_T1.mif"
    run("mrconvert {} {}".format(shlex.quote(str(mni_t1)), shlex.quote(str(mni_t1_mif))))
    mni_use = mni_t1_mif
    t1_brain, t1_mask = prepare_brain_T1_for_registration(outdir, t1_mif)

    mni_mask = outdir / "MNI_mask.mif"
    run("mrthreshold -abs 1e-6 {} {} -force".format(
        shlex.quote(str(mni_use)), shlex.quote(str(mni_mask))
    ))

    # Rigid -> Affine registration (intensity images)
    rigid = outdir / "mni2t1_rigid.txt"
    affine = outdir / "mni2t1_affine.txt"
    run(
        "mrregister {} {} "
        "-type rigid "
        "-mask1 {} -mask2 {} "
        "-rigid {} -force".format(
            shlex.quote(str(mni_use)),
            shlex.quote(str(t1_brain)),
            shlex.quote(str(mni_mask)),
            shlex.quote(str(t1_mask)),
            shlex.quote(str(rigid)),
        )
    )

    run(
        "mrregister {} {} "
        "-type affine "
        "-rigid_init_matrix {} "
        "-mask1 {} -mask2 {} "
        "-affine {} -force".format(
            shlex.quote(str(mni_use)),
            shlex.quote(str(t1_brain)),
            shlex.quote(str(rigid)),        # MRtrix 3.0.2 flag name
            shlex.quote(str(mni_mask)),
            shlex.quote(str(t1_mask)),
            shlex.quote(str(affine)),
        )
    )

    # Apply to the LABEL atlas (NEAREST!!)
    atlas_in_t1 = outdir / "HCPMMP1_in_T1.mif"
    run(
        "mrtransform {} {} -linear {} -interp nearest".format(
            shlex.quote(str(atlas3d_mif)), shlex.quote(str(atlas_in_t1)), shlex.quote(str(affine))
        )
    )
    return atlas_in_t1


def map_T1_to_DWI(atlas_in_t1: Path, diff2struct_txt: Path, mean_b0_mif: Path, outdir: Path) -> Path:
    """
    Apply T1 -> DWI (inverse of provided DWI->T1), regridding to the DWI grid.
    """
    atlas_in_dwi = outdir / "HCPMMP1_in_DWI.mif"
    run(
        "mrtransform {} {} -linear {} -interp nearest -template {}".format(
            shlex.quote(str(atlas_in_t1)),
            shlex.quote(str(atlas_in_dwi)),
            shlex.quote(str(diff2struct_txt)),
            shlex.quote(str(mean_b0_mif)),
        )
    )
    # Quick sanity print of label range (should be ~1..360)
    run("mrstats {} -output min -output max".format(shlex.quote(str(atlas_in_dwi))))
    return atlas_in_dwi


def build_connectomes(sub_id: str, sub_dir: Path, atlas_in_dwi: Path, outdir: Path) -> None:
    """Build SIFT2-weighted/Counts connectome + mean length; save assignments CSV."""
    # Tractogram
    tck = sub_dir / ("sub-{}_tracks_10M.tck".format(sub_id))
    if not tck.exists():
        cands = sorted(sub_dir.glob("*.tck"))
        if not cands:
            raise FileNotFoundError("No .tck found in {}".format(sub_dir))
        tck = cands[0]

    # Optional SIFT2
    wts = sub_dir / ("sub-{}_sift_coeffs.txt".format(sub_id))
    have_sift2 = wts.exists()

    # Assignments for QA
    assign_csv = outdir / ("sub-{}_assignments.csv".format(sub_id))

    if have_sift2:
        run(
            "tck2connectome {} {} {} -tck_weights_in {} -zero_diagonal -symmetric {} -out_assignments {}".format(
                shlex.quote(str(tck)),
                shlex.quote(str(atlas_in_dwi)),
                shlex.quote(str(outdir / ("sub-{}_SC_sift2.csv".format(sub_id)))),
                shlex.quote(str(wts)),
                ASSIGN_ARGS,
                shlex.quote(str(assign_csv)),
            )
        )
    else:
        run(
            "tck2connectome {} {} {} -zero_diagonal -symmetric {} -out_assignments {}".format(
                shlex.quote(str(tck)),
                shlex.quote(str(atlas_in_dwi)),
                shlex.quote(str(outdir / ("sub-{}_SC_counts.csv".format(sub_id)))),
                ASSIGN_ARGS,
                shlex.quote(str(assign_csv)),
            )
        )

    # Mean streamline length per edge (helpful for QC/normalisation)
    run(
        "tck2connectome {} {} {} -scale_length -stat_edge mean -zero_diagonal -symmetric {}".format(
            shlex.quote(str(tck)),
            shlex.quote(str(atlas_in_dwi)),
            shlex.quote(str(outdir / ("sub-{}_meanLength.csv".format(sub_id)))),
            ASSIGN_ARGS,
        )
    )


def main() -> None:
    # Sanity checks
    if not ATLAS_IN.exists():
        raise FileNotFoundError("ATLAS_IN not found: {}".format(ATLAS_IN))
    for sid, p in SUBJECTS.items():
        if not p.exists():
            raise FileNotFoundError("Subject path not found for {}: {}".format(sid, p))

    for sid, sdir in SUBJECTS.items():
        print("\n=== Subject {} ===".format(sid))
        out = sdir / "connectome_local"
        out.mkdir(parents=True, exist_ok=True)

        t1 = sdir / ("sub-{}_T1w.nii.gz".format(sid))
        dwi2t1 = sdir / "diff2struct_mrtrix.txt"
        mean_b0 = sdir / "mean_b0.mif"

        for p, hint in (
            (t1, "T1"),
            (dwi2t1, "DWI->T1 transform"),
            (mean_b0, "mean_b0.mif"),
        ):
            if not p.exists():
                raise FileNotFoundError("Missing {} for {}: {}".format(hint, sid, p))

        # Ensure atlas is 3D uint16 .mif (convert if needed)
        atlas3d_mif = ensure_3d_integer_atlas(ATLAS_IN, out)

        # Register INTENSITY→INTENSITY (MNI T1 -> subject T1), apply to LABEL atlas with NN
        atlas_in_t1 = register_mniT1_to_subjectT1_apply_to_atlas(atlas3d_mif, t1, out)

        # T1 -> DWI (inverse), regrid to DWI grid
        atlas_in_dwi = map_T1_to_DWI(atlas_in_t1, dwi2t1, mean_b0, out)

        # Connectomes (+ assignments CSV) Now with preproc_new i run build_connectomes_light.sh
        # build_connectomes(sid, sdir, atlas_in_dwi, out)

        print("Done: {} -> {}".format(sid, out))


if __name__ == "__main__":
    main()

