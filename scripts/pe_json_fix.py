#!/usr/bin/env python3
import json, sys

"""
Fix/augment BIDS JSON for DWI/anat:
- Ensure PhaseEncodingDirection exists (map from DICOM fields when possible)
- Compute TotalReadoutTime if missing: (ReconMatrixPE - 1) * EffectiveEchoSpacing
- Leave everything else untouched
"""

def infer_ped(j):
    # Prefer already-present BIDS PED
    ped = j.get("PhaseEncodingDirection")
    if ped:
        return ped

    # Vendor/DICOM cues
    ip = j.get("InPlanePhaseEncodingDirectionDICOM") or j.get("PhaseEncodingDirectionDICOM")
    if ip:
        ipu = str(ip).upper()
        # Common for axial EPI:
        #   COL ~ phase along columns -> j (A<->P)
        #   ROW ~ phase along rows    -> i (L<->R)
        if "COL" in ipu:
            return "j"
        if "ROW" in ipu:
            return "i"

    # Siemens private fields sometimes expose polarity via "ReversePE" or similar,
    # but dcm2niix doesn't always propagate. We keep polarity neutral here ("j" not "j-");
    # the shell script may flip using the reverse-b0 if available.
    return None

def compute_trt(j):
    # If already present, keep it
    if "TotalReadoutTime" in j:
        return j["TotalReadoutTime"]

    # Try BIDS-expected keys (dcm2niix fills many of them)
    ees = j.get("EffectiveEchoSpacing") or j.get("EstimatedEffectiveEchoSpacing")
    rpe = j.get("ReconMatrixPE") or j.get("AcquisitionMatrixPE") or j.get("PhaseEncodingSteps")

    try:
        ees = float(ees) if ees is not None else None
    except Exception:
        ees = None

    try:
        # Some writers store this as float in JSON; coerce to int if possible
        if rpe is not None:
            rpe = int(round(float(rpe)))
    except Exception:
        rpe = None

    if ees and rpe and rpe >= 2:
        return (rpe - 1) * ees

    return None

def main(path):
    try:
        with open(path, "r") as f:
            j = json.load(f)
    except Exception:
        # If JSON is malformed or missing, silently ignore
        return

    ped = infer_ped(j)
    if ped:
        j["PhaseEncodingDirection"] = ped

    trt = compute_trt(j)
    if trt is not None:
        j["TotalReadoutTime"] = trt

    with open(path, "w") as f:
        json.dump(j, f, indent=2, sort_keys=True)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: pe_json_fix.py <json_file>", file=sys.stderr)
        sys.exit(1)
    main(sys.argv[1])
