#!/usr/bin/env python
"""
Voxelwise SNR Computation for Preprocessing QC
Computes SNR before and after preprocessing using global noise standard deviation.
Outputs: statistical summaries, voxelwise maps, and visualization-ready data.
"""

import numpy as np
import nibabel as nib
import argparse
import os
import json
import csv
from pathlib import Path
import subprocess
import sys


def load_mif_as_nifti(mif_path, nifti_path=None):
    """Convert .mif to .nii.gz if needed, return nibabel image."""
    if mif_path.endswith('.mif'):
        if nifti_path is None:
            nifti_path = mif_path.replace('.mif', '.nii.gz')
        if not os.path.exists(nifti_path):
            # Convert using mrconvert
            cmd = ['mrconvert', mif_path, nifti_path, '-force']
            subprocess.run(cmd, check=True, capture_output=True)
        return nib.load(nifti_path)
    else:
        return nib.load(mif_path)


def extract_raw_b0(dwi_mif_path, output_path):
    """Extract mean b0 from raw DWI."""
    print(f"  Extracting b0 volumes from {dwi_mif_path}...")
    # Extract b0 volumes
    b0s_path = output_path.replace('.mif', '_b0s.mif')
    cmd = ['dwiextract', dwi_mif_path, '-bzero', '-', '|', 
           'mrmath', '-', 'mean', output_path, '-axis', '3', '-force']
    # Run as shell command since we need pipe
    full_cmd = f"dwiextract {dwi_mif_path} -bzero - | mrmath - mean {output_path} -axis 3 -force"
    result = subprocess.run(full_cmd, shell=True, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"Failed to extract b0: {result.stderr}")
    return output_path


def create_wm_mask(tt5_path, mean_b0_path, output_path, threshold=0.7, erode_passes=2):
    """Create deep white matter mask from 5tt image."""
    print(f"  Creating deep WM mask from {tt5_path}...")
    
    # Extract WM probability (component index 2)
    wm_prob_path = output_path.replace('.mif', '_prob.mif')
    cmd = ['mrconvert', tt5_path, '-coord', '3', '2', wm_prob_path, '-force']
    subprocess.run(cmd, check=True, capture_output=True)
    
    # Threshold
    wm_mask_path = output_path.replace('.mif', '_mask.mif')
    cmd = ['mrthreshold', wm_prob_path, '-abs', str(threshold), wm_mask_path, '-force']
    subprocess.run(cmd, check=True, capture_output=True)
    
    # Erode to get deep WM
    wm_deep_path = output_path.replace('.mif', '_deep.mif')
    cmd = ['maskfilter', wm_mask_path, 'erode', '-npass', str(erode_passes), wm_deep_path, '-force']
    subprocess.run(cmd, check=True, capture_output=True)
    
    # Resample to b0 space
    cmd = ['mrtransform', wm_deep_path, '-template', mean_b0_path, 
           '-interp', 'nearest', output_path, '-force']
    subprocess.run(cmd, check=True, capture_output=True)
    
    return output_path


def compute_global_noise_std(noise_path):
    """Compute global noise standard deviation from noise map."""
    noise_img = load_mif_as_nifti(noise_path)
    noise_data = np.squeeze(noise_img.get_fdata())
    
    # Remove zeros and compute std
    noise_vals = noise_data[noise_data > 0]
    if noise_vals.size == 0:
        raise ValueError("No valid noise values found in noise map")
    
    noise_std = float(np.std(noise_vals))
    noise_mean = float(np.mean(noise_vals))
    
    return noise_std, noise_mean, noise_vals.size


def compute_voxelwise_snr(b0_path, noise_std, wm_mask_path=None, mask_output=True):
    """
    Compute voxelwise SNR map using global noise standard deviation.
    
    Parameters:
    -----------
    b0_path : str
        Path to b0 image (can be .mif or .nii.gz)
    noise_std : float
        Global noise standard deviation
    wm_mask_path : str, optional
        Path to white matter mask
    mask_output : bool
        If True, only compute SNR inside WM mask
    
    Returns:
    --------
    snr_map : np.ndarray
        Voxelwise SNR map
    b0_img : nibabel image
        B0 image object (for saving)
    valid_mask : np.ndarray
        Boolean mask of valid voxels
    """
    # Load b0 image
    b0_img = load_mif_as_nifti(b0_path)
    b0_data = np.squeeze(b0_img.get_fdata())
    
    # Handle 4D data
    if b0_data.ndim == 4:
        b0_data = np.mean(b0_data, axis=-1)
    
    # Load WM mask if provided
    if wm_mask_path and os.path.exists(wm_mask_path):
        wm_mask_img = load_mif_as_nifti(wm_mask_path)
        wm_mask = np.squeeze(wm_mask_img.get_fdata()).astype(bool)
        
        if b0_data.shape != wm_mask.shape:
            raise ValueError(
                f"Shape mismatch: b0 {b0_data.shape} vs WM mask {wm_mask.shape}"
            )
    else:
        wm_mask = np.ones_like(b0_data, dtype=bool)
    
    # Compute voxelwise SNR
    snr_map = np.zeros_like(b0_data, dtype=float)
    snr_map[:] = np.nan
    
    # Valid voxels: positive b0 signal, and optionally inside WM
    if mask_output:
        valid = (b0_data > 0) & wm_mask
    else:
        valid = b0_data > 0
    
    if not np.any(valid):
        raise ValueError("No valid voxels found for SNR computation")
    
    # SNR = signal / noise_std
    snr_map[valid] = b0_data[valid] / noise_std
    
    # Clean up invalid values
    snr_map[snr_map < 0] = np.nan
    snr_map[np.isinf(snr_map)] = np.nan
    
    return snr_map, b0_img, valid


def transform_snr_to_preprocessed_space(snr_raw_map, raw_b0_img, preproc_b0_path, output_path):
    """
    Transform raw SNR map to preprocessed space.
    Uses mrtransform with the preprocessed b0 as template.
    """
    # Save raw SNR map temporarily
    temp_raw_snr = output_path.replace('.nii.gz', '_raw_temp.nii.gz')
    nib.save(nib.Nifti1Image(snr_raw_map, raw_b0_img.affine, raw_b0_img.header), temp_raw_snr)
    
    # Transform to preprocessed space
    print(f"  Transforming raw SNR map to preprocessed space...")
    cmd = ['mrtransform', temp_raw_snr, '-template', preproc_b0_path,
           '-interp', 'linear', output_path, '-force']
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    # Clean up temp file
    if os.path.exists(temp_raw_snr):
        os.remove(temp_raw_snr)
    
    if result.returncode != 0:
        print(f"  WARNING: Transformation failed: {result.stderr}")
        print(f"  Using raw SNR map as-is (may be in different space)")
        return snr_raw_map, raw_b0_img
    
    # Load transformed map
    transformed_img = nib.load(output_path)
    return transformed_img.get_fdata(), transformed_img


def compute_comparison_statistics(snr_pre, snr_post, valid_mask):
    """Compute comprehensive statistics for before/after comparison."""
    # Mask to valid voxels
    snr_pre_masked = snr_pre[valid_mask & ~np.isnan(snr_pre)]
    snr_post_masked = snr_post[valid_mask & ~np.isnan(snr_post)]
    
    if snr_pre_masked.size == 0 or snr_post_masked.size == 0:
        raise ValueError("No valid voxels for comparison")
    
    # Ensure same voxels for paired comparison
    common_valid = valid_mask & ~np.isnan(snr_pre) & ~np.isnan(snr_post)
    snr_pre_paired = snr_pre[common_valid]
    snr_post_paired = snr_post[common_valid]
    
    stats = {
        'preprocessing': {
            'mean': float(np.nanmean(snr_pre_masked)),
            'median': float(np.nanmedian(snr_pre_masked)),
            'std': float(np.nanstd(snr_pre_masked)),
            'min': float(np.nanmin(snr_pre_masked)),
            'max': float(np.nanmax(snr_pre_masked)),
            'percentiles': {
                'p5': float(np.nanpercentile(snr_pre_masked, 5)),
                'p25': float(np.nanpercentile(snr_pre_masked, 25)),
                'p50': float(np.nanpercentile(snr_pre_masked, 50)),
                'p75': float(np.nanpercentile(snr_pre_masked, 75)),
                'p95': float(np.nanpercentile(snr_pre_masked, 95))
            },
            'n_voxels': int(snr_pre_masked.size)
        },
        'postprocessing': {
            'mean': float(np.nanmean(snr_post_masked)),
            'median': float(np.nanmedian(snr_post_masked)),
            'std': float(np.nanstd(snr_post_masked)),
            'min': float(np.nanmin(snr_post_masked)),
            'max': float(np.nanmax(snr_post_masked)),
            'percentiles': {
                'p5': float(np.nanpercentile(snr_post_masked, 5)),
                'p25': float(np.nanpercentile(snr_post_masked, 25)),
                'p50': float(np.nanpercentile(snr_post_masked, 50)),
                'p75': float(np.nanpercentile(snr_post_masked, 75)),
                'p95': float(np.nanpercentile(snr_post_masked, 95))
            },
            'n_voxels': int(snr_post_masked.size)
        },
        'improvement': {
            'mean_delta': float(np.nanmean(snr_post_paired - snr_pre_paired)),
            'median_delta': float(np.nanmedian(snr_post_paired - snr_pre_paired)),
            'percent_improvement': float(
                (np.nanmean(snr_post_paired) - np.nanmean(snr_pre_paired)) / 
                np.nanmean(snr_pre_paired) * 100
            ),
            'correlation': float(np.corrcoef(snr_pre_paired, snr_post_paired)[0, 1]),
            'rmse': float(np.sqrt(np.nanmean((snr_post_paired - snr_pre_paired)**2)))
        },
        'voxel_counts': {
            'total_valid': int(np.sum(common_valid)),
            'improved': int(np.sum(snr_post_paired > snr_pre_paired)),
            'degraded': int(np.sum(snr_post_paired < snr_pre_paired)),
            'unchanged': int(np.sum(np.abs(snr_post_paired - snr_pre_paired) < 0.1))
        },
        'quality_thresholds': {
            'pre_snr_gt_20': int(np.sum(snr_pre_masked > 20)),
            'post_snr_gt_20': int(np.sum(snr_post_masked > 20)),
            'pre_snr_lt_10': int(np.sum(snr_pre_masked < 10)),
            'post_snr_lt_10': int(np.sum(snr_post_masked < 10)),
            'pre_snr_15_20': int(np.sum((snr_pre_masked >= 15) & (snr_pre_masked <= 20))),
            'post_snr_15_20': int(np.sum((snr_post_masked >= 15) & (snr_post_masked <= 20)))
        }
    }
    
    return stats


def save_results(subject_id, out_dir, snr_pre_map, snr_post_map, snr_diff_map,
                 snr_pre_img, snr_post_img, stats, noise_std, noise_mean):
    """Save all outputs: maps, statistics, and summary files."""
    os.makedirs(out_dir, exist_ok=True)
    
    prefix = f"{subject_id}_"
    
    # Save SNR maps
    snr_pre_path = os.path.join(out_dir, f"{prefix}snr_voxelwise_pre.nii.gz")
    snr_post_path = os.path.join(out_dir, f"{prefix}snr_voxelwise_post.nii.gz")
    snr_diff_path = os.path.join(out_dir, f"{prefix}snr_voxelwise_diff.nii.gz")
    
    nib.save(nib.Nifti1Image(snr_pre_map, snr_pre_img.affine, snr_pre_img.header), snr_pre_path)
    nib.save(nib.Nifti1Image(snr_post_map, snr_post_img.affine, snr_post_img.header), snr_post_path)
    nib.save(nib.Nifti1Image(snr_diff_map, snr_post_img.affine, snr_post_img.header), snr_diff_path)
    
    # Save statistics JSON
    stats_path = os.path.join(out_dir, f"{prefix}snr_comparison_stats.json")
    with open(stats_path, 'w') as f:
        json.dump({
            'subject': subject_id,
            'noise_std_global': noise_std,
            'noise_mean_global': noise_mean,
            'statistics': stats
        }, f, indent=2)
    
    # Save CSV summary (for statistical analysis)
    csv_path = os.path.join(out_dir, f"{prefix}snr_summary.csv")
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['metric', 'preprocessing', 'postprocessing', 'improvement', 'improvement_pct'])
        writer.writerow(['mean', stats['preprocessing']['mean'], stats['postprocessing']['mean'],
                        stats['improvement']['mean_delta'], stats['improvement']['percent_improvement']])
        writer.writerow(['median', stats['preprocessing']['median'], stats['postprocessing']['median'],
                        stats['improvement']['median_delta'], ''])
        writer.writerow(['std', stats['preprocessing']['std'], stats['postprocessing']['std'], '', ''])
        writer.writerow(['min', stats['preprocessing']['min'], stats['postprocessing']['min'], '', ''])
        writer.writerow(['max', stats['preprocessing']['max'], stats['postprocessing']['max'], '', ''])
        writer.writerow(['p5', stats['preprocessing']['percentiles']['p5'], 
                        stats['postprocessing']['percentiles']['p5'], '', ''])
        writer.writerow(['p95', stats['preprocessing']['percentiles']['p95'],
                        stats['postprocessing']['percentiles']['p95'], '', ''])
        writer.writerow(['correlation', '', '', stats['improvement']['correlation'], ''])
        writer.writerow(['rmse', '', '', stats['improvement']['rmse'], ''])
        writer.writerow(['n_voxels_improved', '', '', stats['voxel_counts']['improved'], ''])
        writer.writerow(['n_voxels_degraded', '', '', stats['voxel_counts']['degraded'], ''])
        writer.writerow(['snr_gt_20_pre', stats['quality_thresholds']['pre_snr_gt_20'], '', '', ''])
        writer.writerow(['snr_gt_20_post', '', stats['quality_thresholds']['post_snr_gt_20'], '', ''])
    
    # Save text summary (human-readable)
    txt_path = os.path.join(out_dir, f"{prefix}snr_summary.txt")
    with open(txt_path, 'w') as f:
        f.write(f"Voxelwise SNR Comparison: {subject_id}\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Global Noise Statistics:\n")
        f.write(f"  Mean: {noise_mean:.3f}\n")
        f.write(f"  Std:  {noise_std:.3f}\n\n")
        f.write(f"Pre-processing SNR:\n")
        f.write(f"  Mean:   {stats['preprocessing']['mean']:.2f} ± {stats['preprocessing']['std']:.2f}\n")
        f.write(f"  Median: {stats['preprocessing']['median']:.2f}\n")
        f.write(f"  Range:  [{stats['preprocessing']['min']:.2f}, {stats['preprocessing']['max']:.2f}]\n")
        f.write(f"  P5-P95: [{stats['preprocessing']['percentiles']['p5']:.2f}, {stats['preprocessing']['percentiles']['p95']:.2f}]\n\n")
        f.write(f"Post-processing SNR:\n")
        f.write(f"  Mean:   {stats['postprocessing']['mean']:.2f} ± {stats['postprocessing']['std']:.2f}\n")
        f.write(f"  Median: {stats['postprocessing']['median']:.2f}\n")
        f.write(f"  Range:  [{stats['postprocessing']['min']:.2f}, {stats['postprocessing']['max']:.2f}]\n")
        f.write(f"  P5-P95: [{stats['postprocessing']['percentiles']['p5']:.2f}, {stats['postprocessing']['percentiles']['p95']:.2f}]\n\n")
        f.write(f"Improvement:\n")
        f.write(f"  Mean ΔSNR:     {stats['improvement']['mean_delta']:.2f}\n")
        f.write(f"  % Improvement: {stats['improvement']['percent_improvement']:.1f}%\n")
        f.write(f"  Correlation:   {stats['improvement']['correlation']:.3f}\n")
        f.write(f"  RMSE:          {stats['improvement']['rmse']:.2f}\n\n")
        f.write(f"Voxel Counts:\n")
        f.write(f"  Improved:  {stats['voxel_counts']['improved']}\n")
        f.write(f"  Degraded:  {stats['voxel_counts']['degraded']}\n")
        f.write(f"  Unchanged: {stats['voxel_counts']['unchanged']}\n\n")
        f.write(f"Quality Thresholds:\n")
        f.write(f"  SNR > 20 (excellent): {stats['quality_thresholds']['pre_snr_gt_20']} → {stats['quality_thresholds']['post_snr_gt_20']}\n")
        f.write(f"  SNR < 10 (poor):      {stats['quality_thresholds']['pre_snr_lt_10']} → {stats['quality_thresholds']['post_snr_lt_10']}\n")
    
    return {
        'snr_pre_map': snr_pre_path,
        'snr_post_map': snr_post_path,
        'snr_diff_map': snr_diff_path,
        'stats_json': stats_path,
        'stats_csv': csv_path,
        'stats_txt': txt_path
    }


def main():
    parser = argparse.ArgumentParser(
        description="Compute voxelwise SNR before and after preprocessing for QC analysis.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single subject
  python snr_voxelwise_qc.py --sub_dir Preproc_current/sub-AD --sub_id sub-AD --out_dir qc_snr/

  # With custom paths
  python snr_voxelwise_qc.py \\
    --dwi_raw sub-AD_dwi.mif \\
    --b0_preproc mean_b0.mif \\
    --noise noise.mif \\
    --tt5 5tt_coreg.mif \\
    --sub_id sub-AD \\
    --out_dir qc_snr/
        """
    )
    
    # Subject directory mode
    parser.add_argument('--sub_dir', help='Subject directory (e.g., Preproc_current/sub-AD)')
    parser.add_argument('--sub_id', required=True, help='Subject ID (e.g., sub-AD)')
    
    # Individual file mode
    parser.add_argument('--dwi_raw', help='Path to raw DWI .mif file')
    parser.add_argument('--b0_preproc', help='Path to preprocessed mean b0 .mif file')
    parser.add_argument('--noise', help='Path to noise .mif file')
    parser.add_argument('--tt5', help='Path to 5tt_coreg.mif (for WM mask)')
    
    # Output
    parser.add_argument('--out_dir', required=True, help='Output directory for results')
    parser.add_argument('--wm_threshold', type=float, default=0.7, 
                       help='WM probability threshold (default: 0.7)')
    parser.add_argument('--wm_erode', type=int, default=2,
                       help='Number of erosion passes for deep WM (default: 2)')
    parser.add_argument('--no_wm_mask', action='store_true',
                       help='Compute SNR in whole brain instead of WM (fallback if 5tt not available)')
    
    args = parser.parse_args()
    
    # Determine file paths
    if args.sub_dir:
        sub_dir = Path(args.sub_dir)
        dwi_raw = sub_dir / f"{args.sub_id}_dwi.mif"
        b0_preproc = sub_dir / "mean_b0.mif"
        noise = sub_dir / "noise.mif"
        tt5 = sub_dir / "5tt_coreg.mif"
    else:
        if not all([args.dwi_raw, args.b0_preproc, args.noise]):
            parser.error("Either --sub_dir or all of --dwi_raw, --b0_preproc, --noise must be provided")
        dwi_raw = Path(args.dwi_raw)
        b0_preproc = Path(args.b0_preproc)
        noise = Path(args.noise)
        tt5 = Path(args.tt5) if args.tt5 else None
    
    # Check files exist
    for f, name in [(dwi_raw, 'raw DWI'), (b0_preproc, 'preprocessed b0'), (noise, 'noise map')]:
        if not f.exists():
            raise FileNotFoundError(f"{name} not found: {f}")
    
    print("=" * 80)
    print(f"VOXELWISE SNR COMPUTATION: {args.sub_id}")
    print("=" * 80)
    print(f"Raw DWI:        {dwi_raw}")
    print(f"Preproc b0:     {b0_preproc}")
    print(f"Noise map:      {noise}")
    print(f"Output dir:     {args.out_dir}")
    print("=" * 80)
    
    # Step 1: Extract raw b0
    print("\n[1/6] Extracting raw b0...")
    raw_b0_path = os.path.join(args.out_dir, f"{args.sub_id}_mean_b0_raw.mif")
    os.makedirs(args.out_dir, exist_ok=True)
    extract_raw_b0(str(dwi_raw), raw_b0_path)
    
    # Step 2: Compute global noise statistics
    print("\n[2/6] Computing global noise statistics...")
    noise_std, noise_mean, n_noise_voxels = compute_global_noise_std(str(noise))
    print(f"  Noise std: {noise_std:.3f}")
    print(f"  Noise mean: {noise_mean:.3f}")
    print(f"  Valid noise voxels: {n_noise_voxels}")
    
    # Step 3: Create WM mask if needed
    wm_mask_path = None
    use_wm_mask = not args.no_wm_mask
    
    if use_wm_mask and tt5 and tt5.exists():
        print("\n[3/6] Creating deep WM mask...")
        wm_mask_path = os.path.join(args.out_dir, f"{args.sub_id}_wm_deep_dwi.mif")
        create_wm_mask(str(tt5), str(b0_preproc), wm_mask_path, 
                      threshold=args.wm_threshold, erode_passes=args.wm_erode)
    elif use_wm_mask and (not tt5 or not tt5.exists()):
        print("\n[3/6] WARNING: No 5tt_coreg.mif found, computing SNR in whole brain...")
        use_wm_mask = False
    
    # Step 4: Compute voxelwise SNR for raw b0
    print("\n[4/6] Computing voxelwise SNR (pre-processing)...")
    snr_pre_map, snr_pre_img, valid_pre = compute_voxelwise_snr(
        raw_b0_path, noise_std, wm_mask_path, mask_output=use_wm_mask
    )
    mask_label = "WM" if use_wm_mask else "whole brain"
    print(f"  Mean SNR (pre, {mask_label}): {np.nanmean(snr_pre_map[valid_pre]):.2f}")
    
    # Step 5: Transform raw SNR to preprocessed space
    print("\n[5/6] Aligning raw SNR to preprocessed space...")
    snr_pre_transformed_path = os.path.join(args.out_dir, f"{args.sub_id}_snr_pre_transformed.nii.gz")
    snr_pre_transformed, snr_pre_transformed_img = transform_snr_to_preprocessed_space(
        snr_pre_map, snr_pre_img, str(b0_preproc), snr_pre_transformed_path
    )
    
    # Step 6: Compute voxelwise SNR for preprocessed b0
    print("\n[6/6] Computing voxelwise SNR (post-processing)...")
    snr_post_map, snr_post_img, valid_post = compute_voxelwise_snr(
        str(b0_preproc), noise_std, wm_mask_path, mask_output=use_wm_mask
    )
    print(f"  Mean SNR (post, {mask_label}): {np.nanmean(snr_post_map[valid_post]):.2f}")
    
    # Align for comparison
    if snr_pre_transformed.shape != snr_post_map.shape:
        print("  WARNING: Shape mismatch after transformation, using preprocessed space only")
        snr_pre_final = snr_pre_map  # Fallback
        valid_final = valid_post
    else:
        snr_pre_final = snr_pre_transformed
        valid_final = valid_post & ~np.isnan(snr_pre_transformed) & ~np.isnan(snr_post_map)
    
    # Compute difference map
    snr_diff_map = np.zeros_like(snr_post_map)
    snr_diff_map[:] = np.nan
    snr_diff_map[valid_final] = snr_post_map[valid_final] - snr_pre_final[valid_final]
    
    # Compute statistics
    print("\n[7/7] Computing comparison statistics...")
    stats = compute_comparison_statistics(snr_pre_final, snr_post_map, valid_final)
    
    # Save all results
    print("\n[8/8] Saving results...")
    output_files = save_results(
        args.sub_id, args.out_dir, snr_pre_final, snr_post_map, snr_diff_map,
        snr_pre_transformed_img, snr_post_img, stats, noise_std, noise_mean
    )
    
    # Print summary
    print("\n" + "=" * 80)
    print("RESULTS SUMMARY")
    print("=" * 80)
    print(f"Pre-processing SNR:  {stats['preprocessing']['mean']:.2f} ± {stats['preprocessing']['std']:.2f}")
    print(f"Post-processing SNR: {stats['postprocessing']['mean']:.2f} ± {stats['postprocessing']['std']:.2f}")
    print(f"Improvement:         {stats['improvement']['mean_delta']:.2f} ({stats['improvement']['percent_improvement']:.1f}%)")
    print(f"Correlation:         {stats['improvement']['correlation']:.3f}")
    print(f"\nOutput files saved to: {args.out_dir}")
    print(f"  - SNR maps: {output_files['snr_pre_map']}, {output_files['snr_post_map']}, {output_files['snr_diff_map']}")
    print(f"  - Statistics: {output_files['stats_json']}, {output_files['stats_csv']}, {output_files['stats_txt']}")
    print("=" * 80)


if __name__ == "__main__":
    main()

