#!/usr/bin/env python3
"""Validate data pipeline: check .npy feature files for shape, dtype, NaN/Inf, CMVN, label alignment, speaker distribution."""
import os
import json
import random
import numpy as np
from pathlib import Path
from collections import Counter

FEATURES_DIR = r"C:\Users\Administrator\vsprint\data\voxceleb\features"
OUT_PATH = r"C:\Users\Administrator\vsprint\reports\diagnostic\data_pipeline_report.json"

def collect_npy_files(root):
    """Recursively collect all .npy files."""
    files = []
    for dirpath, dirnames, filenames in os.walk(root):
        for fn in filenames:
            if fn.endswith(".npy"):
                files.append(os.path.join(dirpath, fn))
    return files

def check_features(file_list, n_samples=100):
    """Randomly sample n_files and check shape, dtype, value range, NaN/Inf."""
    if len(file_list) > n_samples:
        sampled = random.sample(file_list, n_samples)
    else:
        sampled = file_list
        n_samples = len(sampled)
    
    shapes = set()
    dtypes = set()
    global_min = float("inf")
    global_max = float("-inf")
    has_nan = False
    has_inf = False
    
    for fp in sampled:
        arr = np.load(fp, allow_pickle=False)
        shapes.add(arr.shape)
        dtypes.add(str(arr.dtype))
        local_min = float(np.min(arr))
        local_max = float(np.max(arr))
        if local_min < global_min:
            global_min = local_min
        if local_max > global_max:
            global_max = local_max
        if np.isnan(arr).any():
            has_nan = True
        if np.isinf(arr).any():
            has_inf = True
    
    # Determine n_mels from shape (assume last dim is n_mels or first)
    n_mels = None
    for shape in shapes:
        if len(shape) == 2:
            n_mels = shape[-1]
            break
        elif len(shape) == 1:
            n_mels = shape[0]
    
    return {
        "n_samples": n_samples,
        "n_mels": n_mels,
        "dtype": list(dtypes)[0] if len(dtypes) == 1 else list(dtypes),
        "value_range": {"min": round(global_min, 4), "max": round(global_max, 4)},
        "has_nan": has_nan,
        "has_inf": has_inf,
        "shapes": [list(s) for s in shapes]
    }

def check_cmvn(file_list, n_samples=200):
    """Check if CMVN has been applied by computing global mean/std."""
    if len(file_list) > n_samples:
        sampled = random.sample(file_list, n_samples)
    else:
        sampled = file_list
    
    all_values = []
    for fp in sampled:
        arr = np.load(fp, allow_pickle=False)
        all_values.append(arr.flatten())
    
    concat = np.concatenate(all_values)
    mean = float(np.mean(concat))
    std = float(np.std(concat))
    
    # If mean is close to 0 and std close to 1, CMVN has been applied
    cmvn_applied = abs(mean) < 0.1 and abs(std - 1.0) < 0.2
    
    return {
        "cmvn_status": "applied" if cmvn_applied else "not_applied",
        "global_mean": round(mean, 4),
        "global_std": round(std, 4)
    }

def check_label_alignment(file_list, n_samples=100):
    """Verify that speaker ID in file path is consistent: the idXXXXX directory
    should be an ancestor of the .npy file (structure: features/idXXXXX/video_id/xxx.npy)."""
    if len(file_list) > n_samples:
        sampled = random.sample(file_list, n_samples)
    else:
        sampled = file_list
    
    mismatches = 0
    checked = 0
    for fp in sampled:
        parts = Path(fp).parts
        # Find all speaker directories in path (format: idXXXXX)
        speaker_dirs = [p for p in parts if p.startswith("id") and len(p) > 2]
        
        if not speaker_dirs:
            mismatches += 1
            checked += 1
            continue
        
        # There should be exactly one speaker directory in the path
        if len(speaker_dirs) >= 1:
            # Verify the speaker dir is a direct ancestor (2 levels up from the file)
            # Structure: .../features/idXXXXX/video_id/xxx.npy
            # So parts[-3] should be idXXXXX
            if len(parts) >= 3 and parts[-3].startswith("id"):
                checked += 1
            else:
                mismatches += 1
                checked += 1
        else:
            mismatches += 1
            checked += 1
    
    if mismatches == 0:
        return "correct"
    else:
        return f"mismatch ({mismatches}/{checked})"

def compute_speaker_distribution(root):
    """Count .npy files per speaker directory."""
    speaker_counts = Counter()
    for dirpath, dirnames, filenames in os.walk(root):
        for fn in filenames:
            if fn.endswith(".npy"):
                # Find speaker ID from path
                parts = Path(dirpath).parts
                for p in parts:
                    if p.startswith("id"):
                        speaker_counts[p] += 1
                        break
    
    counts = list(speaker_counts.values())
    if not counts:
        return {
            "n_speakers": 0,
            "min_samples": 0,
            "max_samples": 0,
            "mean_samples": 0,
            "cv": 0
        }
    
    mean_samples = float(np.mean(counts))
    std_samples = float(np.std(counts))
    cv = float(std_samples / mean_samples) if mean_samples > 0 else 0
    
    return {
        "n_speakers": len(counts),
        "min_samples": int(min(counts)),
        "max_samples": int(max(counts)),
        "mean_samples": round(mean_samples, 2),
        "cv": round(cv, 4)
    }

def main():
    random.seed(42)
    
    print("Collecting .npy files...")
    all_files = collect_npy_files(FEATURES_DIR)
    print(f"Found {len(all_files)} .npy files")
    
    if not all_files:
        print("ERROR: No .npy files found!")
        return
    
    print("Checking features (100 random samples)...")
    feature_check = check_features(all_files, n_samples=100)
    print(f"  Shapes: {feature_check['shapes']}")
    print(f"  dtype: {feature_check['dtype']}")
    print(f"  value_range: {feature_check['value_range']}")
    print(f"  has_nan: {feature_check['has_nan']}")
    print(f"  has_inf: {feature_check['has_inf']}")
    
    print("Checking CMVN status (200 samples)...")
    cmvn_result = check_cmvn(all_files, n_samples=200)
    print(f"  status: {cmvn_result['cmvn_status']}")
    print(f"  global_mean: {cmvn_result['global_mean']}")
    print(f"  global_std: {cmvn_result['global_std']}")
    
    print("Checking label alignment (100 samples)...")
    label_alignment = check_label_alignment(all_files, n_samples=100)
    print(f"  alignment: {label_alignment}")
    
    print("Computing speaker distribution...")
    speaker_dist = compute_speaker_distribution(FEATURES_DIR)
    print(f"  n_speakers: {speaker_dist['n_speakers']}")
    print(f"  min_samples: {speaker_dist['min_samples']}")
    print(f"  max_samples: {speaker_dist['max_samples']}")
    print(f"  mean_samples: {speaker_dist['mean_samples']}")
    print(f"  cv: {speaker_dist['cv']}")
    
    report = {
        "feature_check": {
            "n_samples": feature_check["n_samples"],
            "n_mels": feature_check["n_mels"],
            "dtype": feature_check["dtype"],
            "value_range": feature_check["value_range"],
            "has_nan": feature_check["has_nan"],
            "has_inf": feature_check["has_inf"]
        },
        "cmvn_status": cmvn_result["cmvn_status"],
        "cmvn_details": {
            "global_mean": cmvn_result["global_mean"],
            "global_std": cmvn_result["global_std"]
        },
        "label_alignment": label_alignment,
        "speaker_distribution": speaker_dist
    }
    
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    print(f"\nReport written to {OUT_PATH}")
    print(json.dumps(report, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()
