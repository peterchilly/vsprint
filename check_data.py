import os
import numpy as np

# Check data statistics
feature_dir = "data/voxceleb/features"
speakers = [d for d in os.listdir(feature_dir) if os.path.isdir(os.path.join(feature_dir, d))]
print(f"Speakers: {len(speakers)}")

# Count total files
total_files = sum(len(files) for _, _, files in os.walk(feature_dir))
print(f"Total .npy files: {total_files}")

# Check a few samples
sample_path = os.path.join(feature_dir, "id10001", "1zcIwhmdeo4", "00001.npy")
f = np.load(sample_path)
print(f"\nSample: {sample_path}")
print(f"  Shape: {f.shape}")
print(f"  Dtype: {f.dtype}")
print(f"  Min: {f.min():.4f}")
print(f"  Max: {f.max():.4f}")
print(f"  Mean: {f.mean():.4f}")
print(f"  Std: {f.std():.4f}")

# Check more samples from different speakers
import random
random.seed(42)
sampled = random.sample(speakers, min(5, len(speakers)))
print(f"\nSampling 5 speakers for stats:")
all_means = []
all_stds = []
for spk in sampled:
    spk_dir = os.path.join(feature_dir, spk)
    npy_files = []
    for root, dirs, files in os.walk(spk_dir):
        for fn in files:
            if fn.endswith('.npy'):
                npy_files.append(os.path.join(root, fn))
    if npy_files:
        f = np.load(npy_files[0])
        all_means.append(f.mean())
        all_stds.append(f.std())
        print(f"  {spk}: shape={f.shape}, mean={f.mean():.4f}, std={f.std():.4f}")

print(f"\nOverall mean of means: {np.mean(all_means):.4f}")
print(f"Overall mean of stds: {np.mean(all_stds):.4f}")

# Check label distribution
print(f"\nLabel distribution (samples per speaker):")
counts = []
for spk in speakers:
    spk_dir = os.path.join(feature_dir, spk)
    count = sum(len(files) for _, _, files in os.walk(spk_dir))
    counts.append(count)
counts = np.array(counts)
print(f"  Min: {counts.min()}")
print(f"  Max: {counts.max()}")
print(f"  Mean: {counts.mean():.1f}")
print(f"  Median: {np.median(counts):.1f}")
print(f"  Total: {counts.sum()}")
