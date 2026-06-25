import os, sys, time, traceback, signal
sys.path.insert(0, '.')
from pathlib import Path
import numpy as np
from scripts.extract_features import AudioPreprocessor

pp = AudioPreprocessor(sample_rate=16000, n_mels=80)

data_dir = 'F:/voxceleb2/dev/wav'
output_dir = 'F:/voxceleb2/dev/features'
os.makedirs(output_dir, exist_ok=True)

speaker_dirs = sorted([d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))])
print(f'Speaker dirs: {len(speaker_dirs)}', flush=True)

start = time.time()
success = 0
failed = 0
skipped = 0
total_done = 0

for si, speaker in enumerate(speaker_dirs):
    spk_path = os.path.join(data_dir, speaker)
    try:
        for root, dirs, files in os.walk(spk_path):
            for f in files:
                if not f.endswith('.wav'):
                    continue
                audio_path = os.path.join(root, f)
                rel = os.path.relpath(audio_path, data_dir)
                out_path = os.path.join(output_dir, rel.replace('.wav', '.npy'))
                
                if os.path.exists(out_path):
                    skipped += 1
                    total_done += 1
                    continue
                
                os.makedirs(os.path.dirname(out_path), exist_ok=True)
                try:
                    fbank = pp.process_file(audio_path)
                    np.save(out_path, fbank.astype(np.float32))
                    success += 1
                except Exception as e:
                    print(f'FAIL: {rel} | {type(e).__name__}: {e}', flush=True)
                    failed += 1
                
                total_done += 1
                if total_done % 1000 == 0:
                    elapsed = time.time() - start
                    rate = total_done / elapsed if elapsed > 0 else 0
                    print(f'PROGRESS: {total_done} | success={success} skip={skipped} fail={failed} | {rate:.1f} f/s | {elapsed/60:.1f}min', flush=True)
    except Exception as e:
        print(f'SPK_FAIL: {speaker} | {type(e).__name__}: {e}', flush=True)
        traceback.print_exc()
        continue
    
    if (si + 1) % 100 == 0:
        elapsed = time.time() - start
        print(f'SPEAKER: {si+1}/{len(speaker_dirs)} ({speaker}) | total={total_done} | {elapsed/60:.1f}min', flush=True)

elapsed = time.time() - start
print(f'COMPLETE: success={success} skip={skipped} fail={failed} | Time: {elapsed/3600:.2f}h', flush=True)
