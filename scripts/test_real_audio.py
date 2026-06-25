"""
说话人识别实际测试脚本
从 VoxCeleb 数据集中选取音频，测试模型的验证和识别能力
"""
import sys
import os
import io
from pathlib import Path
import random
import numpy as np

# Windows UTF-8 输出
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

PROJECT_ROOT = Path(r"C:\Users\Administrator\vsprint")
sys.path.insert(0, str(PROJECT_ROOT))

# 使用项目的 venv
os.chdir(PROJECT_ROOT)

from src.deploy.inference import SpeakerEncoder, SpeakerVerifier, SpeakerIdentifier


def find_audio_samples(data_dir, max_speakers=5, max_samples=3):
    """从 VoxCeleb 数据集中采集音频样本"""
    data_dir = Path(data_dir)
    speakers = sorted([d for d in data_dir.iterdir() if d.is_dir()])[:max_speakers]
    
    samples = {}
    for spk_dir in speakers:
        wavs = sorted(spk_dir.rglob("*.wav"))[:max_samples]
        if len(wavs) >= 2:
            samples[spk_dir.name] = [str(w) for w in wavs]
    
    return samples


def main():
    print("=" * 70)
    print("  ERes2NetV2 说话人识别 — 实际音频测试")
    print("=" * 70)
    
    # 1. 初始化编码器
    print("\n[1] 加载模型...")
    encoder = SpeakerEncoder(
        config_path="configs/train_config.yaml",
        checkpoint_path="checkpoints/best_model.pth",
    )
    print(f"    模型: ERes2NetV2-{encoder.config['model']['variant']}")
    print(f"    Embedding 维度: {encoder.config['model']['embedding_dim']}")
    print(f"    设备: {encoder.device}")
    
    # 2. 采集音频样本
    print("\n[2] 采集音频样本...")
    data_dir = PROJECT_ROOT / "data" / "voxceleb" / "dev"
    samples = find_audio_samples(str(data_dir), max_speakers=5, max_samples=3)
    
    print(f"    数据目录: {data_dir}")
    print(f"    采集到 {len(samples)} 个说话人:")
    for spk, wavs in samples.items():
        print(f"      {spk}: {len(wavs)} 条音频")
        for w in wavs:
            print(f"        - {Path(w).name} ({Path(w).parent.name})")
    
    if len(samples) < 2:
        print("[FAIL] 需要至少 2 个说话人的音频")
        return
    
    # 3. 提取所有音频的 embedding
    print("\n[3] 提取说话人 embedding...")
    all_files = []
    speaker_labels = {}
    for spk, wavs in samples.items():
        for w in wavs:
            all_files.append(w)
            speaker_labels[w] = spk
    
    embeddings = encoder.extract_from_files(all_files)
    print(f"    提取完成: {embeddings.shape}")
    
    # 4. 说话人验证测试 (1:1)
    print("\n" + "=" * 70)
    print("  [4] 说话人验证测试 (1:1)")
    print("=" * 70)
    
    verifier = SpeakerVerifier(
        config_path="configs/train_config.yaml",
        checkpoint_path="checkpoints/best_model.pth",
        threshold=0.3,  # 用低一些的阈值看原始分数
    )
    
    # 正样本对（同一说话人）
    print("\n--- 正样本对（同一说话人，应该判定为 TRUE）---")
    pos_results = []
    for spk, wavs in samples.items():
        if len(wavs) >= 2:
            for i in range(len(wavs)):
                for j in range(i+1, len(wavs)):
                    score, is_same = verifier.verify(wavs[i], wavs[j])
                    pos_results.append((spk, Path(wavs[i]).name, Path(wavs[j]).name, score, is_same))
                    status = "OK" if is_same else "FAIL"
                    print(f"  {status} [{spk}] {Path(wavs[i]).name} vs {Path(wavs[j]).name}")
                    print(f"    相似度: {score:.4f} | 判定: {'同一人' if is_same else '不同人'}")
    
    # 负样本对（不同说话人）
    print("\n--- 负样本对（不同说话人，应该判定为 FALSE）---")
    neg_results = []
    spk_list = list(samples.keys())
    for i in range(len(spk_list)):
        for j in range(i+1, len(spk_list)):
            spk_a, spk_b = spk_list[i], spk_list[j]
            wavs_a, wavs_b = samples[spk_a], samples[spk_b]
            # 取每个说话人的第一条音频比较
            score, is_same = verifier.verify(wavs_a[0], wavs_b[0])
            neg_results.append((spk_a, spk_b, score, is_same))
            status = "OK" if not is_same else "FAIL"
            print(f"  {status} [{spk_a} vs {spk_b}] {Path(wavs_a[0]).name} vs {Path(wavs_b[0]).name}")
            print(f"    相似度: {score:.4f} | 判定: {'同一人' if is_same else '不同人'}")
    
    # 5. 说话人识别测试 (1:N)
    print("\n" + "=" * 70)
    print("  [5] 说话人识别测试 (1:N)")
    print("=" * 70)
    
    identifier = SpeakerIdentifier(
        config_path="configs/train_config.yaml",
        checkpoint_path="checkpoints/best_model.pth",
    )
    
    # 用每个说话人的第一条音频注册
    print("\n--- 注册说话人 ---")
    for spk, wavs in samples.items():
        identifier.enroll(spk, wavs[0])
    
    # 用每个说话人的其他音频进行识别
    print("\n--- 识别测试 ---")
    correct = 0
    total = 0
    for spk, wavs in samples.items():
        for w in wavs[1:]:  # 跳过注册用的第一条
            results = identifier.identify(w, top_k=3)
            top_name, top_score = results[0]
            is_correct = (top_name == spk)
            if is_correct:
                correct += 1
            total += 1
            status = "OK" if is_correct else "FAIL"
            print(f"  {status} [{spk}] 测试音频: {Path(w).name}")
            print(f"    Top-3 识别结果:")
            for name, score in results:
                marker = " <-- correct" if name == spk else ""
                print(f"      {name}: {score:.4f}{marker}")
    
    # 6. 汇总
    print("\n" + "=" * 70)
    print("  汇总")
    print("=" * 70)
    
    pos_scores = [r[3] for r in pos_results]
    neg_scores = [r[2] for r in neg_results]
    
    print(f"\n正样本对 ({len(pos_scores)} 对):")
    print(f"  平均相似度: {np.mean(pos_scores):.4f} ± {np.std(pos_scores):.4f}")
    print(f"  范围: [{np.min(pos_scores):.4f}, {np.max(pos_scores):.4f}]")
    pos_correct = sum(1 for r in pos_results if r[4])
    print(f"  正确率: {pos_correct}/{len(pos_results)} ({pos_correct/len(pos_results)*100:.1f}%)")
    
    print(f"\n负样本对 ({len(neg_scores)} 对):")
    print(f"  平均相似度: {np.mean(neg_scores):.4f} ± {np.std(neg_scores):.4f}")
    print(f"  范围: [{np.min(neg_scores):.4f}, {np.max(neg_scores):.4f}]")
    neg_correct = sum(1 for r in neg_results if not r[3])
    print(f"  正确率: {neg_correct}/{len(neg_results)} ({neg_correct/len(neg_results)*100:.1f}%)")
    
    print(f"\n说话人识别 (1:N):")
    print(f"  Top-1 准确率: {correct}/{total} ({correct/total*100:.1f}%)")
    
    # 可视化分数分布
    print(f"\n分数分布:")
    pos_bar = "#" * max(1, int(np.mean(pos_scores) * 40))
    neg_bar = "#" * max(1, int(np.mean(neg_scores) * 40))
    print(f"  正样本: {pos_bar:<40} {np.mean(pos_scores):.4f}")
    print(f"  负样本: {neg_bar:<40} {np.mean(neg_scores):.4f}")
    print(f"  间隔:   {'─' * 40}")
    gap = np.mean(pos_scores) - np.mean(neg_scores)
    print(f"  正负样本间隔: {gap:.4f}")
    
    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
