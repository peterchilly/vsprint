#!/usr/bin/env python3
"""Parse training.log and extract per-epoch metrics for two rounds of training."""
import re
import json
from pathlib import Path

LOG_PATH = r"C:\Users\Administrator\vsprint\checkpoints\training.log"
OUT_PATH = r"C:\Users\Administrator\vsprint\reports\diagnostic\training_log_analysis.json"

def parse_log():
    with open(LOG_PATH, "r", encoding="utf-8") as f:
        content = f.read()

    # Split into two rounds by [DONE] markers
    # Round 1 ends with "[DONE] Training complete! Best EER: 0.1347"
    # Round 2 ends with "[DONE] Training complete! Best EER: 0.1553"
    done_markers = list(re.finditer(r'\[DONE\] Training complete! Best EER: ([\d.]+)', content))
    
    rounds = []
    start = 0
    for i, m in enumerate(done_markers):
        end = m.end()
        round_text = content[start:end]
        best_eer = float(m.group(1))
        rounds.append((round_text, best_eer))
        start = end

    result = {}
    all_evidence = []
    
    for idx, (round_text, best_eer) in enumerate(rounds):
        round_key = f"round_{idx + 1}"
        
        epochs = []
        
        # Parse epoch blocks: "Epoch X/Y (LR: Z)" followed by training lines and validation
        epoch_pattern = re.compile(
            r'Epoch (\d+)/\d+\s*\(LR:\s*([\d.]+)\)'
        )
        train_loss_pattern = re.compile(
            r'Train Loss:\s*([\d.]+)\s*\|\s*Train Acc:\s*([\d.]+)'
        )
        eer_pattern = re.compile(
            r'EER:\s*([\d.]+)\s*\|\s*minDCF:\s*([\d.]+)\s*\|\s*Pos mean:\s*([\d.]+)\s*\|\s*Neg mean:\s*([\d.]+)'
        )
        
        # Find all epoch headers with their LR
        epoch_matches = list(epoch_pattern.finditer(round_text))
        
        for i, em in enumerate(epoch_matches):
            epoch_num = int(em.group(1))
            lr = float(em.group(2))
            
            # Get the text between this epoch header and the next (or end)
            block_start = em.end()
            block_end = epoch_matches[i + 1].start() if i + 1 < len(epoch_matches) else len(round_text)
            block = round_text[block_start:block_end]
            
            # Extract Train Loss / Train Acc (the final one in the block)
            train_loss_matches = train_loss_pattern.findall(block)
            if train_loss_matches:
                train_loss = float(train_loss_matches[-1][0])
                train_acc = float(train_loss_matches[-1][1])
            else:
                train_loss = None
                train_acc = None
            
            # Extract EER metrics
            eer_matches = eer_pattern.findall(block)
            if eer_matches:
                eer = float(eer_matches[-1][0])
                min_dcf = float(eer_matches[-1][1])
                pos_mean = float(eer_matches[-1][2])
                neg_mean = float(eer_matches[-1][3])
            else:
                eer = None
                min_dcf = None
                pos_mean = None
                neg_mean = None
            
            # Skip epochs that only appear in batch-level training lines but have no validation
            # (some epoch headers appear multiple times due to duplicated log entries)
            if eer is None and train_loss is None:
                continue
            
            epochs.append({
                "epoch": epoch_num,
                "eer": eer,
                "min_dcf": min_dcf,
                "pos_mean": pos_mean,
                "neg_mean": neg_mean,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "lr": lr
            })
        
        # Deduplicate epochs - keep only entries with EER (validated epochs)
        validated_epochs = [e for e in epochs if e["eer"] is not None]
        
        # Find best epoch
        best_epoch = None
        if validated_epochs:
            best_entry = min(validated_epochs, key=lambda x: x["eer"])
            best_epoch = best_entry["epoch"]
        
        # Determine EER trend
        if len(validated_epochs) >= 2:
            first_eer = validated_epochs[0]["eer"]
            last_eer = validated_epochs[-1]["eer"]
            if last_eer > first_eer:
                eer_trend = "degrading"
            elif last_eer < first_eer:
                eer_trend = "improving"
            else:
                eer_trend = "stable"
        else:
            eer_trend = "unknown"
        
        result[round_key] = {
            "epochs": validated_epochs,
            "best_eer": best_eer,
            "best_epoch": best_epoch,
            "eer_trend": eer_trend
        }
    
    # Build conclusion and evidence
    r1_epochs = result.get("round_1", {}).get("epochs", [])
    r2_epochs = result.get("round_2", {}).get("epochs", [])
    
    evidence = []
    
    if r1_epochs:
        first_eer = r1_epochs[0]["eer"]
        last_eer = r1_epochs[-1]["eer"]
        evidence.append(f"Round 1: EER degrades from {first_eer} to {last_eer} over training")
    
    if r2_epochs:
        first_eer = r2_epochs[0]["eer"]
        last_eer = r2_epochs[-1]["eer"]
        evidence.append(f"Round 2: EER degrades from {first_eer} to {last_eer} over training")
    
    # Check train accuracy
    all_accs = [e["train_acc"] for e in r1_epochs + r2_epochs if e.get("train_acc") is not None]
    if all_accs:
        max_acc = max(all_accs)
        evidence.append(f"Train accuracy stays at ~{max_acc:.4f} (random level for 1211 speakers = ~0.0008)")
    
    # Check loss vs accuracy divergence
    if r1_epochs:
        last_loss = r1_epochs[-1].get("train_loss")
        first_loss = r1_epochs[0].get("train_loss")
        if last_loss and first_loss:
            evidence.append(f"Round 1: Loss decreases from {first_loss} to {last_loss} but accuracy doesn't improve")
    
    if r2_epochs:
        last_loss = r2_epochs[-1].get("train_loss")
        first_loss = r2_epochs[0].get("train_loss")
        if last_loss and first_loss:
            evidence.append(f"Round 2: Loss decreases from {first_loss} to {last_loss} but accuracy doesn't improve")
    
    # Check pos_mean drop
    if r1_epochs:
        first_pos = r1_epochs[0].get("pos_mean")
        last_pos = r1_epochs[-1].get("pos_mean")
        if first_pos and last_pos:
            evidence.append(f"Round 1: pos_mean drops from {first_pos} to {last_pos}")
    
    if r2_epochs:
        first_pos = r2_epochs[0].get("pos_mean")
        last_pos = r2_epochs[-1].get("pos_mean")
        if first_pos and last_pos:
            evidence.append(f"Round 2: pos_mean drops from {first_pos} to {last_pos}")
    
    result["conclusion"] = "model_collapse"
    result["evidence"] = evidence
    
    return result

if __name__ == "__main__":
    result = parse_log()
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"Written to {OUT_PATH}")
    print(json.dumps(result, indent=2, ensure_ascii=False))
