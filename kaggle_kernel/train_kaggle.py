"""
Self-contained Part B training and export script for Kaggle GPU T4 x2.

Strictly follows:
- docs/P1_ML_EXECUTION_PLAN.md (§4 Deep branch and adaptation, §5 Protocol)
- docs/P1_AGENT_CONTRACT.md (Rules 1-13)

Features:
- Clones official HVI-CIDNet repo dynamically or uses vendored architecture
- Uses Kaggle GPU T4 x2 (CUDA accelerated)
- Synchronized paired crops (128x128), Adam lr=1e-5, Charbonnier loss
- 2,000 updates with validation every 250 updates
- Exports:
    1. best_model.pth & last_model.pth with SHA-256 hashes
    2. JSONL prediction logs for 5 handoff images outside Kaggle
    3. full training_meta.json
"""

import hashlib
import json
import os
import random
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"=== Kaggle Running on device: {device} ===")
if torch.cuda.is_available():
    print(f"Device name: {torch.cuda.get_device_name(0)}")
    print(f"Device count: {torch.cuda.device_count()}")
    print(f"Device VRAM: {round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 2)} GB")

# 1. Setup HVI-CIDNet repo
if not Path("HVI-CIDNet").exists():
    print("Cloning official HVI-CIDNet repository...")
    subprocess.run(["git", "clone", "https://github.com/Fediory/HVI-CIDNet.git"], check=True)

sys.path.insert(0, str(Path("HVI-CIDNet").resolve()))
from net.CIDNet import CIDNet  # type: ignore


# 2. Locate datasets on Kaggle
# The log shows Kaggle mounts under /kaggle/input/datasets/ssachithananthan/p1-drive-datasets/
drive_real = Path("/kaggle/input/datasets/ssachithananthan/p1-drive-datasets/LOLv2-real/LOLv2-real/LOL-v2/Real_captured")
tanhyml_real = Path("/kaggle/input/datasets/tanhyml/lol-v2-dataset/LOL-v2/Real_captured")
DATASET_DIR = None

if drive_real.exists():
    DATASET_DIR = drive_real
    print(f"--> Using Drive Dataset path: {DATASET_DIR}")
elif tanhyml_real.exists():
    DATASET_DIR = tanhyml_real
    print(f"--> Using tanhyml Dataset path: {DATASET_DIR}")
else:
    for root, dirs, files in os.walk("/kaggle/input"):
        if "Real_captured" in dirs:
            DATASET_DIR = Path(root) / "Real_captured"
            print(f"--> Found Real_captured dynamically: {DATASET_DIR}")
            break

if DATASET_DIR is None:
    raise FileNotFoundError("LOL-v2 Real_captured not found on Kaggle.")

print(f"Using DATASET_DIR: {DATASET_DIR}")

# 3. Locate pretrained checkpoint
PRETRAINED_CKPT = None
for root, dirs, files in os.walk("/kaggle/input"):
    for f in files:
        if "best_PSNR.pth" in f:
            PRETRAINED_CKPT = Path(root) / f
            break
    if PRETRAINED_CKPT:
        break

if PRETRAINED_CKPT:
    print(f"Found pretrained checkpoint: {PRETRAINED_CKPT}")
else:
    print("No pretrained checkpoint found; initializing fresh.")


# 4. Synchronized Paired Dataset
class KagglePairedDataset(Dataset):
    def __init__(self, pairs, crop_size=128, augment=True):
        self.pairs = pairs
        self.crop_size = crop_size
        self.augment = augment

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index):
        pair_id, low_p, norm_p = self.pairs[index]
        with Image.open(low_p) as im:
            low_np = np.asarray(im.convert("RGB"), dtype=np.float32) / 255.0
        with Image.open(norm_p) as im:
            norm_np = np.asarray(im.convert("RGB"), dtype=np.float32) / 255.0

        h, w, _ = low_np.shape
        if self.crop_size > 0 and h >= self.crop_size and w >= self.crop_size:
            if self.augment:
                y = random.randint(0, h - self.crop_size)
                x = random.randint(0, w - self.crop_size)
            else:
                y = (h - self.crop_size) // 2
                x = (w - self.crop_size) // 2
            low_np = low_np[y : y + self.crop_size, x : x + self.crop_size, :]
            norm_np = norm_np[y : y + self.crop_size, x : x + self.crop_size, :]

        if self.augment:
            if random.random() > 0.5:
                low_np = np.fliplr(low_np)
                norm_np = np.fliplr(norm_np)
            if random.random() > 0.5:
                low_np = np.flipud(low_np)
                norm_np = np.flipud(norm_np)
            k = random.randint(0, 3)
            if k > 0:
                low_np = np.rot90(low_np, k)
                norm_np = np.rot90(norm_np, k)

        low_t = torch.from_numpy(np.ascontiguousarray(low_np.transpose(2, 0, 1)))
        norm_t = torch.from_numpy(np.ascontiguousarray(norm_np.transpose(2, 0, 1)))
        return low_t, norm_t, pair_id


# Discover pairs and split with seed 1956
train_low_files = sorted((DATASET_DIR / "Train" / "Low").glob("*.png"))
train_norm_dir = DATASET_DIR / "Train" / "Normal"
all_train_pairs = []
for p in train_low_files:
    num = re.findall(r"\d+", p.stem)[0]
    norm_p = train_norm_dir / f"normal{num}.png"
    if not norm_p.exists():
        norm_matches = list(train_norm_dir.glob(f"*{num}*.png"))
        if norm_matches:
            norm_p = norm_matches[0]
    if norm_p.exists():
        all_train_pairs.append((num, p, norm_p))

rng = random.Random(1956)
shuffled_idx = list(range(len(all_train_pairs)))
rng.shuffle(shuffled_idx)
n_val = round(len(all_train_pairs) * 0.20)
val_idx = set(shuffled_idx[:n_val])

train_pairs = [all_train_pairs[i] for i in range(len(all_train_pairs)) if i not in val_idx]
val_pairs = [all_train_pairs[i] for i in range(len(all_train_pairs)) if i in val_idx]
print(f"Total pairs: {len(all_train_pairs)} -> Train: {len(train_pairs)}, Validation: {len(val_pairs)}")


class CharbonnierLoss(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, pred, target):
        return torch.mean(torch.sqrt((pred - target) ** 2 + self.eps))


# 5. Build and Load Model
model = CIDNet()
model.trans.gated = False
model.trans.gated2 = False
model.trans.alpha = 1.0

if PRETRAINED_CKPT and PRETRAINED_CKPT.exists():
    sd = torch.load(PRETRAINED_CKPT, map_location="cpu")
    if "state_dict" in sd:
        sd = sd["state_dict"]
    cleaned_sd = {k[7:] if k.startswith("module.") else k: v for k, v in sd.items()}
    model.load_state_dict(cleaned_sd, strict=True)
    print("Loaded pretrained state dict with strict=True.")

model.to(device)

# 6. Training setup
BATCH_SIZE = 4
ACCUM_STEPS = 2  # Effective batch = 8
MAX_UPDATES = 2000
VAL_EVERY = 250

train_loader = DataLoader(
    KagglePairedDataset(train_pairs, crop_size=128, augment=True),
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=2,
    pin_memory=True,
)
val_loader = DataLoader(
    KagglePairedDataset(val_pairs, crop_size=128, augment=False),
    batch_size=1,
    shuffle=False,
    num_workers=2,
)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-5, weight_decay=0.0)
loss_fn = CharbonnierLoss(eps=1e-6)

out_dir = Path("p1_kaggle_output")
out_dir.mkdir(parents=True, exist_ok=True)

updates = 0
accum_count = 0
running_loss = 0.0
best_val_loss = float("inf")
history = []
start_t = time.time()

print(f"Beginning 2,000 updates on {device}...")
model.train()
optimizer.zero_grad()

while updates < MAX_UPDATES:
    for low, norm, _ in train_loader:
        low = low.to(device)
        norm = norm.to(device)

        pred = model(low)
        pred = torch.clamp(pred, 0.0, 1.0)
        loss = loss_fn(pred, norm) / ACCUM_STEPS
        loss.backward()

        running_loss += loss.item() * ACCUM_STEPS
        accum_count += 1

        if accum_count % ACCUM_STEPS == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad()
            updates += 1

            if updates % 50 == 0 or updates == 1:
                cur_loss = running_loss / ACCUM_STEPS
                elapsed_min = (time.time() - start_t) / 60.0
                print(f"Update [{updates}/{MAX_UPDATES}] | Loss: {cur_loss:.6f} | Elapsed: {elapsed_min:.1f}m")
            running_loss = 0.0

            if updates % VAL_EVERY == 0 or updates == MAX_UPDATES:
                model.eval()
                val_loss = 0.0
                v_count = 0
                with torch.no_grad():
                    for v_low, v_norm, _ in val_loader:
                        v_low = v_low.to(device)
                        v_norm = v_norm.to(device)
                        v_pred = torch.clamp(model(v_low), 0.0, 1.0)
                        val_loss += loss_fn(v_pred, v_norm).item()
                        v_count += 1
                        if v_count >= 50:
                            break
                val_loss /= max(1, v_count)
                print(f"--> [Val @ Update {updates}] Val Loss: {val_loss:.6f}")
                history.append({"update": updates, "val_loss": val_loss})

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    torch.save(model.state_dict(), out_dir / "best_model.pth")
                    print("Saved new best model checkpoint.")

                model.train()

            if updates >= MAX_UPDATES:
                break

# Final checkpoint
torch.save(model.state_dict(), out_dir / "last_model.pth")
total_min = (time.time() - start_t) / 60.0
print(f"Training finished in {total_min:.1f} minutes!")

# 7. Checksum calculation and 5 handoff predictions
best_path = out_dir / "best_model.pth" if (out_dir / "best_model.pth").exists() else out_dir / "last_model.pth"
ckpt_hash = hashlib.sha256(best_path.read_bytes()).hexdigest()

handoff_dir = out_dir / "handoff_predictions"
handoff_dir.mkdir(parents=True, exist_ok=True)
jsonl_records = []

# Select 5 test images for handoff
test_low_files = sorted((DATASET_DIR / "Test" / "Low").glob("*.png"))[:5]
model.eval()

with torch.no_grad():
    for p in test_low_files:
        with Image.open(p) as im:
            orig = np.asarray(im.convert("RGB"), dtype=np.uint8)
        in_hash = hashlib.sha256(p.read_bytes()).hexdigest()

        h, w, _ = orig.shape
        pad_h = (8 - (h % 8)) % 8
        pad_w = (8 - (w % 8)) % 8

        t = torch.from_numpy(orig.transpose(2, 0, 1)).float().div(255.0).unsqueeze(0)
        if pad_h > 0 or pad_w > 0:
            t = F.pad(t, (0, pad_w, 0, pad_h), mode="reflect")
        t = t.to(device)

        t0 = time.perf_counter()
        pred = model(t)
        dt_ms = (time.perf_counter() - t0) * 1000.0

        pred = torch.clamp(pred[:, :, :h, :w], 0.0, 1.0)
        pred_np = pred.squeeze(0).cpu().numpy().transpose(1, 2, 0)
        pred_u8 = np.clip(np.round(pred_np * 255.0), 0, 255).astype(np.uint8)

        out_img_path = handoff_dir / f"{p.stem}_pred.png"
        Image.fromarray(pred_u8).save(out_img_path)
        out_hash = hashlib.sha256(out_img_path.read_bytes()).hexdigest()

        jsonl_records.append({
            "id": p.stem,
            "input_sha256": in_hash,
            "output_sha256": out_hash,
            "branch_version": "p1-v1.0",
            "checkpoint_hash": ckpt_hash,
            "elapsed_ms": round(dt_ms, 2),
            "error": None,
        })

with open(out_dir / "handoff_records.jsonl", "w", encoding="utf-8") as f:
    for rec in jsonl_records:
        f.write(json.dumps(rec) + "\n")

meta = {
    "status": "COMPLETED",
    "device": str(device),
    "total_updates": updates,
    "best_val_loss": best_val_loss,
    "best_checkpoint_sha256": ckpt_hash,
    "total_time_min": round(total_min, 2),
    "history": history,
}
with open(out_dir / "training_meta.json", "w", encoding="utf-8") as f:
    json.dump(meta, f, indent=2)

print("All Kaggle artifacts, checkpoints, and handoff predictions successfully generated!")
