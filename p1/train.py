"""
Part B: HVI-CIDNet Fine-Tuning Script.

Strictly follows docs/P1_ML_EXECUTION_PLAN.md Section 4 & docs/P1_AGENT_CONTRACT.md:
- Uses audited paired training/validation split (LOL-v2 Real)
- Adam optimizer, learning rate 1e-5, weight decay 0, grad clip norm 1.0
- Synchronized paired crops (128x128), batch size 1, effective batch 8 via accumulation
- Charbonnier loss mean(sqrt((pred - target)^2 + 1e-6))
- Validates every 250 optimizer updates, caps at 2,000 updates
- Exports best and last checkpoints with full metadata & SHA-256 hashes
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

_REPO_ROOT = Path(__file__).resolve().parent.parent
_EXTERNAL_CIDNET = _REPO_ROOT / "external" / "HVI-CIDNet"
if str(_EXTERNAL_CIDNET) not in sys.path:
    sys.path.insert(0, str(_EXTERNAL_CIDNET))

from net.CIDNet import CIDNet  # type: ignore
from p1.dataset import PairedLowLightDataset


class CharbonnierLoss(nn.Module):
    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return torch.mean(torch.sqrt((pred - target) ** 2 + self.eps))


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def validate(
    model: nn.Module,
    val_loader: DataLoader,
    device: torch.device,
    loss_fn: nn.Module,
    max_val_samples: int = 50,
) -> float:
    model.eval()
    total_loss = 0.0
    count = 0
    with torch.no_grad():
        for low, norm, _ in val_loader:
            low = low.to(device)
            norm = norm.to(device)
            out = model(low)
            out = torch.clamp(out, 0.0, 1.0)
            loss = loss_fn(out, norm)
            total_loss += loss.item()
            count += 1
            if count >= max_val_samples:
                break
    return total_loss / max(1, count)


def main() -> None:
    parser = argparse.ArgumentParser(description="P1 Part B: HVI-CIDNet Fine-Tuning")
    parser.add_argument("--manifest", type=str, default="data_manifests/LOLv2-Real-paired.csv")
    parser.add_argument("--pretrained-ckpt", type=str, default="weights/cidnet/CIDNETweights/lolv2/best_PSNR.pth")
    parser.add_argument("--output-dir", type=str, default="weights/part_b_finetuned")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--accum-steps", type=int, default=8)
    parser.add_argument("--crop-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--max-updates", type=int, default=2000)
    parser.add_argument("--val-every", type=int, default=250)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--debug-sanity", action="store_true", help="Run 200-step debug sanity run on tiny batch")
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Build model
    model = CIDNet()
    model.trans.gated = False
    model.trans.gated2 = False
    model.trans.alpha = 1.0

    # Load initial pretrained checkpoint
    ckpt_p = Path(args.pretrained_ckpt)
    if not ckpt_p.is_absolute():
        ckpt_p = _REPO_ROOT / ckpt_p

    if ckpt_p.exists():
        sd = torch.load(ckpt_p, map_location="cpu", weights_only=True)
        cleaned_sd = {k[7:] if k.startswith("module.") else k: v for k, v in sd.items()}
        model.load_state_dict(cleaned_sd, strict=True)
        print(f"Loaded pretrained backbone from {ckpt_p}")
    else:
        print(f"Warning: Pretrained checkpoint {ckpt_p} not found; initializing fresh.")

    model.to(device)

    # Datasets
    train_dataset = PairedLowLightDataset(
        args.manifest,
        split="train",
        crop_size=args.crop_size,
        augment=True,
        repo_root=_REPO_ROOT,
    )
    val_dataset = PairedLowLightDataset(
        args.manifest,
        split="validation",
        crop_size=args.crop_size,
        augment=False,
        repo_root=_REPO_ROOT,
    )

    if args.debug_sanity:
        # Debug sanity run on 8 samples for 200 updates per Contract Rule 8
        train_dataset.samples = train_dataset.samples[:8]
        args.max_updates = 200
        args.val_every = 50
        print("Running in DEBUG-SANITY mode: 8 samples, 200 updates.")

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=0.0)
    loss_fn = CharbonnierLoss(eps=1e-6)

    out_dir = Path(args.output_dir)
    if not out_dir.is_absolute():
        out_dir = _REPO_ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    optimizer_updates = 0
    accum_count = 0
    running_loss = 0.0
    best_val_loss = float("inf")
    log_history = []
    initial_loss = None

    print(f"Starting training: max_updates={args.max_updates}, effective_bs={args.batch_size * args.accum_steps}...")
    start_train_time = time.time()

    model.train()
    optimizer.zero_grad()

    while optimizer_updates < args.max_updates:
        for low, norm, _ in train_loader:
            low = low.to(device)
            norm = norm.to(device)

            pred = model(low)
            pred = torch.clamp(pred, 0.0, 1.0)
            loss = loss_fn(pred, norm) / args.accum_steps
            loss.backward()

            running_loss += loss.item() * args.accum_steps
            accum_count += 1

            if accum_count % args.accum_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()
                optimizer_updates += 1

                avg_step_loss = running_loss / args.accum_steps
                running_loss = 0.0

                if initial_loss is None:
                    initial_loss = avg_step_loss

                if optimizer_updates % 25 == 0 or optimizer_updates == 1:
                    print(f"Update [{optimizer_updates}/{args.max_updates}] - Step Loss: {avg_step_loss:.6f}")

                # Validation & Checkpointing
                if optimizer_updates % args.val_every == 0 or optimizer_updates == args.max_updates:
                    val_loss = validate(model, val_loader, device, loss_fn)
                    print(f"--> [Validation @ Update {optimizer_updates}] Val Loss: {val_loss:.6f}")
                    log_history.append({"update": optimizer_updates, "train_loss": avg_step_loss, "val_loss": val_loss})

                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        best_path = out_dir / "best_model.pth"
                        torch.save(model.state_dict(), best_path)
                        print(f"Saved new BEST model to {best_path}")

                    model.train()

                if optimizer_updates >= args.max_updates:
                    break

    # Save final model
    last_path = out_dir / "last_model.pth"
    torch.save(model.state_dict(), last_path)
    total_time_min = (time.time() - start_train_time) / 60.0
    print(f"Training completed in {total_time_min:.2f} minutes.")

    # Write training metadata and hashes
    best_hash = hashlib.sha256((out_dir / "best_model.pth").read_bytes()).hexdigest() if (out_dir / "best_model.pth").exists() else None
    last_hash = hashlib.sha256(last_path.read_bytes()).hexdigest()

    meta = {
        "status": "COMPLETED",
        "total_updates": optimizer_updates,
        "initial_loss": initial_loss,
        "best_val_loss": best_val_loss,
        "best_checkpoint_sha256": best_hash,
        "last_checkpoint_sha256": last_hash,
        "history": log_history,
        "device": str(device),
        "total_time_min": round(total_time_min, 2),
    }

    with open(out_dir / "training_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"Saved training metadata to {out_dir / 'training_meta.json'}")


if __name__ == "__main__":
    main()
