#!/usr/bin/env python3
"""CNN classifier for Go board cell recognition.

Resource budget (target machine: 2 CPU, 7.5G RAM, no GPU):
  - all train patches as uint8: 478325 × 784 = 375 MB
  - all val patches as uint8:   106495 × 784 =  83 MB
  - total RAM peak: ~3 GB (patches + torch + overhead)
  - precompute time: ~210s train + 50s val = ~5 min
  - training time: ~30s (5 epoch, batch=512, LeNet-scale, CPU)
  - total: ~6 min

Memory strategy: store patches as uint8 [N, 40, 40], convert to float32
on-the-fly per batch. Never materialise float32 for all patches.

Determinism / replay
--------------------
This run is byte-reproducible: a fixed seed, fixed thread count, deterministic
file enumeration and a constant ONNX export dummy mean re-running with the
same gomrade extraction and the pinned dependencies produces a bit-identical
``cnn.onnx``. See TRAINING.md §5 for the exact verification procedure.

Usage:
  python3 -m training.cnn.train_cnn --data <gomrade 解压目录> \
      [--epochs N] [--batch B] [--out DIR]

Outputs (see TRAINING.md):
  <out>/cnn.onnx       — ONNX model for browser inference (checked into the repo)
  <out>/cnn.pt          — PyTorch checkpoint (scratch, ignored)
  <out>/cnn_meta.json  — metadata (scratch, ignored)
  <out>/cnn_eval.json  — eval results (scratch, ignored)
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np

from .common import (
    load_corpus, truth_matrix, decode_image,
    warp_perspective, CELL_DEFAULT,
)

PATCH = 40
HALF = PATCH // 2
SEED = 20261105


def precompute_patches(samples, cell, label):
    """Warp + cut patches for all samples. Returns uint8 [N,40,40], int64 [N]."""
    n = len(samples)
    lines = truth_matrix(samples[0]).shape[0]
    for s in samples[1:]:
        if truth_matrix(s).shape[0] != lines:
            raise ValueError("all frames in one run must share the same board size")
    ppf = lines * lines
    total = n * ppf
    X = np.empty((total, PATCH, PATCH), dtype=np.uint8)
    y = np.empty(total, dtype=np.int64)
    t0 = time.time()
    for i, s in enumerate(samples):
        rgb = decode_image(s)
        tm = truth_matrix(s)
        warped = warp_perspective(rgb, s["quad"], lines, cell)
        V = warped["V"]
        Wd = warped["Wd"]
        margin = cell
        for row in range(lines):
            for col in range(lines):
                idx = i * ppf + row * lines + col
                cx = margin + col * cell
                cy = margin + row * cell
                x0 = int(cx) - HALF
                y0 = int(cy) - HALF
                sy0 = max(0, y0); sy1 = min(Wd, y0 + PATCH)
                sx0 = max(0, x0); sx1 = min(Wd, x0 + PATCH)
                X[idx, sy0 - y0:sy1 - y0, sx0 - x0:sx1 - x0] = V[sy0:sy1, sx0:sx1]
                y[idx] = tm[row, col]
        if (i + 1) % 200 == 0:
            el = int(time.time() - t0)
            print(f"  {label}: {i+1}/{n} ({el}s, {el*total/((i+1)*ppf):.0f}s est)", flush=True)
    el = int(time.time() - t0)
    print(f"  {label}: done {n} frames, {total} patches, {el}s", flush=True)
    return X, y


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, required=True,
                    help="gomrade extraction root (contains dataset/ and dataset2/)")
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--out", type=str, default="training/out-cnn")
    ap.add_argument("--cell", type=int, default=CELL_DEFAULT)
    ap.add_argument("--seed", type=int, default=SEED,
                    help="training seed; changing it changes the model bytes")
    ap.add_argument("--threads", type=int, default=2,
                    help="torch intra-op threads; pinned for determinism")
    args = ap.parse_args()

    # Determinism must be established before torch creates its RNG or thread
    # pools. Do not move these below any tensor allocation.
    os.environ["PYTHONHASHSEED"] = str(args.seed)
    np.random.seed(args.seed)
    import torch
    import torch.nn as nn
    torch.manual_seed(args.seed)
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=== CNN training (reproducible) ===", flush=True)
    print(f"torch={torch.__version__}, numpy={np.__version__}, "
          f"cell={args.cell}, patch={PATCH}, seed={args.seed}, "
          f"threads={args.threads}", flush=True)
    print(f"data={Path(args.data).resolve()}", flush=True)

    samples = load_corpus(args.data)
    tr = [s for s in samples if s["split"] == "train"]
    va = [s for s in samples if s["split"] == "val"]
    print(f"train={len(tr)} frames, val={len(va)} frames", flush=True)
    if not tr or not va:
        raise SystemExit("load_corpus returned no train/val frames")

    # Fingerprint the exact input ordering so a replay can be audited even if
    # the dataset is later rearranged. Hash of sample ids, not file contents.
    order = "\n".join(s["sample_id"] for s in samples).encode()
    input_fingerprint = hashlib.sha256(order).hexdigest()
    print(f"sample order sha256={input_fingerprint}", flush=True)

    # --- Precompute ---
    print("Precomputing train patches...", flush=True)
    Xtr, ytr = precompute_patches(tr, args.cell, "train")
    print(f"  Xtr={Xtr.shape} {Xtr.dtype} {Xtr.nbytes/1e6:.0f}MB", flush=True)
    gc.collect()

    print("Precomputing val patches...", flush=True)
    Xva, yva = precompute_patches(va, args.cell, "val")
    print(f"  Xva={Xva.shape} {Xva.dtype} {Xva.nbytes/1e6:.0f}MB", flush=True)
    gc.collect()

    # --- Model ---
    model = nn.Sequential(
        nn.Conv2d(1, 16, 3), nn.ReLU(), nn.MaxPool2d(2),  # 40→38→19
        nn.Conv2d(16, 32, 3), nn.ReLU(), nn.MaxPool2d(2), # 19→17→8
        nn.Flatten(),
        nn.Linear(32 * 8 * 8, 64), nn.ReLU(),
        nn.Linear(64, 3),
    )
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.CrossEntropyLoss()

    # --- Train ---
    n = len(Xtr)
    print(f"Training: {n} patches, {args.epochs} epochs, batch={args.batch}", flush=True)
    vacc = 0.0
    for ep in range(args.epochs):
        model.train()
        perm = torch.randperm(n)
        t0 = time.time()
        total_loss = 0.0
        correct = 0
        for start in range(0, n, args.batch):
            idx = perm[start:start + args.batch]
            xb = torch.from_numpy(Xtr[idx]).float().unsqueeze(1) / 255.0
            yb = torch.from_numpy(ytr[idx])
            opt.zero_grad()
            out = model(xb)
            loss = loss_fn(out, yb)
            loss.backward()
            opt.step()
            total_loss += loss.item() * len(idx)
            correct += (out.argmax(1) == yb).sum().item()
        train_acc = correct / n

        model.eval()
        vcorrect = 0
        with torch.no_grad():
            for start in range(0, len(Xva), args.batch):
                xb = torch.from_numpy(Xva[start:start+args.batch]).float().unsqueeze(1) / 255.0
                yb = torch.from_numpy(yva[start:start+args.batch])
                out = model(xb)
                vcorrect += (out.argmax(1) == yb).sum().item()
        vacc = vcorrect / len(Xva)

        print(f"  ep {ep+1}/{args.epochs}: loss={total_loss/n:.4f} "
              f"train_acc={train_acc:.4f} val_acc={vacc:.4f} ({time.time()-t0:.0f}s)", flush=True)

    # --- Confusion matrix ---
    model.eval()
    preds = np.empty(len(Xva), dtype=np.int64)
    with torch.no_grad():
        for start in range(0, len(Xva), args.batch):
            xb = torch.from_numpy(Xva[start:start+args.batch]).float().unsqueeze(1) / 255.0
            out = model(xb)
            preds[start:start+args.batch] = out.argmax(1).numpy()

    cm = np.zeros((3, 3), dtype=int)
    for t, p in zip(yva, preds):
        cm[t, p] += 1
    wrong = int(cm.sum() - np.trace(cm))
    labels = ["empty", "black", "white"]

    print(f"\n=== Result ===", flush=True)
    print(f"val_acc={vacc:.4f} wrong={wrong}", flush=True)
    print(f"          pred_empty pred_black pred_white", flush=True)
    for i, name in enumerate(labels):
        print(f"  true_{name:5s}  {cm[i,0]:8d}   {cm[i,1]:8d}   {cm[i,2]:8d}", flush=True)

    # --- Save ONNX + PT ---
    pt_path = out_dir / "cnn.pt"
    onnx_path = out_dir / "cnn.onnx"
    torch.save(model.state_dict(), str(pt_path))

    # A constant dummy: torch.randn would consume RNG state and make the ONNX
    # export itself non-reproducible across runs with different call order.
    dummy = torch.zeros(1, 1, PATCH, PATCH)
    torch.onnx.export(model, dummy, str(onnx_path),
                      input_names=["patch"], output_names=["logits"],
                      dynamic_axes={"patch": {0: "batch"}, "logits": {0: "batch"}},
                      opset_version=17, dynamo=False)

    onnx_sha = hashlib.sha256(onnx_path.read_bytes()).hexdigest()
    print(f"cnn.onnx sha256={onnx_sha}", flush=True)

    # --- Meta + eval JSON ---
    meta = {
        "method": "cnn",
        "patch_size": PATCH, "cell": args.cell, "channels": 1,
        "architecture": "Conv(1,16,3)-ReLU-MaxPool-Conv(16,32,3)-ReLU-MaxPool-FC(800,64)-FC(64,3)",
        "epochs": args.epochs, "batch_size": args.batch, "lr": args.lr,
        "seed": args.seed, "torch_threads": args.threads,
        "train_frames": len(tr), "val_frames": len(va),
        "sample_order_sha256": input_fingerprint,
        "cnn_onnx_sha256": onnx_sha,
        "val_accuracy": float(vacc),
        "n_params": sum(p.numel() for p in model.parameters()),
    }
    (out_dir / "cnn_meta.json").write_text(json.dumps(meta, indent=2))

    eval_report = {
        "val_accuracy": float(vacc), "wrong": wrong,
        "confusion_matrix": cm.tolist(), "labels": labels,
        "per_class": {
            labels[i]: {
                "precision": cm[i,i] / max(cm[:,i].sum(), 1),
                "recall": cm[i,i] / max(cm[i,:].sum(), 1),
                "support": int(cm[i,:].sum()),
            } for i in range(3)
        },
    }
    (out_dir / "cnn_eval.json").write_text(json.dumps(eval_report, indent=2))

    print(f"\nSaved: {onnx_path}, {pt_path}", flush=True)
    print("Done.", flush=True)


if __name__ == "__main__":
    main()
