"""Shared dataset loading and geometry primitives for CNN training.

The repository ships no training data: load_corpus(data_dir) reads an external
dataset directory (see TRAINING.md). warp_perspective provides the perspective
correction used when cutting patches.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image

# The repository ships NO training data. load_corpus(data_dir) reads an
# external dataset directory (see TRAINING.md); every referenced path is
# resolved relative to that directory at load time.
CLASSES = (".", "B", "W")
CLASS_INDEX = {c: i for i, c in enumerate(CLASSES)}

# Cell size (pixels per grid cell) used when cutting patches. Must match
# PATCH in train_cnn.py and CELL_DEFAULT in inference/src/cnn_inference.mjs.
CELL_DEFAULT = 40


def load_corpus(data_dir):
    """Load an external dataset (see TRAINING.md) and return usable samples.

    `data_dir` must contain a ``dataset.json`` plus the image / truth files it
    references. The ``test`` split is never returned. All paths in the returned
    sample dicts are absolute, resolved under `data_dir`, so downstream
    functions (decode_image / truth_matrix) need no global configuration.
    """
    root = Path(data_dir).resolve()
    ds = json.loads((root / "dataset.json").read_text(encoding="utf-8"))
    usable = []
    for s in ds["samples"]:
        if s["split"] not in ("train", "val"):
            continue
        usable.append({
            "sample_id": s["sample_id"],
            "split": s["split"],
            "image_path": str(root / s["image"]),
            "truth_path": str(root / s["truth"]),
            "quad": np.asarray(s["quad"], dtype=float),
        })
    return usable


def truth_matrix(sample):
    lines = Path(sample["truth_path"]).read_text().splitlines()
    out = np.zeros((len(lines), len(lines)), dtype=np.int8)
    for r, line in enumerate(lines):
        for c, ch in enumerate(line):
            out[r, c] = CLASS_INDEX[ch]
    return out


def decode_image(sample):
    """RGB raster in [0,1]."""
    with Image.open(sample["image_path"]) as im:
        rgb = np.asarray(im.convert("RGB"), dtype=np.float64) / 255.0
    return rgb


def warp_perspective(rgb, quad, n_lines, cell):
    """Warp board region to square image. Delegates to warp module."""
    from .warp import warp_perspective as _warp
    return _warp(rgb, quad, n_lines, cell)
