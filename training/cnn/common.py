"""Dataset loading and geometry primitives for CNN training.

The training corpus is the *unmodified* gomrade dataset (Kaggle, CC BY-NC-ND
4.0). This repository ships no data: --data points at a local gomrade
extraction containing ``dataset/`` and ``dataset2/``. load_corpus() reads the
original files directly -- there is no dataset.json, no pre-generated manifest
of samples, and no re-encoded copies. See TRAINING.md for provenance, the
exact frame counts, and the license terms.

Only in-memory transformations are applied: truth text is whitespace-stripped
and the four board corners are parsed from the YAML sidecar. No derivative of
the dataset is written to disk or distributed by this repository.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
from PIL import Image

CLASSES = (".", "B", "W")
CLASS_INDEX = {c: i for i, c in enumerate(CLASSES)}

# Cell size (pixels per grid cell) used when cutting patches. Must match
# PATCH in train_cnn.py and CELL_DEFAULT in inference/src/cnn_inference.mjs.
CELL_DEFAULT = 40

# Unit -> train/val/test/exclude. A whole unit (one game directory) always
# lands in exactly one split. Derived from seed 20261105. The raw gomrade
# files carry no split information, so this ~3 KB manifest is the only dataset
# metadata shipped by this repository; it contains no images, truths or quads.
SPLIT_FILE = Path(__file__).with_name("split.tsv")

_TOP_KEY = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*:\s*$")
_INT_LINE = re.compile(r"^\s*-\s*(?:-\s*)?(-?\d+)\s*$")
_IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".PNG", ".JPG", ".JPEG")
_SUBDIRS = (("g1", "dataset"), ("g2", "dataset2"))


def _read_split():
    """Read the unit -> role table shipped with the repository."""
    roles = {}
    for line in SPLIT_FILE.read_text(encoding="utf-8").splitlines():
        if not line or line.startswith("#"):
            continue
        unit, role, _why = line.split("\t", 2)
        roles[unit] = role
    return roles


def _parse_quad(yml_path):
    """Extract pts_clicks (four [x,y] board corners) from a gomrade YAML file.

    gomrade ships these as a tiny, fixed-shape block; parse just that block with
    the stdlib so training needs no PyYAML. Returns None on any mismatch. The
    coordinates are already in the original image's pixel space (all training
    units have EXIF orientation 1 / none, so no rotation is applied).
    """
    if not yml_path.is_file():
        return None
    lines = yml_path.read_text(encoding="utf-8").replace("\r\n", "\n").split("\n")
    for i, line in enumerate(lines):
        if line.strip() != "pts_clicks:":
            continue
        nums = []
        for j in range(i + 1, len(lines)):
            if _TOP_KEY.match(lines[j]):
                break
            m = _INT_LINE.match(lines[j])
            if m:
                nums.append(int(m.group(1)))
            elif lines[j].strip():
                break
        if len(nums) != 8:
            return None
        n = nums
        return [[n[0], n[1]], [n[2], n[3]], [n[4], n[5]], [n[6], n[7]]]
    return None


def _paired_frames(unit_dir):
    """(image, truth) pairs: same basename, one image + one .txt truth file.

    Files without a sibling image (e.g. colors.txt / pts.txt coordinate dumps,
    .old snapshots) are skipped automatically.
    """
    pairs = []
    for truth in sorted(unit_dir.iterdir(), key=lambda p: p.name):
        if truth.suffix.lower() != ".txt":
            continue
        stem = truth.with_suffix("")
        image = next(
            (Path(str(stem) + sfx) for sfx in _IMAGE_SUFFIXES
             if Path(str(stem) + sfx).is_file()),
            None,
        )
        if image is not None:
            pairs.append((image, truth))
    return pairs


def load_corpus(data_dir):
    """Enumerate a gomrade extraction directly; return train/val samples.

    Only units marked train/val in split.tsv AND carrying four board corners
    are used. Sealed test units, excluded subdirectories and units without a
    parsable yml are skipped. Enumeration is sorted, so patch order -- and
    therefore the trained weights -- is deterministic across runs.

    Returned paths are absolute so decode_image()/truth_matrix() need no global
    configuration.
    """
    root = Path(data_dir).resolve()
    roles = _read_split()
    usable = []
    for prefix, subdir in _SUBDIRS:
        base = root / subdir
        if not base.is_dir():
            raise SystemExit(f"missing gomrade subdirectory: {base}")
        for unit_dir in sorted((p for p in base.iterdir() if p.is_dir()),
                               key=lambda p: p.name):
            unit = f"{prefix}:{unit_dir.name}"
            role = roles.get(unit, "exclude")
            if role not in ("train", "val"):
                continue
            quad = _parse_quad(unit_dir / "board_extractor_state.yml")
            if quad is None:
                continue
            for image, truth in _paired_frames(unit_dir):
                usable.append({
                    "sample_id": f"{unit}/{image.stem}",
                    "split": role,
                    "image_path": str(image),
                    "truth_path": str(truth),
                    "quad": np.asarray(quad, dtype=float),
                })
    return usable


def truth_matrix(sample):
    """Read a gomrade truth .txt into an int8 [N,N] matrix.

    gomrade stores 19 rows of 19 tokens separated by spaces ('.' 'B' 'W').
    Whitespace is dropped in memory; the file on disk is never rewritten.
    """
    lines = [
        "".join(line.split())
        for line in Path(sample["truth_path"]).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    n = len(lines)
    if not n or any(len(line) != n for line in lines):
        raise ValueError(f"non-square truth matrix: {sample['truth_path']}")
    out = np.zeros((n, n), dtype=np.int8)
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
