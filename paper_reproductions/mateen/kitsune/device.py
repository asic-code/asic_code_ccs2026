"""Apple-Silicon-aware torch device selector for the official Mateen code.

The official code defines `device = "cuda" if torch.cuda.is_available()
else "cpu"` at module load time in AE.py / utils.py / main.py /
selection_utils.py / merge_utils.py. We import each of those modules
once, then overwrite `device` on each so subsequent calls land on MPS
(or whatever DEVICE resolves to).
"""
from __future__ import annotations
import sys
import os
import torch


def best_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


DEVICE = best_device()
DEVICE_STR = str(DEVICE)


def _ensure_official_path() -> None:
    p = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "..",
        "ref-src",
        "Mateen",
        "MateenUtils",
    )
    p = os.path.abspath(p)
    if p not in sys.path:
        sys.path.insert(0, p)


def patch_official_device() -> None:
    """Import the five official modules and replace their `device`
    attribute with our DEVICE so model.to(device) lands on MPS."""
    _ensure_official_path()
    import AE  # noqa: F401
    import utils  # noqa: F401
    import data_processing  # noqa: F401
    import selection_utils  # noqa: F401
    import merge_utils  # noqa: F401
    import main as mateen_main  # noqa: F401
    for mod in (AE, utils, selection_utils, merge_utils, mateen_main):
        mod.device = DEVICE_STR  # match the original "cuda"/"cpu" string form
