"""
PyTorch Dataset and DataLoader utilities for MLSPred-Bench.

Each benchmark stores pre-windowed, class-balanced EEG windows:
    {split}_values.hdf5  →  key "tracings", shape (N, 1280, 20)
    {split}_labels.csv   →  N rows, values 0 (interictal) or 1 (preictal)

EEGNet expects input (B, n_chans, n_times) = (B, 20, 1280), so we
transpose the HDF5's (N, 1280, 20) → (N, 20, 1280) on load.

Usage
-----
    from dataset import build_dataloaders
    from config import BenchmarkConfig

    cfg = BenchmarkConfig()
    train_dl, val_dl, test_dl = build_dataloaders(cfg)
"""

import os
from typing import Optional, Tuple

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class EEGDataset(Dataset):
    """
    Dataset for one split of a single MLSPred-Bench benchmark.

    Parameters
    ----------
    hdf5_path : str
        Path to `{split}_values.hdf5`.
    csv_path : str
        Path to `{split}_labels.csv`.
    normalize : bool
        If True, z-score each window per channel. Default True.
    """

    def __init__(self, hdf5_path: str, csv_path: str, normalize: bool = True):
        with h5py.File(hdf5_path, "r") as f:
            # (N, 1280, 20) → transpose → (N, 20, 1280)
            self.X = torch.from_numpy(
                f["tracings"][:].transpose(0, 2, 1).astype(np.float32)
            )

        self.y = torch.from_numpy(
            np.loadtxt(csv_path, delimiter=",", dtype=np.int64)
        )

        assert len(self.X) == len(self.y), (
            f"Sample count mismatch: X={len(self.X)}, y={len(self.y)}"
        )

        if normalize:
            # Per-channel z-score across the time dimension for every window
            mean = self.X.mean(dim=-1, keepdim=True)      # (N, 20, 1)
            std  = self.X.std(dim=-1, keepdim=True) + 1e-8
            self.X = (self.X - mean) / std

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.y[idx]

    @property
    def class_counts(self) -> Tuple[int, int]:
        n0 = int((self.y == 0).sum())
        n1 = int((self.y == 1).sum())
        return n0, n1


# ---------------------------------------------------------------------------

def _find_split(bmrk_dir: str, split: str) -> Optional[Tuple[str, str]]:
    """
    Return (hdf5_path, csv_path) for a split if both files exist, else None.
    Handles both 'tests' and 'test' filename variants.
    """
    variants = [split, "tests"] if split == "test" else [split]
    for name in variants:
        h5 = os.path.join(bmrk_dir, f"{name}_values.hdf5")
        cs = os.path.join(bmrk_dir, f"{name}_labels.csv")
        if os.path.exists(h5) and os.path.exists(cs):
            return h5, cs
    return None


def build_dataloaders(
    cfg,
    normalize: bool = True,
) -> Tuple[Optional[DataLoader], Optional[DataLoader], Optional[DataLoader]]:
    """
    Build train / val / test DataLoaders from a BenchmarkConfig.

    Returns None for any split whose files are not present on disk.

    Parameters
    ----------
    cfg : BenchmarkConfig
    normalize : bool
        Whether to z-score normalise each window. Default True.

    Returns
    -------
    (train_loader, val_loader, test_loader)
        Any absent split is returned as None.
    """
    bmrk_dir = cfg.data.bmrk_dir
    tc = cfg.train

    loaders = []
    for split in ("train", "valid", "test"):
        paths = _find_split(bmrk_dir, split)
        if paths is None:
            print(f"  [{split}] not found in {bmrk_dir} — skipping")
            loaders.append(None)
            continue

        ds = EEGDataset(*paths, normalize=normalize)
        n0, n1 = ds.class_counts
        is_train = (split == "train")
        print(
            f"  [{split}] {len(ds):>6,} windows  "
            f"(interictal={n0:,}, preictal={n1:,})"
        )
        loaders.append(
            DataLoader(
                ds,
                batch_size=tc.batch_size,
                shuffle=is_train,
                num_workers=tc.num_workers,
                pin_memory=tc.pin_memory,
                drop_last=is_train,
            )
        )

    return tuple(loaders)
