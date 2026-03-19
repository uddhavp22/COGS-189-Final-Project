"""
Training script — EEGNet on MLSPred-Bench.

Usage
-----
    # Default: benchmark 12, cuda if available
    python train.py

    # Specific benchmark and options
    python train.py --benchmark 3 --epochs 50 --lr 5e-4 --device cpu

    # Override data directory
    python train.py --data_dir /path/to/data --benchmark 12
"""

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR
import wandb

from config import BenchmarkConfig, BENCHMARK_PARAMS
from dataset import build_dataloaders
from evaluate import compute_metrics, print_metrics
from models import EEGNet, Lamine_ViT


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_model(cfg: BenchmarkConfig, device: torch.device, model_name: str = "eegnet"):
    mc = cfg.model
    dc = cfg.data
    if model_name == "vit_subject":
        model = Lamine_ViT(
            n_chans=dc.n_chans,
            n_times=dc.n_times,
            n_outputs=2,
        ).to(device)
    else:
        model = EEGNet(
            n_chans=dc.n_chans,
            n_outputs=2,
            n_times=dc.n_times,
            F1=mc.F1,
            D=mc.D,
            F2=mc.F2,
            kernel_length=mc.kernel_length,
            depthwise_kernel_length=mc.depthwise_kernel_length,
            pool1_kernel_size=mc.pool1_kernel_size,
            pool2_kernel_size=mc.pool2_kernel_size,
            drop_prob=mc.drop_prob,
            pool_mode=mc.pool_mode,
            conv_spatial_max_norm=mc.conv_spatial_max_norm,
            final_conv_length=mc.final_conv_length,
            final_layer_with_constraint=mc.final_layer_with_constraint,
            norm_rate=mc.norm_rate,
        ).to(device)
    return model


# ---------------------------------------------------------------------------
# One epoch helpers
# ---------------------------------------------------------------------------

def train_epoch(model, loader, optimizer, criterion, device, scaler):
    model.train()
    total_loss = 0.0
    all_preds, all_labels, all_probs = [], [], []

    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)

        if scaler is not None:
            with torch.autocast(device_type="cuda"):
                logits = model(x)
                loss = criterion(logits, y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

        total_loss += loss.item() * x.size(0)
        probs = torch.softmax(logits.detach(), dim=1)[:, 1].cpu().numpy()
        preds = logits.detach().argmax(dim=1).cpu().numpy()
        all_preds.extend(preds.tolist())
        all_labels.extend(y.cpu().numpy().tolist())
        all_probs.extend(probs.tolist())

    avg_loss = total_loss / len(loader.dataset)
    metrics = compute_metrics(np.array(all_labels), np.array(all_preds), np.array(all_probs))
    return avg_loss, metrics


@torch.no_grad()
def eval_epoch(model, loader, criterion, device, return_arrays=False):
    model.eval()
    total_loss = 0.0
    all_preds, all_labels, all_probs = [], [], []

    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        logits = model(x)
        loss = criterion(logits, y)
        total_loss += loss.item() * x.size(0)
        probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
        preds = logits.argmax(dim=1).cpu().numpy()
        all_preds.extend(preds.tolist())
        all_labels.extend(y.cpu().numpy().tolist())
        all_probs.extend(probs.tolist())

    avg_loss = total_loss / len(loader.dataset)
    y_true = np.array(all_labels)
    y_pred = np.array(all_preds)
    y_prob = np.array(all_probs)
    metrics = compute_metrics(y_true, y_pred, y_prob)
    if return_arrays:
        return avg_loss, metrics, y_true, y_pred, y_prob
    return avg_loss, metrics


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(cfg: BenchmarkConfig, model_name: str = "eegnet"):
    set_seed(cfg.train.seed)

    device = torch.device(
        cfg.train.device if torch.cuda.is_available() else "cpu"
    )
    print(f"Device       : {device}")
    print(f"Benchmark    : {cfg.data.benchmark_id}  "
          f"(SPH={cfg.data.sph_min} min, SOP={cfg.data.sop_min} min)")
    print(f"Window       : {cfg.data.window_size_sec}s × {cfg.data.sfreq} Hz"
          f" = {cfg.data.n_times} samples")

    wandb.init(
        project="seizure-prediction",
        name=cfg.train.experiment_name,
        config={
            # data
            "benchmark_id":   cfg.data.benchmark_id,
            "sph_min":        cfg.data.sph_min,
            "sop_min":        cfg.data.sop_min,
            "n_chans":        cfg.data.n_chans,
            "sfreq":          cfg.data.sfreq,
            "window_size_sec":cfg.data.window_size_sec,
            # model
            "F1":             cfg.model.F1,
            "D":              cfg.model.D,
            "F2":             cfg.model.F2,
            "kernel_length":  cfg.model.kernel_length,
            "drop_prob":      cfg.model.drop_prob,
            # training
            "batch_size":     cfg.train.batch_size,
            "num_epochs":     cfg.train.num_epochs,
            "learning_rate":  cfg.train.learning_rate,
            "weight_decay":   cfg.train.weight_decay,
            "scheduler":      cfg.train.scheduler,
            "patience":       cfg.train.patience,
            "seed":           cfg.train.seed,
        },
    )

    results_dir = Path(cfg.train.results_dir)
    ckpt_dir    = Path(cfg.train.checkpoint_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    print("\nLoading data ...")
    train_loader, val_loader, test_loader = build_dataloaders(cfg)

    if train_loader is None:
        print("No training data found. Exiting.")
        sys.exit(1)

    # ------------------------------------------------------------------
    model = build_model(cfg, device, model_name)
    print(f"\n{model_name} — {model.num_params():,} trainable parameters")
    wandb.config.update({"n_params": model.num_params()})
    wandb.watch(model, log="gradients", log_freq=100)

    criterion = nn.CrossEntropyLoss()

    optimizer = Adam(
        model.parameters(),
        lr=cfg.train.learning_rate,
        weight_decay=cfg.train.weight_decay,
    )

    if cfg.train.scheduler == "cosine":
        scheduler = CosineAnnealingLR(optimizer, T_max=cfg.train.num_epochs)
    elif cfg.train.scheduler == "step":
        scheduler = StepLR(optimizer, step_size=30, gamma=0.1)
    else:
        scheduler = None

    scaler = torch.cuda.amp.GradScaler() if device.type == "cuda" else None

    # ------------------------------------------------------------------
    history = {"train_loss": [], "val_loss": [], "train_metrics": [], "val_metrics": []}
    best_val_auroc = -float("inf")
    patience_counter = 0
    best_ckpt = ckpt_dir / f"{cfg.train.experiment_name}_best.pt"

    print(f"\nTraining for up to {cfg.train.num_epochs} epochs ...\n")
    for epoch in range(1, cfg.train.num_epochs + 1):
        t0 = time.time()
        train_loss, train_m = train_epoch(model, train_loader, optimizer, criterion, device, scaler)
        val_loss,   val_m   = eval_epoch(model, val_loader, criterion, device) if val_loader else (0.0, {})

        if scheduler is not None:
            scheduler.step()

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_metrics"].append(train_m)
        history["val_metrics"].append(val_m)

        print(
            f"Epoch {epoch:3d}/{cfg.train.num_epochs}  "
            f"loss={train_loss:.4f}/{val_loss:.4f}  "
            f"acc={val_m.get('accuracy', 0):.4f}  "
            f"auc={val_m.get('auroc', 0):.4f}  "
            f"sen={val_m.get('sensitivity', 0):.4f}  "
            f"spe={val_m.get('specificity', 0):.4f}  "
            f"({time.time()-t0:.1f}s)"
        )

        wandb.log({
            "epoch":              epoch,
            "train/loss":         train_loss,
            "train/accuracy":     train_m.get("accuracy", 0),
            "train/sensitivity":  train_m.get("sensitivity", 0),
            "train/specificity":  train_m.get("specificity", 0),
            "train/auroc":        train_m.get("auroc", 0),
            "train/f1_preictal":  train_m.get("f1_preictal", 0),
            "train/fpr_per_hour": train_m.get("fpr_per_hour", 0),
            "val/loss":           val_loss,
            "val/accuracy":       val_m.get("accuracy", 0),
            "val/sensitivity":    val_m.get("sensitivity", 0),
            "val/specificity":    val_m.get("specificity", 0),
            "val/auroc":          val_m.get("auroc", 0),
            "val/f1_preictal":    val_m.get("f1_preictal", 0),
            "val/fpr_per_hour":   val_m.get("fpr_per_hour", 0),
            "lr":                 optimizer.param_groups[0]["lr"],
        }, step=epoch)

        val_auroc = val_m.get("auroc", 0.0)
        if val_auroc > best_val_auroc:
            best_val_auroc = val_auroc
            patience_counter = 0
            torch.save({"epoch": epoch, "state_dict": model.state_dict(),
                        "val_metrics": val_m}, best_ckpt)
        else:
            patience_counter += 1
            if patience_counter >= cfg.train.patience:
                print(f"\nEarly stopping at epoch {epoch}.")
                break

    # ------------------------------------------------------------------
    # Test evaluation
    ckpt = torch.load(best_ckpt, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["state_dict"])
    print(f"\nBest checkpoint: epoch {ckpt['epoch']}")

    if test_loader is not None:
        test_loss, test_metrics, y_true, y_pred, y_prob = eval_epoch(
            model, test_loader, criterion, device, return_arrays=True
        )
        print("\n--- Test Results ---")
        print_metrics(test_metrics)
    else:
        print("\nNo test split found — reporting best validation metrics.")
        # re-run val loader to get arrays for plots
        test_loss, test_metrics, y_true, y_pred, y_prob = eval_epoch(
            model, val_loader, criterion, device, return_arrays=True
        )
        test_loss    = best_val_loss
        test_metrics = ckpt["val_metrics"]
        print_metrics(test_metrics)

    # -- Confusion matrix
    wandb.log({
        "test/confusion_matrix": wandb.plot.confusion_matrix(
            probs=None,
            y_true=y_true.tolist(),
            preds=y_pred.tolist(),
            class_names=["interictal", "preictal"],
        )
    })

    # -- ROC curve
    wandb.log({
        "test/roc_curve": wandb.plot.roc_curve(
            y_true=y_true,
            y_probas=np.stack([1 - y_prob, y_prob], axis=1),
            labels=["interictal", "preictal"],
        )
    })

    # -- Precision-Recall curve
    wandb.log({
        "test/pr_curve": wandb.plot.pr_curve(
            y_true=y_true,
            y_probas=np.stack([1 - y_prob, y_prob], axis=1),
            labels=["interictal", "preictal"],
        )
    })

    wandb.log({
        "test/loss":         test_loss,
        "test/accuracy":     test_metrics.get("accuracy", 0),
        "test/sensitivity":  test_metrics.get("sensitivity", 0),
        "test/specificity":  test_metrics.get("specificity", 0),
        "test/auroc":        test_metrics.get("auroc", 0),
        "test/f1_preictal":  test_metrics.get("f1_preictal", 0),
        "test/fpr_per_hour": test_metrics.get("fpr_per_hour", 0),
        "best_epoch":        ckpt["epoch"],
    })
    wandb.finish()

    # ------------------------------------------------------------------
    results = {
        "experiment":    cfg.train.experiment_name,
        "benchmark_id":  cfg.data.benchmark_id,
        "sph_min":       cfg.data.sph_min,
        "sop_min":       cfg.data.sop_min,
        "history":       history,
        "test_loss":     test_loss,
        "test_metrics":  test_metrics,
        "best_val_auroc": best_val_auroc,
        "best_epoch":    ckpt["epoch"],
        "n_params":      model.num_params(),
    }
    out = results_dir / f"{cfg.train.experiment_name}_results.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved → {out}")

    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="EEGNet — MLSPred-Bench")
    p.add_argument("--benchmark",  type=int,   default=None, help="Benchmark ID 1-12")
    p.add_argument("--data_dir",   type=str,   default=None, help="Path to data/ folder")
    p.add_argument("--batch_size", type=int,   default=None)
    p.add_argument("--epochs",     type=int,   default=None)
    p.add_argument("--lr",         type=float, default=None)
    p.add_argument("--drop_prob",  type=float, default=None)
    p.add_argument("--device",     type=str,   default=None)
    p.add_argument("--name",       type=str,   default=None, help="Experiment name")
    p.add_argument("--model",      type=str,   default="eegnet", choices=["eegnet", "vit_subject"], help="Model architecture")
    p.add_argument("--no_wandb",   action="store_true",      help="Disable wandb logging")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    cfg = BenchmarkConfig()

    if args.no_wandb:
        os.environ["WANDB_MODE"] = "disabled"

    if args.benchmark  is not None: cfg.data.benchmark_id       = args.benchmark
    if args.data_dir   is not None: cfg.data.data_dir            = args.data_dir
    if args.batch_size is not None: cfg.train.batch_size         = args.batch_size
    if args.epochs     is not None: cfg.train.num_epochs         = args.epochs
    if args.lr         is not None: cfg.train.learning_rate      = args.lr
    if args.drop_prob  is not None: cfg.model.drop_prob          = args.drop_prob
    if args.device     is not None: cfg.train.device             = args.device
    if args.name       is not None: cfg.train.experiment_name    = args.name
    else:
        cfg.train.experiment_name = f"{args.model}_bmrk{cfg.data.benchmark_id:02d}"

    main(cfg, model_name=args.model)
