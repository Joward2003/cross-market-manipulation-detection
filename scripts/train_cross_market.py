#!/usr/bin/env python3
"""Train one label-free TS2Vec / optional-DANN representation on CUDA.

Cases, case dates, controls and any enforcement label are deliberately absent
from this command.  Its only inputs are the unlabeled sequence store and the
pre-declared calendar split.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.modeling.network import TS2VecDANN, augment_window, coral_loss, dann_coefficient, hierarchical_ssl_loss
from src.modeling.store import BalancedDomainBatchSampler, SequenceStore, WindowDataset, fit_market_scalers


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/model_training_memory_efficient.yaml")
    parser.add_argument("--dataset", required=True, help="Key under datasets in the YAML configuration")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--run-dir", type=Path, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--allow-cpu", action="store_true", help="Diagnostic only; production runs require CUDA.")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_loader(dataset: WindowDataset, *, batch_size: int, sampler=None, workers: int = 0, shuffle: bool = False) -> DataLoader:
    kwargs = {"num_workers": workers, "pin_memory": torch.cuda.is_available()}
    if workers:
        kwargs.update({"persistent_workers": True, "prefetch_factor": 2})
    if sampler is not None:
        return DataLoader(dataset, batch_sampler=sampler, **kwargs)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=False, **kwargs)


@torch.no_grad()
def validation(model: nn.Module, loader: DataLoader, device: torch.device, max_steps: int) -> dict[str, float]:
    model.eval()
    losses, embeddings, domains, logits_all = [], [], [], []
    for step, (x, domain) in enumerate(loader):
        if step >= max_steps:
            break
        x, domain = x.to(device, non_blocking=True), domain.to(device, non_blocking=True)
        h1, z1, logits = model(x, 0.0)
        h2, z2, _ = model(x, 0.0)
        losses.append(float(hierarchical_ssl_loss(h1, z1, h2, z2).item()))
        embeddings.append(z1.cpu())
        domains.append(domain.cpu())
        logits_all.append(logits.cpu())
    z = torch.cat(embeddings) if embeddings else torch.zeros((1, 1))
    per_dim_std = float(z.std(dim=0).mean())
    covariance = torch.cov(z.T) if z.shape[0] > 1 else torch.eye(z.shape[1])
    eigenvalues = torch.linalg.eigvalsh(covariance).clamp_min(1e-12)
    p = eigenvalues / eigenvalues.sum()
    effective_rank = float(torch.exp(-(p * p.log()).sum()))
    domain = torch.cat(domains) if domains else torch.zeros(1, dtype=torch.long)
    logits = torch.cat(logits_all) if logits_all else torch.zeros((1, 1))
    return {"validation_ts2vec_loss": float(np.mean(losses)), "embedding_dimension_std": per_dim_std,
            "effective_rank": effective_rank, "domain_head_accuracy": float((logits.argmax(1) == domain).float().mean())}


def save_checkpoint(path: Path, *, model: nn.Module, optimizer, scheduler, epoch: int, best_metric: float,
                    scalers: dict, config: dict, dataset_name: str, seed: int) -> None:
    torch.save({
        "model": model.state_dict(), "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
        "epoch": epoch, "best_metric": best_metric, "scalers": scalers, "config": config,
        "dataset": dataset_name, "seed": seed,
    }, path)


def main() -> None:
    args = parse_args()
    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    if args.dataset not in cfg["datasets"]:
        raise KeyError(f"Unknown dataset {args.dataset!r}; choose one of {list(cfg['datasets'])}")
    if not torch.cuda.is_available() and not args.allow_cpu:
        raise RuntimeError("CUDA is unavailable. Refusing CPU training; run scripts/verify_cuda.py first.")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = bool(cfg["optimization"]["allow_tf32"])
        torch.backends.cudnn.allow_tf32 = bool(cfg["optimization"]["allow_tf32"])
        torch.set_float32_matmul_precision("high")
    set_seed(args.seed)
    scope = cfg["datasets"][args.dataset]
    markets = list(scope["markets"])
    run_dir = args.run_dir or ROOT / "runs" / args.dataset / f"seed_{args.seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "resolved_config.yaml").write_text(yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False), encoding="utf-8")
    store = SequenceStore(ROOT / cfg["data"]["sequence_store"])
    split = cfg["split"]["primary"]
    train_endpoints = store.select_window_ends("train", markets, split["train_years"])
    validation_parts = [
        store.select_window_ends("train", [market], split["validation_years"],
            max_items=int(cfg["sampler"]["validation_windows_per_market"]), seed=args.seed + i)
        for i, market in enumerate(markets)
    ]
    validation_endpoints = np.concatenate(validation_parts)
    scalers = fit_market_scalers(
        store, train_endpoints, markets, sample_limit=int(cfg["anomaly_heads"]["fit_sample_size"]), seed=args.seed,
        mad_constant=float(cfg["scaling"]["mad_consistency_constant"]), epsilon=float(cfg["scaling"]["epsilon"]),
    )
    (run_dir / "scalers.json").write_text(json.dumps(scalers, indent=2), encoding="utf-8")
    clip = tuple(map(float, cfg["scaling"]["clip"]))
    train_dataset = WindowDataset(store, train_endpoints, scalers, clip=clip)
    val_dataset = WindowDataset(store, validation_endpoints, scalers, clip=clip)
    optim_cfg, sampler_cfg, loader_cfg = cfg["optimization"], cfg["sampler"], cfg["loader"]
    batch_size = int(optim_cfg["batch_size"])
    sampler = BalancedDomainBatchSampler(train_dataset, scope["market_mix"], batch_size, int(sampler_cfg["steps_per_epoch"]), args.seed)
    train_loader = make_loader(train_dataset, batch_size=batch_size, sampler=sampler, workers=int(loader_cfg["workers"]))
    val_loader = make_loader(val_dataset, batch_size=batch_size, workers=int(loader_cfg["workers"]))
    enc = cfg["encoder"]
    model = TS2VecDANN(enc["input_dim"], enc["hidden_dim"], enc["output_dim"], enc["dilations"], enc["dropout"], len(store.market_to_domain)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(optim_cfg["learning_rate"]), weight_decay=float(optim_cfg["weight_decay"]))
    total_steps = int(optim_cfg["epochs"]) * len(train_loader)
    warmup_steps = max(1, int(total_steps * float(optim_cfg["warmup_fraction"])))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda s: min((s + 1) / warmup_steps, 1.0) * 0.5 * (1 + math.cos(math.pi * max(s - warmup_steps, 0) / max(total_steps - warmup_steps, 1))))
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda" and optim_cfg["amp"] == "fp16"))
    first_epoch, best_metric, stale_epochs = 0, float("inf"), 0
    checkpoint_path = run_dir / "last.pt"
    if args.resume and checkpoint_path.exists():
        saved = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(saved["model"]); optimizer.load_state_dict(saved["optimizer"]); scheduler.load_state_dict(saved["scheduler"])
        first_epoch, best_metric = int(saved["epoch"]) + 1, float(saved["best_metric"])
    metric_log = (run_dir / "metrics.jsonl").open("a", encoding="utf-8")
    global_step = first_epoch * len(train_loader)
    aug = enc["augmentation"]
    for epoch in range(first_epoch, int(optim_cfg["epochs"])):
        model.train()
        running_ssl, running_domain, running_coral, running_count = 0.0, 0.0, 0.0, 0
        progress_bar = tqdm(train_loader, desc=f"{args.dataset} seed={args.seed} epoch={epoch + 1}", unit="batch")
        for x, domain in progress_bar:
            x, domain = x.to(device, non_blocking=True), domain.to(device, non_blocking=True)
            progress = global_step / max(total_steps - 1, 1)
            coefficient = dann_coefficient(progress, cfg["objective"]["dann"]["ssl_only_warmup_fraction"], cfg["objective"]["dann"]["ramp_end_fraction"], cfg["objective"]["dann"]["loss_weight_max"]) if scope["alignment"] == "dann" else 0.0
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", dtype=torch.float16, enabled=(device.type == "cuda" and optim_cfg["amp"] == "fp16")):
                x1 = augment_window(x, aug["point_mask_probability"], aug["contiguous_mask_probability"])
                x2 = augment_window(x, aug["point_mask_probability"], aug["contiguous_mask_probability"])
                h1, z1, logits = model(x1, coefficient)
                h2, z2, _ = model(x2, coefficient)
                ssl = hierarchical_ssl_loss(h1, z1, h2, z2)
                domain_loss = F.cross_entropy(logits, domain) if coefficient else torch.zeros((), device=device)
                coral = coral_loss(z1, domain, target_domain=store.market_to_domain["cn"]) if scope["alignment"] == "coral" else torch.zeros((), device=device)
                # `coefficient` is already applied by the gradient-reversal
                # layer. Multiplying CE again would weaken the encoder's
                # domain-adversarial gradient by lambda^2.
                loss = ssl + domain_loss + float(cfg["objective"]["coral"]["loss_weight"]) * coral
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(optim_cfg["gradient_clip_norm"]))
            scaler.step(optimizer); scaler.update(); scheduler.step()
            running_ssl += float(ssl.detach()); running_domain += float(domain_loss.detach()); running_coral += float(coral.detach()); running_count += 1
            global_step += 1
            progress_bar.set_postfix(ssl=f"{running_ssl / running_count:.3f}", grl=f"{coefficient:.3f}")
        report = validation(model, val_loader, device, int(sampler_cfg["validation_steps"]))
        report.update({"epoch": epoch + 1, "train_ssl_loss": running_ssl / running_count, "train_domain_loss": running_domain / running_count, "train_coral_loss": running_coral / running_count,
                       "learning_rate": optimizer.param_groups[0]["lr"], "dataset": args.dataset, "seed": args.seed})
        metric_log.write(json.dumps(report) + "\n"); metric_log.flush()
        if report["embedding_dimension_std"] < float(cfg["representation_quality"]["collapse_floor_dimension_std"]):
            raise RuntimeError(f"Representation collapse guard triggered: {report['embedding_dimension_std']:.6f}")
        save_checkpoint(run_dir / "last.pt", model=model, optimizer=optimizer, scheduler=scheduler, epoch=epoch, best_metric=best_metric, scalers=scalers, config=cfg, dataset_name=args.dataset, seed=args.seed)
        if report["validation_ts2vec_loss"] < best_metric - float(optim_cfg["early_stopping"]["min_delta"]):
            best_metric, stale_epochs = report["validation_ts2vec_loss"], 0
            save_checkpoint(run_dir / "best.pt", model=model, optimizer=optimizer, scheduler=scheduler, epoch=epoch, best_metric=best_metric, scalers=scalers, config=cfg, dataset_name=args.dataset, seed=args.seed)
        else:
            stale_epochs += 1
            if stale_epochs >= int(optim_cfg["early_stopping"]["patience"]):
                print(f"Early stopping after epoch {epoch + 1}.", flush=True)
                break
    metric_log.close()
    print(f"Done. Best checkpoint: {run_dir / 'best.pt'}", flush=True)


if __name__ == "__main__":
    main()
