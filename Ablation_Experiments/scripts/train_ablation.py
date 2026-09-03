"""Unified target-domain trainer for the eight Trans_GAN ablation settings.

This script intentionally lives outside Swin-Unet-transLearning/ so the original
training implementation is not modified. It reuses the repository's dataset,
model, and DiceLoss implementations and exposes only the ablated components:
DT, CG-PT, PU, SDB, and HPL.

G1: scratch + all parameters trainable
G2: original source checkpoint + all parameters trainable (DT)
G3: CycleGAN-pretrained checkpoint + all parameters trainable
G4: CG-PT + progressive unfreezing
G5: CG-PT + progressive unfreezing + SDB
G6: CG-PT + progressive unfreezing + HPL
G7: scratch + progressive unfreezing + SDB + HPL
G8: CG-PT + progressive unfreezing + SDB + HPL
"""
import argparse
import csv
import json
import logging
import os
import random
import sys
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from tqdm import tqdm

HERE = os.path.dirname(os.path.abspath(__file__))
ABLATION_ROOT = os.path.dirname(HERE)
REPO_ROOT = os.path.dirname(ABLATION_ROOT)
SWIN_ROOT = os.path.join(REPO_ROOT, "Swin-Unet-transLearning")
if SWIN_ROOT not in sys.path:
    sys.path.insert(0, SWIN_ROOT)

from config import get_config  # noqa: E402
from datasets.dataset_transfer import OrchardTransferDataset, TransferGenerator  # noqa: E402
from networks.vision_transformer import SwinUnet  # noqa: E402
from utils import DiceLoss  # noqa: E402


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def unwrap(model):
    return model.module if hasattr(model, "module") else model


def load_checkpoint(model, path):
    if not path:
        return
    ckpt = torch.load(path, map_location="cpu")
    state = ckpt.get("model", ckpt.get("state_dict", ckpt)) if isinstance(ckpt, dict) else ckpt
    state = {k.replace("module.", "", 1): v for k, v in state.items()}
    target = model.swin_unet.state_dict()
    loadable = {}
    for key, value in state.items():
        key2 = key
        if key2.startswith("swin_unet.swin_unet."):
            key2 = key2[len("swin_unet.swin_unet."):]
        elif key2.startswith("swin_unet."):
            key2 = key2[len("swin_unet."):]
        if key2 in target and target[key2].shape == value.shape:
            loadable[key2] = value
    msg = model.swin_unet.load_state_dict(loadable, strict=False)
    logging.info("Loaded %d tensors from %s; missing=%d unexpected=%d", len(loadable), path,
                 len(msg.missing_keys), len(msg.unexpected_keys))


def set_trainable(model, progressive, phase=None):
    m = unwrap(model)
    if not progressive:
        for p in m.parameters():
            p.requires_grad = True
        return

    net = m.swin_unet
    for p in m.parameters():
        p.requires_grad = False
    decoder = [net.layers_up, net.norm, net.norm_up, net.up, net.output]
    if phase == "stage1_decoder":
        modules = decoder
    elif phase == "stage2_deep":
        modules = decoder + [net.layers[3]]
    elif phase == "stage3_mid":
        modules = decoder + [net.layers[3], net.layers[2]]
    else:
        raise ValueError(f"Unknown progressive phase: {phase}")
    for module in modules:
        if module is not None:
            for p in module.parameters():
                p.requires_grad = True


def build_optimizer(model, lr, weight_decay):
    params = [p for p in model.parameters() if p.requires_grad]
    if not params:
        raise RuntimeError("No trainable parameters for this ablation setting.")
    return torch.optim.AdamW(params, lr=lr, betas=(0.9, 0.999), weight_decay=weight_decay)


def semantic_loss(logits, labels, ce_loss, dice_loss):
    return 0.5 * ce_loss(logits, labels) + 0.5 * dice_loss(logits, labels, softmax=True)


def hpl_loss(features):
    if not features or len(features) < 2:
        return features[0].new_tensor(0.0) if features else None
    maps = []
    for f in features:
        if f.ndim != 4:
            continue
        x = torch.mean(torch.abs(f), dim=1, keepdim=True)
        x = x / (x.mean(dim=(2, 3), keepdim=True) + 1e-6)
        maps.append(x)
    if len(maps) < 2:
        return maps[0].new_tensor(0.0) if maps else None
    loss = maps[0].new_tensor(0.0)
    for a, b in zip(maps[:-1], maps[1:]):
        b = F.interpolate(b, size=a.shape[-2:], mode="bilinear", align_corners=False)
        loss = loss + F.l1_loss(a, b)
    return loss / (len(maps) - 1)


def evaluate(model, loader, ce_loss, dice_loss, device):
    model.eval()
    total = 0.0
    tp = tn = fp = fn = 0
    with torch.no_grad():
        for batch in loader:
            x = batch["image"].to(device, non_blocking=True)
            y = batch["label"].to(device, non_blocking=True).long()
            out, _ = model(x)
            total += semantic_loss(out, y, ce_loss, dice_loss).item()
            pred = out.argmax(1)
            tp += ((pred == 1) & (y == 1)).sum().item()
            tn += ((pred == 0) & (y == 0)).sum().item()
            fp += ((pred == 1) & (y == 0)).sum().item()
            fn += ((pred == 0) & (y == 1)).sum().item()
    n = max(1, len(loader))
    pa = (tp + tn) / max(1, tp + tn + fp + fn)
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = 2 * precision * recall / max(1e-12, precision + recall)
    iou_fg = tp / max(1, tp + fp + fn)
    iou_bg = tn / max(1, tn + fn + fp)
    miou = 0.5 * (iou_fg + iou_bg)
    return {
        "val_loss": total / n,
        "PA": pa,
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "IoU_fg": iou_fg,
        "mIoU": miou,
    }


def stable(window, tolerance):
    if len(window) < window.maxlen:
        return False
    vals = list(window)
    return max(vals) - min(vals) <= tolerance


def main():
    p = argparse.ArgumentParser(description="Run one ablation setting and one random seed.")
    p.add_argument("--config", required=True)
    p.add_argument("--root_path", required=True)
    p.add_argument("--list_dir", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--cfg", required=True)
    p.add_argument("--source_ckpt", default="")
    p.add_argument("--img_size", type=int, default=256)
    p.add_argument("--num_classes", type=int, default=2)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--base_lr", type=float, default=1e-4)
    p.add_argument("--min_lr", type=float, default=1e-6)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--max_epochs", type=int, default=60)
    p.add_argument("--max_epochs_per_stage", type=int, default=20)
    p.add_argument("--sliding_window", type=int, default=3)
    p.add_argument("--tolerance", type=float, default=0.005)
    p.add_argument("--consecutive", type=int, default=3)
    p.add_argument("--perc_weight", type=float, default=None)
    p.add_argument("--clip_grad", type=float, default=5.0)
    p.add_argument("--seed", type=int, required=True)
    args = p.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    if args.perc_weight is None:
        args.perc_weight = float(cfg.get("perc_weight", 0.0))
    seed_everything(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    logging.basicConfig(filename=os.path.join(args.output_dir, "ablation.log"), level=logging.INFO,
                        format="[%(asctime)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info("CONFIG=%s", cfg)
    logging.info("SEED=%d", args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = get_config(args)
    config.defrost()
    config.MODEL.PRETRAIN_CKPT = args.source_ckpt
    config.DATA.IMG_SIZE = args.img_size
    config.MODEL.NUM_CLASSES = args.num_classes
    config.DATA.BATCH_SIZE = args.batch_size
    config.freeze()

    model = SwinUnet(config, img_size=args.img_size, num_classes=args.num_classes)
    if args.source_ckpt:
        load_checkpoint(model, args.source_ckpt)
    model.to(device)

    train_set = OrchardTransferDataset(args.root_path, args.list_dir, "train",
                                       TransferGenerator((args.img_size, args.img_size), True))
    val_set = OrchardTransferDataset(args.root_path, args.list_dir, "val",
                                     TransferGenerator((args.img_size, args.img_size), False))
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True)

    ce_loss = nn.CrossEntropyLoss()
    dice_loss = DiceLoss(args.num_classes)
    scaler = GradScaler(enabled=device.type == "cuda")
    progressive = bool(cfg.get("PU", False))
    use_sdb = bool(cfg.get("SDB", False))
    use_hpl = bool(cfg.get("HPL", False))

    if progressive:
        phases = ["stage1_decoder", "stage2_deep", "stage3_mid"]
        stage_epochs = max(1, args.max_epochs_per_stage)
    else:
        phases = ["full_network"]
        stage_epochs = args.max_epochs

    history = []
    best_miou = -1.0
    global_epoch = 0

    for phase in phases:
        set_trainable(model, progressive, None if phase == "full_network" else phase)
        remaining = min(stage_epochs, args.max_epochs - global_epoch)
        if remaining <= 0:
            break
        optimizer = build_optimizer(model, args.base_lr, args.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=remaining, eta_min=args.min_lr)
        loss_window = deque(maxlen=max(1, args.sliding_window))
        stable_hits = 0
        stage_start = global_epoch
        logging.info("PHASE=%s epochs=%d trainable=%d", phase, remaining,
                     sum(p.numel() for p in model.parameters() if p.requires_grad))

        for local_epoch in range(remaining):
            model.train()
            running_total = running_sem = running_hpl = 0.0
            optimizer.zero_grad(set_to_none=True)
            for step, batch in enumerate(tqdm(train_loader, desc=f"{cfg['name']} {phase} {local_epoch + 1}/{remaining}")):
                x = batch["image"].to(device, non_blocking=True)
                y = batch["label"].to(device, non_blocking=True).long()
                with autocast(enabled=device.type == "cuda"):
                    out, features = model(x)
                    l_sem = semantic_loss(out, y, ce_loss, dice_loss)
                    l_hpl = hpl_loss(features) if use_hpl else l_sem.new_tensor(0.0)
                    if l_hpl is None:
                        l_hpl = l_sem.new_tensor(0.0)
                    total = l_sem + args.perc_weight * l_hpl
                    total_for_backward = total
                scaler.scale(total_for_backward).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                running_total += total.item()
                running_sem += l_sem.item()
                running_hpl += l_hpl.item()
            scheduler.step()

            train_total = running_total / max(1, len(train_loader))
            train_sem = running_sem / max(1, len(train_loader))
            train_hpl = running_hpl / max(1, len(train_loader))
            metrics = evaluate(model, val_loader, ce_loss, dice_loss, device)
            loss_window.append(train_sem)
            if use_sdb and stable(loss_window, args.tolerance):
                stable_hits += 1
            else:
                stable_hits = 0

            can_switch = (progressive and use_sdb and stable_hits >= max(1, args.consecutive)
                          and (global_epoch - stage_start + 1) < remaining)
            history.append([global_epoch + 1, phase, train_total, train_sem, train_hpl,
                            metrics["val_loss"], metrics["PA"], metrics["Precision"],
                            metrics["Recall"], metrics["F1"], metrics["IoU_fg"],
                            metrics["mIoU"], optimizer.param_groups[0]["lr"], stable_hits])
            logging.info("epoch=%d phase=%s total=%.6f semantic=%.6f HPL=%.6f val=%.6f PA=%.6f F1=%.6f mIoU=%.6f stable=%d/%d",
                         global_epoch + 1, phase, train_total, train_sem, train_hpl,
                         metrics["val_loss"], metrics["PA"], metrics["F1"], metrics["mIoU"],
                         stable_hits, args.consecutive)

            state = {"model": unwrap(model).state_dict(), "epoch": global_epoch + 1,
                     "phase": phase, "seed": args.seed, "config": cfg,
                     "metrics": metrics, "history": history}
            torch.save(state, os.path.join(args.output_dir, "last.pth"))
            if metrics["mIoU"] > best_miou:
                best_miou = metrics["mIoU"]
                torch.save(state, os.path.join(args.output_dir, "best.pth"))
            global_epoch += 1
            if can_switch:
                logging.info("SDB_SWITCH %s -> next stage at epoch=%d", phase, global_epoch)
                break

    history_path = os.path.join(args.output_dir, "history.csv")
    with open(history_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "phase", "train_total", "semantic_loss", "HPL_loss",
                         "val_loss", "PA", "Precision", "Recall", "F1", "IoU_fg",
                         "mIoU", "lr", "stable_count"])
        writer.writerows(history)

    final_metrics = evaluate(model, val_loader, ce_loss, dice_loss, device)
    with open(os.path.join(args.output_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump({"experiment": cfg["name"], "seed": args.seed, **final_metrics}, f, indent=2)
    logging.info("Finished %s seed=%d epochs=%d best_mIoU=%.6f final_mIoU=%.6f",
                 cfg["name"], args.seed, global_epoch, best_miou, final_metrics["mIoU"])


if __name__ == "__main__":
    main()
