"""Independent evaluation for a saved ablation checkpoint.

Use this after training to recompute PA, Precision, Recall, F1, foreground IoU,
and binary mean IoU from the validation split. This avoids relying on a training
loop's metric naming and makes the manuscript table reproducible.
"""
import argparse
import json
import os
import sys

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
SWIN_ROOT = os.path.join(ROOT, "Swin-Unet-transLearning")
if SWIN_ROOT not in sys.path:
    sys.path.insert(0, SWIN_ROOT)

from config import get_config  # noqa: E402
from datasets.dataset_transfer import OrchardTransferDataset, TransferGenerator  # noqa: E402
from networks.vision_transformer import SwinUnet  # noqa: E402


def load_checkpoint(model, path):
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
    print(f"Loaded {len(loadable)} tensors; missing={len(msg.missing_keys)} unexpected={len(msg.unexpected_keys)}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--root_path", required=True)
    p.add_argument("--list_dir", required=True)
    p.add_argument("--cfg", required=True)
    p.add_argument("--output", default="")
    p.add_argument("--img_size", type=int, default=256)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--num_workers", type=int, default=4)
    args = p.parse_args()

    class Args:
        pass
    cfg_args = Args()
    cfg_args.cfg = args.cfg
    cfg_args.opts = None
    cfg = get_config(cfg_args)
    cfg.defrost()
    cfg.DATA.IMG_SIZE = args.img_size
    cfg.DATA.BATCH_SIZE = args.batch_size
    cfg.MODEL.NUM_CLASSES = 2
    cfg.freeze()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SwinUnet(cfg, img_size=args.img_size, num_classes=2)
    load_checkpoint(model, args.checkpoint)
    model.to(device).eval()

    ds = OrchardTransferDataset(args.root_path, args.list_dir, "val",
                                TransferGenerator((args.img_size, args.img_size), False))
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, pin_memory=True)
    ce = nn.CrossEntropyLoss()
    total_loss = 0.0
    tp = tn = fp = fn = 0
    with torch.no_grad():
        for batch in loader:
            x = batch["image"].to(device, non_blocking=True)
            y = batch["label"].to(device, non_blocking=True).long()
            out, _ = model(x)
            total_loss += ce(out, y).item()
            pred = out.argmax(1)
            tp += ((pred == 1) & (y == 1)).sum().item()
            tn += ((pred == 0) & (y == 0)).sum().item()
            fp += ((pred == 1) & (y == 0)).sum().item()
            fn += ((pred == 0) & (y == 1)).sum().item()

    pa = (tp + tn) / max(1, tp + tn + fp + fn)
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = 2 * precision * recall / max(1e-12, precision + recall)
    iou_fg = tp / max(1, tp + fp + fn)
    iou_bg = tn / max(1, tn + fp + fn)
    miou = 0.5 * (iou_fg + iou_bg)
    result = {
        "checkpoint": os.path.abspath(args.checkpoint),
        "val_loss_CE": total_loss / max(1, len(loader)),
        "PA": pa,
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "IoU_fg": iou_fg,
        "IoU_bg": iou_bg,
        "mIoU": miou,
        "TP": tp, "TN": tn, "FP": fp, "FN": fn
    }
    print(json.dumps(result, indent=2))
    output = args.output or os.path.join(os.path.dirname(args.checkpoint), "evaluation.json")
    with open(output, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(f"Saved: {output}")


if __name__ == "__main__":
    main()
