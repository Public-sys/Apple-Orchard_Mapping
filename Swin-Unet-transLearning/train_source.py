"""Phase 1: source-domain pre-training on CycleGAN-generated TIFFs.

Split files contain paired entries:
    generated_image.tif,label.tif

This stage intentionally uses cross-entropy only, matching the manuscript's
source-domain pre-training description. The resulting source_best.pth can be
passed to train_transfer.py for target-domain progressive fine-tuning.
"""
import argparse
import csv
import os
import random
import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import get_config
from datasets.dataset_transfer import OrchardTransferDataset, TransferGenerator
from networks.vision_transformer import SwinUnet


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_checkpoint(model, path, device):
    ckpt = torch.load(path, map_location=device)
    state = ckpt.get('model', ckpt.get('state_dict', ckpt)) if isinstance(ckpt, dict) else ckpt
    clean = {}
    for k, v in state.items():
        key = k.replace('module.', '', 1)
        if key.startswith('swin_unet.'):
            key = key[len('swin_unet.'):]
        clean[key] = v
    missing, unexpected = model.swin_unet.load_state_dict(clean, strict=False)
    print(f'Checkpoint loaded: {path}; missing={len(missing)}, unexpected={len(unexpected)}')


def evaluate(model, loader, ce_loss, device):
    model.eval()
    total = 0.0
    inter = 0
    union = 0
    with torch.no_grad():
        for batch in loader:
            x = batch['image'].to(device, non_blocking=True)
            y = batch['label'].to(device, non_blocking=True).long()
            out, _ = model(x)
            total += ce_loss(out, y).item()
            pred = out.argmax(1) == 1
            gt = y == 1
            inter += (pred & gt).sum().item()
            union += (pred | gt).sum().item()
    return total / max(1, len(loader)), inter / max(1, union)


def main():
    p = argparse.ArgumentParser(description='Source-domain pre-training on CycleGAN-generated TIFFs.')
    p.add_argument('--root_path', required=True, help='Root directory containing image/label TIFFs.')
    p.add_argument('--list_dir', required=True, help='Directory containing train.txt and val.txt.')
    p.add_argument('--output_dir', required=True, help='Directory for source checkpoints and logs.')
    p.add_argument('--cfg', default='configs/apple_transfer.yaml')
    p.add_argument('--batch_size', type=int, default=16)
    p.add_argument('--epochs', type=int, default=60)
    p.add_argument('--img_size', type=int, default=256)
    p.add_argument('--num_classes', type=int, default=2)
    p.add_argument('--base_lr', type=float, default=1e-4)
    p.add_argument('--min_lr', type=float, default=1e-6)
    p.add_argument('--weight_decay', type=float, default=1e-4)
    p.add_argument('--num_workers', type=int, default=4)
    p.add_argument('--clip_grad', type=float, default=5.0)
    p.add_argument('--n_gpu', type=int, default=1)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--source_ckpt', default='', help='Optional Swin-Unet checkpoint for warm start.')
    p.add_argument('--opts', nargs='+', default=None)
    args = p.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Use the repository architecture; CLI values override the key training fields.
    config = get_config(args)
    config.DATA.BATCH_SIZE = args.batch_size
    config.DATA.IMG_SIZE = args.img_size
    config.MODEL.NUM_CLASSES = args.num_classes
    config.TRAIN.EPOCHS = args.epochs
    config.TRAIN.BASE_LR = args.base_lr
    config.TRAIN.MIN_LR = args.min_lr
    config.TRAIN.WEIGHT_DECAY = args.weight_decay

    model = SwinUnet(config, img_size=args.img_size, num_classes=args.num_classes, zero_head=False)
    if args.source_ckpt:
        load_checkpoint(model, args.source_ckpt, device)
    model.to(device)

    train_set = OrchardTransferDataset(args.root_path, args.list_dir, 'train',
                                       TransferGenerator((args.img_size, args.img_size), True))
    val_set = OrchardTransferDataset(args.root_path, args.list_dir, 'val',
                                     TransferGenerator((args.img_size, args.img_size), False))
    train_loader = DataLoader(train_set, batch_size=args.batch_size * args.n_gpu, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size * args.n_gpu, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True)

    ce_loss = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.base_lr, betas=(0.9, 0.999),
                                  weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.min_lr)
    scaler = GradScaler(enabled=device.type == 'cuda')
    best = -1.0
    history = []

    for epoch in range(args.epochs):
        model.train()
        running = 0.0
        optimizer.zero_grad(set_to_none=True)
        for batch in tqdm(train_loader, desc=f'source {epoch + 1}/{args.epochs}'):
            x = batch['image'].to(device, non_blocking=True)
            y = batch['label'].to(device, non_blocking=True).long()
            with autocast(enabled=device.type == 'cuda'):
                out, _ = model(x)
                loss = ce_loss(out, y)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            running += loss.item()
        scheduler.step()

        train_loss = running / max(1, len(train_loader))
        val_loss, miou = evaluate(model, val_loader, ce_loss, device)
        lr = optimizer.param_groups[0]['lr']
        history.append([epoch + 1, train_loss, val_loss, miou, lr])
        state = {'model': model.state_dict(), 'epoch': epoch + 1, 'miou': miou, 'history': history}
        torch.save(state, os.path.join(args.output_dir, 'source_last.pth'))
        if miou > best:
            best = miou
            torch.save(state, os.path.join(args.output_dir, 'source_best.pth'))
        print(f'epoch={epoch + 1} train_ce={train_loss:.6f} val_ce={val_loss:.6f} mIoU={miou:.6f} lr={lr:.3e}')

    with open(os.path.join(args.output_dir, 'source_history.csv'), 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['epoch', 'train_ce', 'val_ce', 'mIoU', 'lr'])
        writer.writerows(history)
    print(f'Finished source pre-training. Best validation mIoU={best:.6f}')


if __name__ == '__main__':
    main()
