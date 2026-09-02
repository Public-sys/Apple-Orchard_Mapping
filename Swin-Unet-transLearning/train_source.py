import argparse
import os
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import get_config
from networks.vision_transformer import SwinUnet
from datasets.dataset_transfer import OrchardTransferDataset, TransferGenerator


def unwrap(model):
    return model.module if hasattr(model, 'module') else model


def main(args):
    os.makedirs(args.output_dir, exist_ok=True)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    config = get_config(args)
    config.defrost()
    config.DATA.BATCH_SIZE = args.batch_size
    config.DATA.IMG_SIZE = args.img_size
    config.MODEL.NUM_CLASSES = args.num_classes
    config.freeze()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SwinUnet(config, img_size=args.img_size, num_classes=args.num_classes).to(device)

    train_set = OrchardTransferDataset(args.root_path, args.list_dir, 'train',
                                       TransferGenerator((args.img_size, args.img_size), True))
    val_set = OrchardTransferDataset(args.root_path, args.list_dir, 'val',
                                     TransferGenerator((args.img_size, args.img_size), False))
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  betas=(0.9, 0.999), weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.min_lr)

    best = float('inf')
    for epoch in range(args.epochs):
        model.train()
        train_loss = 0.0
        for batch in tqdm(train_loader, desc=f'Source pretrain {epoch + 1}/{args.epochs}'):
            x = batch['image'].to(device, non_blocking=True)
            y = batch['label'].to(device, non_blocking=True).long()
            optimizer.zero_grad(set_to_none=True)
            logits, _ = model(x)
            loss = criterion(logits, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            train_loss += loss.item()
        train_loss /= max(1, len(train_loader))

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                x = batch['image'].to(device, non_blocking=True)
                y = batch['label'].to(device, non_blocking=True).long()
                logits, _ = model(x)
                val_loss += criterion(logits, y).item()
        val_loss /= max(1, len(val_loader))
        scheduler.step()

        state = {'model': unwrap(model).state_dict(), 'epoch': epoch,
                 'train_ce': train_loss, 'val_ce': val_loss}
        torch.save(state, os.path.join(args.output_dir, 'source_last.pth'))
        if val_loss < best:
            best = val_loss
            torch.save(state, os.path.join(args.output_dir, 'source_best.pth'))
        print(f'epoch={epoch:03d} train_CE={train_loss:.6f} val_CE={val_loss:.6f}')


parser = argparse.ArgumentParser(description='Phase 1 source-domain pretraining on CycleGAN-generated Sentinel-2-style images')
parser.add_argument('--root_path', required=True)
parser.add_argument('--list_dir', required=True)
parser.add_argument('--output_dir', required=True)
parser.add_argument('--cfg', required=True)
parser.add_argument('--batch_size', type=int, default=16)
parser.add_argument('--num_workers', type=int, default=4)
parser.add_argument('--img_size', type=int, default=256)
parser.add_argument('--num_classes', type=int, default=2)
parser.add_argument('--epochs', type=int, default=60)
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--min_lr', type=float, default=1e-6)
parser.add_argument('--weight_decay', type=float, default=1e-4)
parser.add_argument('--seed', type=int, default=1234)
parser.add_argument('--opts', nargs='+', default=None)
parser.add_argument('--resume', default='')
parser.add_argument('--tag', default='source_pretrain')

if __name__ == '__main__':
    main(parser.parse_args())
