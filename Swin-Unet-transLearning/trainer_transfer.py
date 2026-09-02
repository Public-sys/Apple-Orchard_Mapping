import logging
import os
import random
import sys
from typing import Dict

import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from utils import DiceLoss


def unwrap(model):
    return model.module if hasattr(model, 'module') else model


def set_phase(model, phase: str):
    """Progressive unfreezing for source-pretrained Swin-Unet.

    decoder_stage4: decoder + deepest encoder stage (stage 4)
    decoder_stage4_stage3: additionally unfreeze encoder stage 3
    full: all parameters
    """
    m = unwrap(model)
    net = m.swin_unet
    for p in m.parameters():
        p.requires_grad = False

    if phase == 'decoder_stage4':
        modules = [net.layers[3], net.layers_up, net.norm, net.norm_up, net.up, net.output]
    elif phase == 'decoder_stage4_stage3':
        modules = [net.layers[2], net.layers[3], net.layers_up, net.norm, net.norm_up, net.up, net.output]
    elif phase == 'full':
        for p in m.parameters():
            p.requires_grad = True
        return
    else:
        raise ValueError(f'Unknown phase: {phase}')

    for module in modules:
        if module is not None:
            for p in module.parameters():
                p.requires_grad = True


def build_optimizer(model, lr, weight_decay=0.01):
    trainable = [p for p in model.parameters() if p.requires_grad]
    return torch.optim.AdamW(trainable, lr=lr, betas=(0.9, 0.999), weight_decay=weight_decay)


def feature_map_loss(features):
    """Feature consistency loss without assuming equal channel counts/resolutions."""
    if len(features) < 2:
        return features[0].new_tensor(0.0) if features else torch.tensor(0.0)
    maps = []
    for f in features:
        if f.ndim != 4:
            continue
        x = torch.mean(torch.abs(f), dim=1, keepdim=True)
        x = x / (x.mean(dim=(2, 3), keepdim=True) + 1e-6)
        maps.append(x)
    loss = maps[0].new_tensor(0.0) if maps else torch.tensor(0.0)
    for a, b in zip(maps[:-1], maps[1:]):
        b = torch.nn.functional.interpolate(b, size=a.shape[-2:], mode='bilinear', align_corners=False)
        loss = loss + torch.mean(torch.abs(a - b))
    return loss / max(1, len(maps) - 1)


def evaluate(model, loader, device, ce_loss, dice_loss):
    model.eval()
    total = 0.0
    inter = 0.0
    union = 0.0
    with torch.no_grad():
        for batch in loader:
            image = batch['image'].to(device, non_blocking=True)
            label = batch['label'].to(device, non_blocking=True).long()
            outputs, _ = model(image)
            loss = 0.5 * ce_loss(outputs, label) + 0.5 * dice_loss(outputs, label, softmax=True)
            total += loss.item()
            pred = outputs.argmax(1)
            p = pred == 1
            g = label == 1
            inter += (p & g).sum().item()
            union += (p | g).sum().item()
    return total / max(1, len(loader)), inter / max(1, union)


def trainer_transfer(args, model, snapshot_path):
    from datasets.dataset_synapse import Synapse_dataset, RandomGenerator

    os.makedirs(snapshot_path, exist_ok=True)
    logging.basicConfig(filename=os.path.join(snapshot_path, 'transfer.log'), level=logging.INFO,
                        format='[%(asctime)s] %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info('Transfer-learning configuration: %s', args)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    tfm = transforms.Compose([RandomGenerator(output_size=[args.img_size, args.img_size])])
    train_set = Synapse_dataset(args.root_path, args.list_dir, 'train', transform=tfm)
    val_set = Synapse_dataset(args.root_path, args.list_dir, 'val', transform=tfm)
    train_loader = DataLoader(train_set, batch_size=args.batch_size * args.n_gpu, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True, drop_last=False)
    val_loader = DataLoader(val_set, batch_size=args.batch_size * args.n_gpu, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True)

    ce_loss = nn.CrossEntropyLoss()
    dice_loss = DiceLoss(args.num_classes)
    scaler = GradScaler(enabled=(device.type == 'cuda'))

    # Stage schedule is explicit and reproducible; all target labels are used in each stage.
    schedule = [
        ('decoder_stage4', args.stage4_epochs, args.base_lr),
        ('decoder_stage4_stage3', args.stage3_epochs, args.base_lr * args.lr_decay),
        ('full', args.full_fine_tune_epochs, args.base_lr * args.lr_decay * args.lr_decay),
    ]

    global_epoch = 0
    best_miou = -1.0
    history = []

    for phase, phase_epochs, lr in schedule:
        if phase_epochs <= 0:
            continue
        set_phase(model, phase)
        optimizer = build_optimizer(model, lr, args.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=phase_epochs,
                                                                eta_min=args.min_lr)
        logging.info('===== %s: %d epochs, lr=%g =====', phase, phase_epochs, lr)
        logging.info('Trainable parameters: %d', sum(p.numel() for p in model.parameters() if p.requires_grad))

        for local_epoch in range(phase_epochs):
            model.train()
            running = 0.0
            optimizer.zero_grad(set_to_none=True)
            pbar = tqdm(train_loader, desc=f'{phase} epoch {local_epoch + 1}/{phase_epochs}')
            for step, batch in enumerate(pbar):
                image = batch['image'].to(device, non_blocking=True)
                label = batch['label'].to(device, non_blocking=True).long()
                with autocast(enabled=(device.type == 'cuda')):
                    outputs, features = model(image)
                    l_ce = ce_loss(outputs, label)
                    l_dice = dice_loss(outputs, label, softmax=True)
                    l_sem = 0.5 * l_ce + 0.5 * l_dice
                    l_perc = feature_map_loss(features)
                    total = l_sem + args.perc_weight * l_perc
                    total = total / max(1, args.accumulation_steps)
                scaler.scale(total).backward()
                if (step + 1) % max(1, args.accumulation_steps) == 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)
                running += total.item() * max(1, args.accumulation_steps)
                pbar.set_postfix(loss=f'{running / (step + 1):.4f}', lr=f'{optimizer.param_groups[0]["lr"]:.2e}')

            scheduler.step()
            val_loss, miou = evaluate(model, val_loader, device, ce_loss, dice_loss)
            epoch_loss = running / max(1, len(train_loader))
            history.append((global_epoch, phase, epoch_loss, val_loss, miou, optimizer.param_groups[0]['lr']))
            logging.info('Epoch %d | phase=%s | train=%.5f | val=%.5f | mIoU=%.5f',
                         global_epoch, phase, epoch_loss, val_loss, miou)

            torch.save({'model': unwrap(model).state_dict(), 'epoch': global_epoch,
                        'phase': phase, 'miou': miou, 'history': history},
                       os.path.join(snapshot_path, 'last.pth'))
            if miou > best_miou:
                best_miou = miou
                torch.save({'model': unwrap(model).state_dict(), 'epoch': global_epoch,
                            'phase': phase, 'miou': miou, 'history': history},
                           os.path.join(snapshot_path, 'best.pth'))
            global_epoch += 1

    np.savetxt(os.path.join(snapshot_path, 'history.csv'), np.array(history, dtype=object),
               delimiter=',', fmt='%s',
               header='epoch,phase,train_loss,val_loss,mIoU,lr', comments='')
    logging.info('Finished. Best validation mIoU=%.5f', best_miou)
    return history
