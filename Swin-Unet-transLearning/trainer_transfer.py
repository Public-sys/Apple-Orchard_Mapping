import logging
import os
import sys
from collections import deque
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from tqdm import tqdm
from datasets.dataset_transfer import OrchardTransferDataset, TransferGenerator
from utils import DiceLoss


def unwrap(model):
    return model.module if hasattr(model, 'module') else model


def set_phase(model, phase):
    """Progressive unfreezing: I decoder only; II + deepest encoder; III + middle encoder.
    The shallowest encoder stage remains frozen throughout transfer learning.
    """
    m = unwrap(model)
    net = m.swin_unet
    for p in m.parameters():
        p.requires_grad = False

    decoder_modules = [net.layers_up, net.norm, net.norm_up, net.up, net.output]
    if phase == 'stage1_decoder':
        modules = decoder_modules
    elif phase == 'stage2_deep':
        modules = decoder_modules + [net.layers[3]]
    elif phase == 'stage3_mid':
        modules = decoder_modules + [net.layers[3], net.layers[2]]
    else:
        raise ValueError(f'Unknown transfer phase: {phase}')

    for module in modules:
        if module is not None:
            for p in module.parameters():
                p.requires_grad = True


def build_optimizer(model, lr, weight_decay):
    params = [p for p in model.parameters() if p.requires_grad]
    if not params:
        raise RuntimeError('No trainable parameters after progressive-unfreezing setup.')
    return torch.optim.AdamW(params, lr=lr, betas=(0.9, 0.999), weight_decay=weight_decay)


def feature_map_loss(features):
    """Hierarchical feature consistency loss used as the perceptual/feature term."""
    if not features or len(features) < 2:
        return features[0].new_tensor(0.) if features else None
    maps = []
    for f in features:
        if f.ndim != 4:
            continue
        x = torch.mean(torch.abs(f), dim=1, keepdim=True)
        x = x / (x.mean(dim=(2, 3), keepdim=True) + 1e-6)
        maps.append(x)
    if len(maps) < 2:
        return maps[0].new_tensor(0.) if maps else None
    loss = maps[0].new_tensor(0.)
    for a, b in zip(maps[:-1], maps[1:]):
        b = F.interpolate(b, size=a.shape[-2:], mode='bilinear', align_corners=False)
        loss = loss + F.l1_loss(a, b)
    return loss / (len(maps) - 1)


def semantic_loss(logits, labels, ce_loss, dice_loss, sem_weight):
    l_ce = ce_loss(logits, labels)
    l_dice = dice_loss(logits, labels, softmax=True)
    return sem_weight * (0.5 * l_ce + 0.5 * l_dice)


def evaluate(model, loader, ce_loss, dice_loss, sem_weight, device):
    model.eval()
    total = 0.0
    inter = 0
    union = 0
    with torch.no_grad():
        for batch in loader:
            x = batch['image'].to(device, non_blocking=True)
            y = batch['label'].to(device, non_blocking=True).long()
            out, _ = model(x)
            total += semantic_loss(out, y, ce_loss, dice_loss, sem_weight).item()
            pred = out.argmax(1) == 1
            gt = y == 1
            inter += (pred & gt).sum().item()
            union += (pred | gt).sum().item()
    return total / max(1, len(loader)), inter / max(1, union)


def converged(loss_window, tolerance):
    if len(loss_window) < loss_window.maxlen:
        return False
    values = list(loss_window)
    return (max(values) - min(values)) <= tolerance


def trainer_transfer(args, model, snapshot_path):
    os.makedirs(snapshot_path, exist_ok=True)
    logging.basicConfig(filename=os.path.join(snapshot_path, 'transfer.log'), level=logging.INFO,
                        format='[%(asctime)s] %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    train_tf = TransferGenerator((args.img_size, args.img_size), True)
    val_tf = TransferGenerator((args.img_size, args.img_size), False)
    train_set = OrchardTransferDataset(args.root_path, args.list_dir, 'train', train_tf)
    val_set = OrchardTransferDataset(args.root_path, args.list_dir, 'val', val_tf)
    train_loader = DataLoader(train_set, batch_size=args.batch_size * args.n_gpu, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size * args.n_gpu, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True)

    ce_loss = nn.CrossEntropyLoss()
    dice_loss = DiceLoss(args.num_classes)
    scaler = GradScaler(enabled=device.type == 'cuda')

    phases = [
        ('stage1_decoder', args.base_lr),
        ('stage2_deep', args.base_lr),
        ('stage3_mid', args.base_lr),
    ]
    max_stage_epochs = max(1, args.max_epochs_per_stage)
    total_max_epochs = 3 * max_stage_epochs
    best = -1.0
    history = []
    epoch_id = 0
    consecutive_hits = 0
    loss_window = deque(maxlen=max(1, args.sliding_window))

    for phase_index, (phase, lr) in enumerate(phases):
        if epoch_id >= total_max_epochs:
            break
        set_phase(model, phase)
        opt = build_optimizer(model, lr, args.weight_decay)
        remaining = min(max_stage_epochs, total_max_epochs - epoch_id)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=remaining, eta_min=args.min_lr)
        stage_start = epoch_id
        consecutive_hits = 0
        loss_window.clear()
        logging.info('PHASE=%s max_epochs=%d lr=%g trainable=%d', phase, remaining, lr,
                     sum(p.numel() for p in model.parameters() if p.requires_grad))

        for local_epoch in range(remaining):
            model.train()
            running = 0.0
            opt.zero_grad(set_to_none=True)
            for step, batch in enumerate(tqdm(train_loader, desc=f'{phase} {local_epoch + 1}/{remaining}')):
                x = batch['image'].to(device, non_blocking=True)
                y = batch['label'].to(device, non_blocking=True).long()
                with autocast(enabled=device.type == 'cuda'):
                    out, features = model(x)
                    l_sem = semantic_loss(out, y, ce_loss, dice_loss, args.sem_weight)
                    l_perc = feature_map_loss(features)
                    total = l_sem + args.perc_weight * (l_perc if l_perc is not None else 0.0)
                    total = total / max(1, args.accumulation_steps)
                scaler.scale(total).backward()
                if (step + 1) % max(1, args.accumulation_steps) == 0:
                    scaler.unscale_(opt)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
                    scaler.step(opt)
                    scaler.update()
                    opt.zero_grad(set_to_none=True)
                running += total.item() * max(1, args.accumulation_steps)

            sch.step()
            train_loss = running / max(1, len(train_loader))
            val_loss, miou = evaluate(model, val_loader, ce_loss, dice_loss, args.sem_weight, device)
            history.append([epoch_id, phase, train_loss, val_loss, miou, opt.param_groups[0]['lr']])
            loss_window.append(train_loss)

            if converged(loss_window, args.tolerance):
                consecutive_hits += 1
            else:
                consecutive_hits = 0
            can_switch = (len(loss_window) == loss_window.maxlen and
                          consecutive_hits >= max(1, args.consecutive) and
                          (epoch_id - stage_start + 1) < remaining)

            logging.info('epoch=%d phase=%s train=%.6f val=%.6f mIoU=%.6f stable=%d/%d',
                         epoch_id, phase, train_loss, val_loss, miou,
                         consecutive_hits, args.consecutive)

            state = {'model': unwrap(model).state_dict(), 'epoch': epoch_id,
                     'phase': phase, 'miou': miou, 'history': history}
            torch.save(state, os.path.join(snapshot_path, 'last.pth'))
            if miou > best:
                best = miou
                torch.save(state, os.path.join(snapshot_path, 'best.pth'))

            epoch_id += 1
            if can_switch:
                logging.info('SDB_SWITCH phase=%s -> next_stage at epoch=%d', phase, epoch_id)
                break

    np.savetxt(os.path.join(snapshot_path, 'history.csv'), np.asarray(history, dtype=object),
               delimiter=',', fmt='%s', header='epoch,phase,train_loss,val_loss,mIoU,lr', comments='')
    logging.info('Finished; total_epochs=%d; best validation mIoU=%.6f', epoch_id, best)
