import logging
import os
import sys
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
    m = unwrap(model)
    net = m.swin_unet
    for p in m.parameters():
        p.requires_grad = False
    if phase == 'decoder_stage4':
        modules = [net.layers[3], net.layers_up, net.norm, net.norm_up, net.up, net.output]
    elif phase == 'decoder_stage4_stage3':
        modules = [net.layers[2], net.layers[3], net.layers_up, net.norm, net.norm_up, net.up, net.output]
    elif phase == 'full':
        for p in m.parameters(): p.requires_grad = True
        return
    else:
        raise ValueError(phase)
    for module in modules:
        if module is not None:
            for p in module.parameters(): p.requires_grad = True


def build_optimizer(model, lr, weight_decay):
    return torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr,
                             betas=(0.9, 0.999), weight_decay=weight_decay)


def feature_map_loss(features):
    if not features or len(features) < 2:
        return None
    maps = []
    for f in features:
        if f.ndim != 4: continue
        x = torch.mean(torch.abs(f), dim=1, keepdim=True)
        x = x / (x.mean(dim=(2,3), keepdim=True) + 1e-6)
        maps.append(x)
    if len(maps) < 2: return None
    loss = maps[0].new_tensor(0.)
    for a, b in zip(maps[:-1], maps[1:]):
        b = F.interpolate(b, size=a.shape[-2:], mode='bilinear', align_corners=False)
        loss = loss + F.l1_loss(a, b)
    return loss / (len(maps)-1)


def evaluate(model, loader, ce_loss, dice_loss, device):
    model.eval(); total=0.; inter=0.; union=0.
    with torch.no_grad():
        for batch in loader:
            x=batch['image'].to(device); y=batch['label'].to(device).long()
            out,_=model(x)
            total += (0.5*ce_loss(out,y)+0.5*dice_loss(out,y,softmax=True)).item()
            p=out.argmax(1)==1; g=y==1
            inter += (p&g).sum().item(); union += (p|g).sum().item()
    return total/max(1,len(loader)), inter/max(1,union)


def trainer_transfer(args, model, snapshot_path):
    os.makedirs(snapshot_path, exist_ok=True)
    logging.basicConfig(filename=os.path.join(snapshot_path,'transfer.log'), level=logging.INFO,
                        format='[%(asctime)s] %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'); model.to(device)
    train_tf=TransferGenerator((args.img_size,args.img_size), True)
    val_tf=TransferGenerator((args.img_size,args.img_size), False)
    train_set=OrchardTransferDataset(args.root_path,args.list_dir,'train',train_tf)
    val_set=OrchardTransferDataset(args.root_path,args.list_dir,'val',val_tf)
    train_loader=DataLoader(train_set,batch_size=args.batch_size*args.n_gpu,shuffle=True,num_workers=args.num_workers,pin_memory=True)
    val_loader=DataLoader(val_set,batch_size=args.batch_size*args.n_gpu,shuffle=False,num_workers=args.num_workers,pin_memory=True)
    ce_loss=nn.CrossEntropyLoss(); dice_loss=DiceLoss(args.num_classes)
    scaler=GradScaler(enabled=device.type=='cuda')
    schedule=[('decoder_stage4',args.stage4_epochs,args.base_lr),
              ('decoder_stage4_stage3',args.stage3_epochs,args.base_lr*args.lr_decay),
              ('full',args.full_fine_tune_epochs,args.base_lr*args.lr_decay**2)]
    best=-1.; history=[]; epoch_id=0
    for phase,epochs,lr in schedule:
        if epochs<=0: continue
        set_phase(model,phase); opt=build_optimizer(model,lr,args.weight_decay)
        sch=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=epochs,eta_min=args.min_lr)
        logging.info('PHASE=%s epochs=%d lr=%g trainable=%d',phase,epochs,lr,sum(p.numel() for p in model.parameters() if p.requires_grad))
        for e in range(epochs):
            model.train(); running=0.; opt.zero_grad(set_to_none=True)
            for step,batch in enumerate(tqdm(train_loader,desc=f'{phase} {e+1}/{epochs}')):
                x=batch['image'].to(device,non_blocking=True); y=batch['label'].to(device,non_blocking=True).long()
                with autocast(enabled=device.type=='cuda'):
                    out,features=model(x); l_sem=.5*ce_loss(out,y)+.5*dice_loss(out,y,softmax=True)
                    lp=feature_map_loss(features); total=l_sem+(args.perc_weight*lp if lp is not None else 0.)
                    total=total/max(1,args.accumulation_steps)
                scaler.scale(total).backward()
                if (step+1)%max(1,args.accumulation_steps)==0:
                    scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(model.parameters(),args.clip_grad)
                    scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True)
                running += total.item()*max(1,args.accumulation_steps)
            sch.step(); vl,miou=evaluate(model,val_loader,ce_loss,dice_loss,device)
            tl=running/max(1,len(train_loader)); history.append([epoch_id,phase,tl,vl,miou,opt.param_groups[0]['lr']])
            logging.info('epoch=%d phase=%s train=%.5f val=%.5f mIoU=%.5f',epoch_id,phase,tl,vl,miou)
            state={'model':unwrap(model).state_dict(),'epoch':epoch_id,'phase':phase,'miou':miou,'history':history}
            torch.save(state,os.path.join(snapshot_path,'last.pth'))
            if miou>best:
                best=miou; torch.save(state,os.path.join(snapshot_path,'best.pth'))
            epoch_id+=1
    np.savetxt(os.path.join(snapshot_path,'history.csv'),np.asarray(history,dtype=object),delimiter=',',fmt='%s',header='epoch,phase,train_loss,val_loss,mIoU,lr',comments='')
    logging.info('Finished; best validation mIoU=%.5f',best)
