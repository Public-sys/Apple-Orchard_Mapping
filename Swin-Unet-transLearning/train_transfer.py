import argparse
import os
import random
import numpy as np
import torch
import torch.backends.cudnn as cudnn

from networks.vision_transformer import SwinUnet
from config import get_config
from trainer_transfer import trainer_transfer

parser = argparse.ArgumentParser(description='JL1KF01A -> Sentinel-2 apple orchard transfer learning')
parser.add_argument('--root_path', required=True, help='target Sentinel-2 npz directory')
parser.add_argument('--list_dir', required=True, help='directory containing train.txt and val.txt')
parser.add_argument('--output_dir', required=True)
parser.add_argument('--cfg', required=True)
parser.add_argument('--source_ckpt', required=True, help='source-domain pretrained Swin-Unet checkpoint')
parser.add_argument('--batch_size', type=int, default=4)
parser.add_argument('--n_gpu', type=int, default=1)
parser.add_argument('--num_workers', type=int, default=4)
parser.add_argument('--img_size', type=int, default=256)
parser.add_argument('--num_classes', type=int, default=2)
parser.add_argument('--base_lr', type=float, default=1e-4)
parser.add_argument('--min_lr', type=float, default=1e-6)
parser.add_argument('--weight_decay', type=float, default=0.01)
parser.add_argument('--stage4_epochs', type=int, default=10)
parser.add_argument('--stage3_epochs', type=int, default=20)
parser.add_argument('--full_fine_tune_epochs', type=int, default=120)
parser.add_argument('--lr_decay', type=float, default=0.1)
parser.add_argument('--perc_weight', type=float, default=0.05)
parser.add_argument('--clip_grad', type=float, default=5.0)
parser.add_argument('--accumulation_steps', type=int, default=1)
parser.add_argument('--seed', type=int, default=1234)
parser.add_argument('--opts', nargs='+', default=None)
args = parser.parse_args()

# Config is kept compatible with the existing Swin-Unet implementation.
config = get_config(args)
config.defrost()
config.MODEL.PRETRAIN_CKPT = args.source_ckpt
config.DATA.IMG_SIZE = args.img_size
config.MODEL.NUM_CLASSES = args.num_classes
config.DATA.BATCH_SIZE = args.batch_size
config.freeze()

os.makedirs(args.output_dir, exist_ok=True)
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
cudnn.deterministic = True
cudnn.benchmark = False
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(args.seed)

model = SwinUnet(config, img_size=args.img_size, num_classes=args.num_classes)

# Flexible checkpoint loader: accepts {'model': state_dict}, {'state_dict': ...}, or raw state_dict.
def load_source_checkpoint(model, path):
    ckpt = torch.load(path, map_location='cpu')
    state = ckpt.get('model', ckpt.get('state_dict', ckpt)) if isinstance(ckpt, dict) else ckpt
    state = {k.replace('module.', ''): v for k, v in state.items()}
    target = model.swin_unet.state_dict()
    loadable = {}
    skipped = []
    for k, v in state.items():
        # Accept checkpoints saved from Swin-Unet wrappers.
        kk = k[17:] if k.startswith('swin_unet.swin_unet.') else k
        if kk.startswith('swin_unet.'):
            kk = kk[len('swin_unet.'):]
        if kk in target and target[kk].shape == v.shape:
            loadable[kk] = v
        else:
            skipped.append(k)
    msg = model.swin_unet.load_state_dict(loadable, strict=False)
    print(f'Loaded {len(loadable)} tensors from source checkpoint; skipped {len(skipped)} tensors.')
    print(msg)

load_source_checkpoint(model, args.source_ckpt)
trainer_transfer(args, model, args.output_dir)
