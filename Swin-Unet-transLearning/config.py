import os
import yaml
from yacs.config import CfgNode as CN

_C = CN()
_C.BASE = ['']

_C.DATA = CN()
_C.DATA.BATCH_SIZE = 16
_C.DATA.DATA_PATH = ''
_C.DATA.DATASET = 'apple_orchard'
_C.DATA.IMG_SIZE = 256
_C.DATA.INTERPOLATION = 'bicubic'
_C.DATA.ZIP_MODE = False
_C.DATA.CACHE_MODE = 'part'
_C.DATA.PIN_MEMORY = True
_C.DATA.NUM_WORKERS = 4

_C.MODEL = CN()
_C.MODEL.TYPE = 'swin'
_C.MODEL.NAME = 'swin_tiny_patch4_window8_256'
_C.MODEL.PRETRAIN_CKPT = ''
_C.MODEL.RESUME = ''
_C.MODEL.NUM_CLASSES = 2
_C.MODEL.DROP_RATE = 0.0
_C.MODEL.DROP_PATH_RATE = 0.1
_C.MODEL.LABEL_SMOOTHING = 0.0

_C.MODEL.SWIN = CN()
_C.MODEL.SWIN.PATCH_SIZE = 4
_C.MODEL.SWIN.IN_CHANS = 3
_C.MODEL.SWIN.EMBED_DIM = 96
_C.MODEL.SWIN.DEPTHS = [2, 2, 6, 2]
_C.MODEL.SWIN.DECODER_DEPTHS = [2, 2, 6, 2]
_C.MODEL.SWIN.NUM_HEADS = [3, 6, 12, 24]
_C.MODEL.SWIN.WINDOW_SIZE = 8
_C.MODEL.SWIN.MLP_RATIO = 4.0
_C.MODEL.SWIN.QKV_BIAS = True
_C.MODEL.SWIN.QK_SCALE = None
_C.MODEL.SWIN.APE = False
_C.MODEL.SWIN.PATCH_NORM = True
_C.MODEL.SWIN.FINAL_UPSAMPLE = 'expand_first'

_C.TRAIN = CN()
_C.TRAIN.START_EPOCH = 0
_C.TRAIN.EPOCHS = 60
_C.TRAIN.WARMUP_EPOCHS = 0
_C.TRAIN.WEIGHT_DECAY = 1e-4
_C.TRAIN.BASE_LR = 1e-4
_C.TRAIN.MIN_LR = 1e-6
_C.TRAIN.CLIP_GRAD = 5.0
_C.TRAIN.ACCUMULATION_STEPS = 1
_C.TRAIN.USE_CHECKPOINT = False

_C.TRAIN.PHF = CN()
_C.TRAIN.PHF.ENABLE = True
_C.TRAIN.PHF.INITIAL_PHASE = 'decoder_only'
_C.TRAIN.PHF.MAX_EPOCHS_PER_STAGE = 20
_C.TRAIN.PHF.SLIDING_WINDOW = 3
_C.TRAIN.PHF.TOLERANCE = 0.005
_C.TRAIN.PHF.CONSECUTIVE = 3
_C.TRAIN.PHF.SEM_WEIGHT = 1.0
_C.TRAIN.PHF.PERC_WEIGHT = 0.05

_C.TRAIN.OPTIMIZER = CN()
_C.TRAIN.OPTIMIZER.NAME = 'adamw'
_C.TRAIN.OPTIMIZER.EPS = 1e-8
_C.TRAIN.OPTIMIZER.BETAS = (0.9, 0.999)

_C.AUG = CN()
_C.AUG.COLOR_JITTER = 0.0
_C.AUG.MIXUP = 0.0
_C.AUG.CUTMIX = 0.0

_C.TEST = CN()
_C.TEST.CROP = True
_C.AMP_OPT_LEVEL = ''
_C.OUTPUT = './results/apple_transfer'
_C.TAG = 'Trans_GAN_SwinUNet'
_C.SAVE_FREQ = 1
_C.PRINT_FREQ = 20
_C.SEED = 1234
_C.EVAL_MODE = False
_C.THROUGHPUT_MODE = False
_C.LOCAL_RANK = 0


def _update_config_from_file(config, cfg_file):
    config.defrost()
    with open(cfg_file, 'r', encoding='utf-8') as f:
        yaml_cfg = yaml.load(f, Loader=yaml.FullLoader)
    for cfg in yaml_cfg.setdefault('BASE', ['']):
        if cfg:
            _update_config_from_file(config, os.path.join(os.path.dirname(cfg_file), cfg))
    config.merge_from_file(cfg_file)
    config.freeze()


def update_config(config, args):
    _update_config_from_file(config, args.cfg)
    config.defrost()
    if getattr(args, 'opts', None):
        config.merge_from_list(args.opts)
    if getattr(args, 'batch_size', None):
        config.DATA.BATCH_SIZE = args.batch_size
    if getattr(args, 'resume', None):
        config.MODEL.RESUME = args.resume
    if getattr(args, 'tag', None):
        config.TAG = args.tag
    config.freeze()


def get_config(args):
    config = _C.clone()
    update_config(config, args)
    return config
