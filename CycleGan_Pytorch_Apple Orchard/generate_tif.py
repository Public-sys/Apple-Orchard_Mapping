"""CycleGAN inference for 8-bit RGB GeoTIFF patches.

Preserves CRS, affine transform, width and height and performs no resizing.
"""
import argparse
import os
from pathlib import Path

import numpy as np
import rasterio
import torch

# networks.py is located in this CycleGAN directory.
from networks import define_G


def load_generator(model, checkpoint, device):
    ckpt = torch.load(checkpoint, map_location=device)
    state = ckpt
    if isinstance(ckpt, dict):
        for key in ('state_dict', 'model', 'netG', 'G_A', 'G_B'):
            if key in ckpt and isinstance(ckpt[key], dict):
                state = ckpt[key]
                break
    if not isinstance(state, dict):
        raise ValueError(f'Unsupported checkpoint format: {checkpoint}')

    clean = {}
    for key, value in state.items():
        key = key.replace('module.', '', 1)
        for prefix in ('model.', 'netG.'):
            if key.startswith(prefix):
                key = key[len(prefix):]
        clean[key] = value

    missing, unexpected = model.load_state_dict(clean, strict=False)
    if missing:
        print(f'Warning: {len(missing)} parameters missing.')
    if unexpected:
        print(f'Warning: {len(unexpected)} checkpoint parameters unused.')
    model.eval()


def read_rgb(path):
    with rasterio.open(path) as src:
        if src.count < 3:
            raise ValueError(f'{path}: expected at least 3 bands, got {src.count}')
        if any(dtype != 'uint8' for dtype in src.dtypes[:3]):
            raise ValueError(f'{path}: expected uint8 RGB TIFF, got {src.dtypes[:3]}')
        data = src.read([1, 2, 3])
        profile = src.profile.copy()
    return data, profile


def translate(model, data, device):
    x = torch.from_numpy(data.astype(np.float32) / 127.5 - 1.0).unsqueeze(0).to(device)
    with torch.inference_mode():
        y = model(x)
        if isinstance(y, (tuple, list)):
            y = y[0]
    y = (y.squeeze(0).cpu().numpy() + 1.0) * 127.5
    return np.clip(np.rint(y), 0, 255).astype(np.uint8)


def main():
    p = argparse.ArgumentParser(description='CycleGAN inference for RGB GeoTIFF patches.')
    p.add_argument('--input_dir', required=True)
    p.add_argument('--output_dir', required=True)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--ngf', type=int, default=64)
    p.add_argument('--norm', choices=['batch', 'instance'], default='instance')
    p.add_argument('--netG', default='resnet_9blocks', choices=['resnet_9blocks', 'resnet_6blocks', 'unet_128', 'unet_256'])
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = p.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(args.device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but unavailable.')

    model = define_G(
        3, 3, args.ngf, args.netG, args.norm,
        use_dropout=False, init_type='normal', init_gain=0.02
    )
    load_generator(model, args.checkpoint, device)
    model.to(device)

    files = sorted(
        p for p in Path(args.input_dir).iterdir()
        if p.is_file() and p.suffix.lower() in {'.tif', '.tiff'}
    )
    if not files:
        raise RuntimeError(f'No TIFF files found in {args.input_dir}')

    for src_path in files:
        data, profile = read_rgb(src_path)
        output = translate(model, data, device)
        if output.shape != data.shape:
            raise RuntimeError(f'Shape changed: {data.shape} -> {output.shape}')

        profile.update(driver='GTiff', dtype='uint8', count=3, compress='deflate')
        dst_path = Path(args.output_dir) / src_path.name
        with rasterio.open(dst_path, 'w', **profile) as dst:
            dst.write(output)
            for i, name in enumerate(('Red', 'Green', 'Blue'), 1):
                dst.set_band_description(i, name)
        print(f'{src_path.name} -> {dst_path}')

    print(f'Generated {len(files)} GeoTIFFs in {args.output_dir}')


if __name__ == '__main__':
    main()
