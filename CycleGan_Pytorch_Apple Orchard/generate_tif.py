# -*- coding: utf-8 -*-
"""Generate Sentinel-2-style GeoTIFF patches with a trained CycleGAN generator.

The script reads 3-band RGB GeoTIFFs, runs G_A (A -> B), and writes 3-band
uint8 GeoTIFFs while preserving CRS, transform, width, height, and band count.
It intentionally does not resize or crop inference images, so source labels stay
pixel-aligned with the generated images.
"""
import argparse
from pathlib import Path
import numpy as np
import rasterio
import torch

from models import networks


def to_uint8(arr, scale_max=None):
    arr = arr.astype(np.float32)
    if np.issubdtype(arr.dtype, np.integer):
        pass
    if scale_max is None:
        # For the manuscript pipeline the expected input is 8-bit RGB.
        if arr.max() > 255 or arr.min() < 0:
            raise ValueError(
                'Input TIFF is not 8-bit RGB. Supply --scale_max to explicitly '
                'map the source value range to 0-255.'
            )
        return np.clip(arr, 0, 255).astype(np.uint8)
    return np.clip(arr / float(scale_max) * 255.0, 0, 255).astype(np.uint8)


def load_generator(checkpoint, device, input_nc=3, output_nc=3, ngf=64,
                   netG='resnet_9blocks', norm='instance'):
    net = networks.define_G(input_nc, output_nc, ngf, netG, norm, False,
                            'normal', 0.02, gpu_ids=[])
    state = torch.load(checkpoint, map_location='cpu')
    if isinstance(state, dict) and 'state_dict' in state:
        state = state['state_dict']
    state = {k.replace('module.', ''): v for k, v in state.items()}
    net.load_state_dict(state, strict=True)
    net.to(device).eval()
    return net


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--input_dir', required=True)
    p.add_argument('--output_dir', required=True)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--netG', default='resnet_9blocks')
    p.add_argument('--ngf', type=int, default=64)
    p.add_argument('--scale_max', type=float, default=None,
                   help='Optional source value maximum for non-uint8 TIFFs, e.g. 10000 for scaled reflectance.')
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = p.parse_args()

    in_dir = Path(args.input_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = sorted(list(in_dir.glob('*.tif')) + list(in_dir.glob('*.tiff')))
    if not paths:
        raise RuntimeError(f'No TIFF files found in {in_dir}')

    device = torch.device(args.device)
    net = load_generator(args.checkpoint, device, ngf=args.ngf, netG=args.netG)

    with torch.no_grad():
        for src_path in paths:
            with rasterio.open(src_path) as src:
                arr = src.read()
                meta = src.meta.copy()
                transform = src.transform
                crs = src.crs

            if arr.ndim != 3 or arr.shape[0] < 3:
                raise ValueError(f'{src_path}: expected at least 3 bands, got {arr.shape}')
            arr = to_uint8(arr[:3], args.scale_max)
            h, w = arr.shape[1:]

            x = torch.from_numpy(arr.astype(np.float32) / 127.5 - 1.0)
            x = x.unsqueeze(0).to(device)
            y = net(x)
            y = ((y.squeeze(0).cpu().numpy() + 1.0) * 127.5).clip(0, 255).astype(np.uint8)

            if y.shape != (3, h, w):
                raise RuntimeError(f'{src_path}: generator changed shape {arr.shape} -> {y.shape}')

            meta.update(driver='GTiff', dtype='uint8', count=3, height=h, width=w,
                        transform=transform, crs=crs)
            dst_path = out_dir / f'{src_path.stem}_fakeB.tif'
            with rasterio.open(dst_path, 'w', **meta) as dst:
                dst.write(y)
            print(f'Generated: {src_path.name} -> {dst_path.name}')


if __name__ == '__main__':
    main()
