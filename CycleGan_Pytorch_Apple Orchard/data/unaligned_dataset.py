import os
import numpy as np
from data.base_dataset import BaseDataset, is_image_file, get_transform
import rasterio
import torch


class UnalignedDataset(BaseDataset):
    """Unaligned CycleGAN dataset for RGB GeoTIFF patches.

    TIFF metadata are retained for inference output. Pixel values are converted
    to uint8 for the standard 3-channel CycleGAN pipeline; therefore the input
    TIFFs used here should already be 8-bit RGB patches.
    """

    def __init__(self, opt):
        BaseDataset.__init__(self, opt)
        self.dir_A = os.path.join(opt.dataroot, opt.phase + 'A')
        self.dir_B = os.path.join(opt.dataroot, opt.phase + 'B')
        self.A_paths = sorted([os.path.join(self.dir_A, f) for f in os.listdir(self.dir_A) if is_image_file(f)])
        self.B_paths = sorted([os.path.join(self.dir_B, f) for f in os.listdir(self.dir_B) if is_image_file(f)])
        self.A_size = len(self.A_paths)
        self.B_size = len(self.B_paths)
        if self.A_size == 0 or self.B_size == 0:
            raise RuntimeError(f'CycleGAN requires non-empty {opt.phase}A and {opt.phase}B directories.')
        self.transform_A = get_transform(opt, grayscale=(opt.input_nc == 1))
        self.transform_B = get_transform(opt, grayscale=(opt.output_nc == 1))

        self.meta_A = {}
        for path in self.A_paths:
            with rasterio.open(path) as src:
                self.meta_A[path] = {
                    'meta': src.meta.copy(),
                    'transform': src.transform,
                    'crs': src.crs,
                    'height': src.height,
                    'width': src.width,
                    'count': src.count,
                }
        self.meta_B = {}
        for path in self.B_paths:
            with rasterio.open(path) as src:
                self.meta_B[path] = {
                    'meta': src.meta.copy(),
                    'transform': src.transform,
                    'crs': src.crs,
                    'height': src.height,
                    'width': src.width,
                    'count': src.count,
                }

    @staticmethod
    def _read_rgb_tif(path):
        with rasterio.open(path) as src:
            arr = src.read()
        if arr.ndim != 3:
            raise ValueError(f'{path}: expected multi-band TIFF, got {arr.shape}')
        if arr.shape[0] < 3:
            raise ValueError(f'{path}: CycleGAN requires at least 3 input bands, got {arr.shape[0]}')
        # Use the first three bands as RGB, consistent with the manuscript setup.
        arr = arr[:3]
        return np.transpose(arr, (1, 2, 0)).astype(np.uint8)

    def __getitem__(self, index):
        A_path = self.A_paths[index % self.A_size]
        B_path = self.B_paths[index % self.B_size]
        img_A = self._read_rgb_tif(A_path)
        img_B = self._read_rgb_tif(B_path)
        A = self.transform_A(img_A)
        B = self.transform_B(img_B)
        return {'A': A, 'B': B, 'A_paths': A_path, 'B_paths': B_path}

    def __len__(self):
        return max(self.A_size, self.B_size)

    def get_meta(self, path, domain='A'):
        return (self.meta_A if domain == 'A' else self.meta_B)[path]
