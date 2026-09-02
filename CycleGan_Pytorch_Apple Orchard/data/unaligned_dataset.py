import os
import numpy as np
from PIL import Image
from data.base_dataset import BaseDataset, is_image_file, get_transform
import rasterio


class UnalignedDataset(BaseDataset):
    """Unaligned CycleGAN dataset for 8-bit RGB GeoTIFF patches.

    The manuscript pipeline uses 8-bit RGB imagery. To avoid silently changing
    reflectance values, non-uint8 TIFFs are rejected rather than cast. GeoTIFF
    metadata are cached for later inference/output handling.
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

        self.meta_A = self._collect_metadata(self.A_paths)
        self.meta_B = self._collect_metadata(self.B_paths)

    @staticmethod
    def _collect_metadata(paths):
        metadata = {}
        for path in paths:
            with rasterio.open(path) as src:
                metadata[path] = {
                    'meta': src.meta.copy(),
                    'transform': src.transform,
                    'crs': src.crs,
                    'height': src.height,
                    'width': src.width,
                    'count': src.count,
                    'dtype': src.dtypes[0],
                }
        return metadata

    @staticmethod
    def _read_rgb_tif(path):
        with rasterio.open(path) as src:
            if src.count < 3:
                raise ValueError(f'{path}: CycleGAN requires at least 3 bands, got {src.count}')
            if src.dtypes[0] != 'uint8' or any(dtype != 'uint8' for dtype in src.dtypes[:3]):
                raise ValueError(
                    f'{path}: expected 8-bit RGB TIFF for CycleGAN, got dtypes={src.dtypes[:3]}. '
                    'Convert/scale the imagery explicitly before training rather than casting here.'
                )
            arr = src.read([1, 2, 3])
        arr = np.transpose(arr, (1, 2, 0))
        return Image.fromarray(arr, mode='RGB')

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
