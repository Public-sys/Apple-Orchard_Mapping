import os
import random
import numpy as np
import torch
from scipy import ndimage
from scipy.ndimage import zoom
from torch.utils.data import Dataset

try:
    import rasterio
except ImportError:
    rasterio = None


def random_rot_flip(image, label):
    k = np.random.randint(0, 4)
    image = np.rot90(image, k, axes=(-2, -1)) if image.ndim == 3 else np.rot90(image, k)
    label = np.rot90(label, k)
    axis = np.random.randint(1, 3) if image.ndim == 3 else np.random.randint(0, 2)
    image = np.flip(image, axis=axis).copy()
    label = np.flip(label, axis=axis - 1 if image.ndim == 3 else axis).copy()
    return image, label


def random_rotate(image, label):
    angle = np.random.uniform(-15, 15)
    if image.ndim == 3:
        image = np.stack([ndimage.rotate(c, angle, order=1, reshape=False, mode='reflect') for c in image])
    else:
        image = ndimage.rotate(image, angle, order=1, reshape=False, mode='reflect')
    label = ndimage.rotate(label, angle, order=0, reshape=False, mode='nearest')
    return image, label


def read_tif(path):
    if rasterio is None:
        raise ImportError('TIFF input requires rasterio. Install it with: pip install rasterio')
    with rasterio.open(path) as src:
        arr = src.read()
    return arr


def prepare_image(image):
    image = np.asarray(image)
    if image.ndim == 2:
        image = image[None, ...]
    elif image.ndim == 3:
        # Rasterio TIFFs are normally CHW. Legacy arrays may be HWC.
        if image.shape[0] <= 8:
            pass
        elif image.shape[-1] <= 8:
            image = np.transpose(image, (2, 0, 1))
        else:
            raise ValueError(f'Cannot infer channel dimension from image shape {image.shape}')
    else:
        raise ValueError(f'Expected 2-D or 3-D image, got {image.shape}')
    return image.astype(np.float32)


def normalize_image(image):
    """Convert common remote-sensing TIFF ranges to the model's [0, 1] input range.

    8-bit imagery is divided by 255. Float data already in [0,1] are preserved.
    Other integer/float ranges are min-max normalized per sample to avoid silently
    passing incompatible radiometric scales to the network.
    """
    image = image.astype(np.float32, copy=False)
    if np.issubdtype(image.dtype, np.floating) and np.nanmin(image) >= 0 and np.nanmax(image) <= 1.0:
        return image
    max_value = float(np.nanmax(image))
    min_value = float(np.nanmin(image))
    if min_value >= 0 and max_value <= 255:
        return image / 255.0
    if max_value > min_value:
        return (image - min_value) / (max_value - min_value)
    return np.zeros_like(image, dtype=np.float32)


class TransferGenerator:
    def __init__(self, output_size=(256, 256), augment=True):
        self.output_size = tuple(output_size)
        self.augment = augment

    def __call__(self, sample):
        image = prepare_image(sample['image'])
        label = np.asarray(sample['label']).squeeze().astype(np.int64)
        if self.augment:
            r = random.random()
            if r < 0.5:
                image, label = random_rot_flip(image, label)
            elif r < 0.75:
                image, label = random_rotate(image, label)

        _, h, w = image.shape
        oh, ow = self.output_size
        if (h, w) != (oh, ow):
            image = zoom(image, (1, oh / h, ow / w), order=1)
            label = zoom(label, (oh / h, ow / w), order=0)

        image = normalize_image(image)
        image = torch.from_numpy(np.ascontiguousarray(image)).float()
        label = torch.from_numpy(np.ascontiguousarray(label)).long()
        return {'image': image, 'label': label}


class OrchardTransferDataset(Dataset):
    """Dataset supporting paired TIFFs and the legacy NPZ format.

    Each split file contains either:
      image_path,label_path
    or, for NPZ:
      sample_path

    Relative paths are resolved against base_dir. TIFF georeferencing is not
    altered; the model consumes raster values only.
    """
    def __init__(self, base_dir, list_dir, split, transform=None):
        self.base_dir = base_dir
        self.transform = transform
        list_path = os.path.join(list_dir, split + '.txt')
        if not os.path.isfile(list_path):
            raise FileNotFoundError(f'Missing split file: {list_path}')
        with open(list_path, 'r', encoding='utf-8') as f:
            self.sample_list = [x.strip() for x in f if x.strip() and not x.lstrip().startswith('#')]

    def __len__(self):
        return len(self.sample_list)

    def _resolve(self, p):
        return p if os.path.isabs(p) else os.path.join(self.base_dir, p)

    def __getitem__(self, idx):
        fields = [x.strip() for x in self.sample_list[idx].split(',') if x.strip()]
        if len(fields) == 2:
            image_path = self._resolve(fields[0])
            label_path = self._resolve(fields[1])
            if not os.path.splitext(image_path)[1]:
                image_path += '.tif'
            if not os.path.splitext(label_path)[1]:
                label_path += '.tif'
            if not os.path.isfile(image_path):
                raise FileNotFoundError(f'Image not found: {image_path}')
            if not os.path.isfile(label_path):
                raise FileNotFoundError(f'Label not found: {label_path}')
            image = read_tif(image_path) if image_path.lower().endswith(('.tif', '.tiff')) else np.load(image_path)
            label = read_tif(label_path).squeeze() if label_path.lower().endswith(('.tif', '.tiff')) else np.load(label_path).squeeze()
            if image.ndim == 3 and label.ndim == 3:
                label = label[0]
            case_name = os.path.splitext(os.path.basename(image_path))[0]
        else:
            name = fields[0]
            path = self._resolve(name)
            if not os.path.splitext(path)[1]:
                path += '.npz'
            if not path.lower().endswith('.npz'):
                raise ValueError(f'Single-path entries must point to NPZ samples: {path}')
            data = np.load(path)
            if 'image' in data and 'label' in data:
                image, label = data['image'], data['label']
            elif 'data' in data and 'seg' in data:
                image, label = data['data'], data['seg']
            else:
                raise KeyError(f'{path}: expected image/label or data/seg arrays')
            case_name = os.path.splitext(os.path.basename(path))[0]

        if image.shape[-2:] != label.shape[-2:]:
            raise ValueError(f'Spatial mismatch: image {image.shape}, label {label.shape}')
        sample = {'image': image, 'label': label}
        if self.transform:
            sample = self.transform(sample)
        sample['case_name'] = case_name
        return sample
