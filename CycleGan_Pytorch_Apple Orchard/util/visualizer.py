import os
import ntpath
import numpy as np
import rasterio
from . import util

global_meta = None
global_affine = None
global_crs = None
global_ori_height = None
global_ori_width = None
global_ori_count = None


def set_geo_info(meta, affine, crs, height, width, count):
    """Store the source GeoTIFF metadata for the current inference image."""
    global global_meta, global_affine, global_crs, global_ori_height, global_ori_width, global_ori_count
    global_meta = meta.copy()
    global_affine = affine
    global_crs = crs
    global_ori_height = height
    global_ori_width = width
    global_ori_count = count


def save_images(webpage, visuals, image_path, aspect_ratio=1.0, width=256):
    image_dir = webpage.get_image_dir()
    short_path = ntpath.basename(image_path[0])
    name = os.path.splitext(short_path)[0]
    webpage.add_header(name)
    ims, txts, links = [], [], []

    for label, im_data in visuals.items():
        im_numpy = util.tensor2im(im_data)
        im_numpy = np.clip(im_numpy, 0, 255).astype(np.uint8)
        if im_numpy.ndim == 2:
            im_final = im_numpy[None, ...]
        elif im_numpy.ndim == 3:
            im_final = np.transpose(im_numpy, (2, 0, 1))
        else:
            raise ValueError(f'Unexpected generated image shape: {im_numpy.shape}')

        # Never use np.resize here: it can repeat/truncate pixels and corrupt
        # spatial structure. CycleGAN inference must produce the same patch size.
        if global_ori_height is not None and global_ori_width is not None:
            if im_final.shape[1:] != (global_ori_height, global_ori_width):
                raise ValueError(
                    f'Generated image size {im_final.shape[1:]} does not match source '
                    f'GeoTIFF size {(global_ori_height, global_ori_width)}. '
                    'Use crop_size/load_size=256 for 256x256 training patches.'
                )
            if global_ori_count is not None and im_final.shape[0] != global_ori_count:
                raise ValueError(
                    f'Generated channel count {im_final.shape[0]} does not match source '
                    f'GeoTIFF channel count {global_ori_count}.'
                )

        save_name = f"{name}_{label}.tif"
        save_path = os.path.join(image_dir, save_name)

        if global_meta is not None:
            out_meta = global_meta.copy()
            out_meta.update({
                'driver': 'GTiff',
                'dtype': im_final.dtype,
                'height': im_final.shape[1],
                'width': im_final.shape[2],
                'count': im_final.shape[0],
                'transform': global_affine,
                'crs': global_crs
            })
            with rasterio.open(save_path, 'w', **out_meta) as dst:
                dst.write(im_final)
        else:
            with rasterio.open(
                save_path, 'w', driver='GTiff',
                height=im_final.shape[1], width=im_final.shape[2],
                count=im_final.shape[0], dtype=im_final.dtype
            ) as dst:
                dst.write(im_final)

        ims.append(save_name)
        txts.append(label)
        links.append(save_name)
    webpage.add_images(ims, txts, links, width=width)
