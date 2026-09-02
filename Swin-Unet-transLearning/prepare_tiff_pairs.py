"""Create deterministic train/val(/test) split files for paired TIFF datasets.

Expected directory layout:
    image_dir/*.tif
    label_dir/*.tif

Images and labels are matched by filename stem. The generated split files contain:
    image_path,label_path

Paths are written relative to --root_path when possible.
"""
import argparse
import os
import random


def collect_tifs(directory):
    if not os.path.isdir(directory):
        raise FileNotFoundError(directory)
    return {os.path.splitext(f)[0]: os.path.join(directory, f)
            for f in os.listdir(directory)
            if f.lower().endswith(('.tif', '.tiff'))}


def write_split(path, pairs):
    with open(path, 'w', encoding='utf-8') as f:
        for image_path, label_path in pairs:
            f.write(f'{image_path},{label_path}\n')


def main():
    p = argparse.ArgumentParser(description='Prepare paired TIFF train/val/test lists.')
    p.add_argument('--root_path', required=True, help='Common root used to store relative paths in split files.')
    p.add_argument('--image_dir', required=True, help='Directory containing image TIFFs.')
    p.add_argument('--label_dir', required=True, help='Directory containing label TIFFs.')
    p.add_argument('--list_dir', required=True, help='Directory where train.txt/val.txt/test.txt are written.')
    p.add_argument('--train_ratio', type=float, default=0.70)
    p.add_argument('--val_ratio', type=float, default=0.15)
    p.add_argument('--seed', type=int, default=42)
    args = p.parse_args()

    if not 0 < args.train_ratio < 1 or not 0 <= args.val_ratio < 1 or args.train_ratio + args.val_ratio >= 1:
        raise ValueError('Require 0 < train_ratio < 1, 0 <= val_ratio < 1, and train_ratio + val_ratio < 1.')

    images = collect_tifs(args.image_dir)
    labels = collect_tifs(args.label_dir)
    common = sorted(set(images) & set(labels))
    missing_labels = sorted(set(images) - set(labels))
    missing_images = sorted(set(labels) - set(images))
    if missing_labels or missing_images:
        raise RuntimeError(f'Unmatched TIFF stems: missing_labels={missing_labels[:10]}, missing_images={missing_images[:10]}')
    if not common:
        raise RuntimeError('No paired TIFFs found.')

    root = os.path.abspath(args.root_path)
    pairs = []
    for stem in common:
        ip = os.path.abspath(images[stem])
        lp = os.path.abspath(labels[stem])
        try:
            ip = os.path.relpath(ip, root)
            lp = os.path.relpath(lp, root)
        except ValueError:
            pass
        pairs.append((ip, lp))

    rng = random.Random(args.seed)
    rng.shuffle(pairs)
    n = len(pairs)
    n_train = int(n * args.train_ratio)
    n_val = int(n * args.val_ratio)
    if n_train == 0 or n_val == 0 or n_train + n_val >= n:
        raise RuntimeError(f'Too few samples for the requested split: n={n}')
    train = pairs[:n_train]
    val = pairs[n_train:n_train + n_val]
    test = pairs[n_train + n_val:]

    os.makedirs(args.list_dir, exist_ok=True)
    write_split(os.path.join(args.list_dir, 'train.txt'), train)
    write_split(os.path.join(args.list_dir, 'val.txt'), val)
    write_split(os.path.join(args.list_dir, 'test.txt'), test)
    print(f'Paired TIFFs: {n}; train={len(train)}, val={len(val)}, test={len(test)}; seed={args.seed}')


if __name__ == '__main__':
    main()
