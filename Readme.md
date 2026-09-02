# Apple Orchard Mapping

This repository contains the experimental implementations used for apple-orchard semantic segmentation and cross-domain transfer learning.

## 1. Proposed Trans_GAN framework

The proposed framework is organized as two sequential components:

1. **CycleGAN cross-domain image translation**: JL1KF01A PMS06-style RGB imagery is translated into synthetic Sentinel-2 MSI-style RGB imagery while the original orchard labels are retained.
2. **Swin-Unet transfer learning**: the segmentation network is first pretrained on the CycleGAN-generated source-domain images and then progressively fine-tuned on real Sentinel-2 MSI imagery.

The complete training logic is implemented in the `CycleGan_Pytorch_Apple Orchard` and `Swin-Unet-transLearning` directories.

## 2. Repository structure

```text
Apple-Orchard_Mapping/
├── CycleGan_Pytorch_Apple Orchard/
│   ├── datasets/
│   ├── data/
│   ├── models/
│   ├── options/
│   ├── networks.py
│   ├── train.py
│   └── generate_tif.py
├── Swin-Unet-transLearning/
│   ├── configs/apple_transfer.yaml
│   ├── datasets/dataset_transfer.py
│   ├── config.py
│   ├── train_source.py
│   ├── train_transfer.py
│   ├── trainer_transfer.py
│   └── vision_transformer.py
├── 1.docx
└── Readme.md
```

## 3. TIFF data requirements

The transfer-learning pipeline uses paired GeoTIFF image/label files. For the CycleGAN image-translation stage, the first three bands of the input imagery must be **8-bit RGB (`uint8`) TIFF**. The inference script preserves the input CRS, affine transform, width and height and does not perform spatial resizing.

A typical source-domain organization is:

```text
source_data/
├── images/                 # CycleGAN-generated Sentinel-2-style TIFFs
├── labels/                 # original orchard labels, paired by filename
└── train.txt               # image_path,label_path
```

A typical target-domain organization is:

```text
target_data/
├── images/
├── labels/
└── train.txt
```

Each list entry is a comma-separated pair:

```text
images/image_0001.tif,labels/image_0001.tif
```

The image and label must have the same spatial dimensions.

## 4. CycleGAN training

The CycleGAN implementation uses a ResNet-9-block generator, PatchGAN (`basic`) discriminator, LSGAN objective, batch size 16, learning rate `0.0002`, `beta1=0.5`, `lambda_A=10`, `lambda_B=10`, and identity-loss weight `0.5`. The default training schedule is 50 epochs plus 50 decay epochs.

Run from the CycleGAN directory:

```bash
cd "CycleGan_Pytorch_Apple Orchard"
python train.py --dataroot ./datasets/demo --name orchard_mapping_cyclegan --model cycle_gan --netG resnet_9blocks --n_epochs 50 --n_epochs_decay 50 --batch_size 16 --lr 0.0002 --lambda_A 10.0 --lambda_B 10.0 --lambda_identity 0.5 --display_id 1
```

For the manuscript workflow, replace `datasets/demo` with the actual source and target image directories used in the experiment.

## 5. Generate synthetic Sentinel-2-style GeoTIFFs

After CycleGAN training, use `generate_tif.py` to translate RGB TIFFs while preserving geospatial metadata:

```bash
cd "CycleGan_Pytorch_Apple Orchard"
python generate_tif.py --input_dir /path/to/jilin_tif --output_dir /path/to/cyclegan_s2_tif --checkpoint /path/to/G_A_checkpoint.pth --netG resnet_9blocks --ngf 64 --norm instance
```

The script reads `.tif`/`.tiff` files, requires the first three bands to be `uint8`, and writes three-band `uint8` GeoTIFF outputs with the source CRS and affine transform retained.

## 6. Phase 1: source-domain pretraining

Phase 1 trains Swin-Unet on the CycleGAN-generated source-domain images and their original orchard labels. The source-domain loss is cross-entropy only, consistent with the manuscript description.

Example:

```bash
cd Swin-Unet-transLearning
python train_source.py --root_path /path/to/source_data --list_dir /path/to/source_data --output_dir /path/to/source_output --epochs 60 --batch_size 16 --base_lr 1e-4 --min_lr 1e-6 --weight_decay 1e-4 --img_size 256 --num_classes 2
```

The best source-domain model is saved as:

```text
source_output/source_best.pth
```

This checkpoint is used to initialize Phase 2.

## 7. Phase 2: target-domain progressive fine-tuning

Target-domain fine-tuning uses three stages:

| Stage | Trainable parameters | Maximum duration |
|---|---|---:|
| Stage I | decoder only | 20 epochs |
| Stage II | decoder + deepest encoder stage | 20 epochs |
| Stage III | decoder + deepest two encoder stages | 20 epochs |

The shallow encoder layers remain frozen. The semantic loss is:

```text
L_sem = 0.5 L_CE + 0.5 L_Dice
```

and the total target-domain objective is:

```text
L_total = 1.0 L_sem + 0.05 L_perc
```

where `L_perc` is the hierarchical feature-consistency loss implemented in `trainer_transfer.py`.

The optimizer is AdamW with `beta1=0.9`, `beta2=0.999`, and weight decay `1e-4`. The default batch size is 16 and the nominal learning-rate range is `1e-4` to `1e-6`.

### Semantic-loss-driven stage decision

The stage-decision block monitors the epoch-averaged semantic loss. A sliding window of 3 epochs is used. A transition criterion is satisfied when the range of semantic loss within the window is no greater than `0.005` for 3 consecutive qualifying checks. Each stage is additionally capped at 20 epochs, giving a maximum of 60 fine-tuning epochs.

Example:

```bash
python train_transfer.py --root_path /path/to/target_data --list_dir /path/to/target_data --output_dir /path/to/target_output --source_ckpt /path/to/source_output/source_best.pth --epochs 60 --batch_size 16 --base_lr 1e-4 --min_lr 1e-6 --weight_decay 1e-4 --max_epochs_per_stage 20 --sliding_window 3 --tolerance 0.005 --consecutive 3 --sem_weight 1.0 --perc_weight 0.05 --img_size 256 --num_classes 2
```

Outputs include:

```text
target_output/
├── best.pth
├── last.pth
├── history.csv
└── transfer.log
```

## 8. Important reproducibility note

The numerical results reported in the manuscript should be generated from the actual training runs. This repository documents the training configuration and implementation; it does not substitute hypothetical accuracy, IoU, F1, or ablation values for experimentally obtained results.

## 9. Comparative baseline models

The repository also references the following baseline implementations used for comparison:

- U-Net: `bigmb/Unet-Pytorch`
- DeepLabV3+ and PSPNet: `Tramac/awesome-semantic-segmentation-pytorch`
- SegNet: `alexgkendall/SegNet-Tutorial`
- Swin-Unet: `HuCaoFighting/Swin-Unet`

## 10. Implementation-to-manuscript mapping

| Manuscript component | Repository implementation |
|---|---|
| CycleGAN style translation | `CycleGan_Pytorch_Apple Orchard/` |
| GeoTIFF inference and metadata preservation | `CycleGan_Pytorch_Apple Orchard/generate_tif.py` |
| Source-domain pretraining | `Swin-Unet-transLearning/train_source.py` |
| Target-domain fine-tuning | `Swin-Unet-transLearning/train_transfer.py` |
| Progressive unfreezing and stage decision | `Swin-Unet-transLearning/trainer_transfer.py` |
| Loss weights and hyperparameters | `Swin-Unet-transLearning/config.py` and `configs/apple_transfer.yaml` |
| TIFF paired dataset loading | `Swin-Unet-transLearning/datasets/dataset_transfer.py` |
