**Model Implementation**

The experimental framework consists of two parts: baseline comparisons and our proposed method.



**1. Comparative Baseline Models**



The following open-source implementations were used as benchmarks. We thank the authors for their contributions to the community:



U-Net: bigmb/Unet-Pytorch



DeepLabV3+ \& PSPNet: Tramac/awesome-semantic-segmentation-pytorch



SegNet: alexgkendall/SegNet-Tutorial



Swin-Unet: HuCaoFighting/Swin-Unet



**2. Our Proposed Method**



The core components developed in this study are organized into the following modules in the root directory:



CycleGan\_Pytorch\_Apple Orchard: Optimized for cross-sensor (JL1KF01A PMS06 image to Sentinel-2 MSI image) domain adaptation.

## Dataset
This repository contains the source code for apple orchard mapping. The sample dataset used for reproducibility is available in the Release `v1.0`:
> This release contains 220 pairs of JL1KF01A PMS06 image patches and corresponding orchard label patches, for the reproducibility and validation of the apple orchard mapping framework proposed in this study.
>
> The dataset includes:
> - JL1KF01A PMS06 image tiles
> - Binary orchard annotation masks
>
> Note: The full original JL1KF01A imagery is commercial data and cannot be redistributed. This subset is provided for validation purposes only.
> Sentinel-2 MSI data can be accessed from the ESA Copernicus Data Space Ecosystem.

Download link: https://github.com/Public-sys/Apple-Orchard_Mapping/releases/tag/v1.0




Swin-Unet-transLearning: The fine-tuned architecture and transfer learning scripts tailored for orchard mapping.





