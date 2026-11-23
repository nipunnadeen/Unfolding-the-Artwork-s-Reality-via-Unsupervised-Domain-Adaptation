# Unfolding-the-Artwork-s-Reality-via-Unsupervised-Domain-Adaptation

### Enhancing Art-to-Real Image Translation Using Domain Adaptation

## 1. Project Overview

This project explores the application of **Unsupervised Domain Adaptation (UDA)** to improve the quality and realism of artwork-to-photograph image translation.

Traditional computer vision models are primarily trained using natural photographs. However, artistic images such as paintings have significantly different visual characteristics, including textures, colors, shapes, and artistic representations. This difference creates a domain gap between artistic images and real-world photographs.

This research extends the Art2Real image-to-image translation framework by incorporating domain adaptation techniques to reduce the visual domain gap and improve the generation of realistic images from artistic inputs.

The implementation builds upon CycleGAN and introduces additional feature-level adversarial adaptation and pixel-level adaptation components.

## 2. Research Objectives

The primary objectives of this project are:

- Investigate the challenges associated with translating artistic images into realistic photographs.
- Explore Unsupervised Domain Adaptation techniques for reducing the domain gap between artistic and realistic image distributions.
- Extend the existing Art2Real framework with additional domain adaptation components.
- Improve the learned feature representations across source and target domains.
- Generate realistic visual representations while preserving the semantic content of the original artwork.

## 3. Proposed Methodology

The proposed approach combines image-to-image translation with domain adaptation.

### 3.1 Art2Real Framework

The project uses the Art2Real framework, which builds upon CycleGAN to perform unpaired image-to-image translation.

The framework employs:

- Generator networks to translate images between artistic and realistic domains.
- Discriminator networks to distinguish generated images from real images.
- Cycle-consistency loss to preserve the underlying content during translation.
- Identity loss to encourage content preservation.
- Contextual patch-based learning to improve the realism of generated image details.

### 3.2 Feature-Level Domain Adaptation

Feature-level adaptation is introduced to reduce the discrepancy between the representations learned from source and target domains.

The implementation includes:

- A feature extraction network.
- A domain discriminator.
- Adversarial learning to encourage domain-invariant feature representations.
- Supervised feature-learning objectives within the adaptation implementation.

The objective is to improve the compatibility of learned representations across different visual domains.

### 3.3 Pixel-Level Adaptation

The project also includes a pixel-level adaptation module intended to explore domain adaptation at the image representation level.

This module provides an additional adaptation approach alongside the feature-level implementation.

### 3.4 Contextual Learning

The implementation retains contextual patch-based learning from Art2Real.

This component uses patch similarity and nearest-neighbor retrieval to encourage generated image regions to resemble real-world image details.

Multiple patch scales and configurable retrieval parameters are supported through the training options.

## 4. System Architecture

The overall research implementation consists of the following components:

| Component | Responsibility |
|---|---|
| CycleGAN | Unpaired image-to-image translation |
| Generators | Translate images between artistic and realistic domains |
| Image Discriminators | Evaluate generated image realism |
| Feature Extractor | Learn visual representations |
| Feature Discriminator | Support adversarial domain adaptation |
| Pixel Adaptation | Explore pixel-level domain adaptation |
| Contextual Loss | Encourage realistic local image details |
| Data Loader | Prepare source and target domain samples |

## 5. Project Structure

```text
Unfolding-the-Artwork-s-Reality-via-Unsupervised-Domain-Adaptation/
│
├── art2real/
│   ├── data/
│   │
│   ├── models/
│   │   ├── cycle_gan_model.py
│   │   ├── feature_loss.py
│   │   ├── fcn.py
│   │   ├── pixel_adapt.py
│   │   ├── contextual_loss.py
│   │   └── networks.py
│   │
│   ├── options/
│   │   ├── base_options.py
│   │   ├── train_options.py
│   │   └── test_options.py
│   │
│   ├── util/
│   │
│   ├── train.py
│   └── test.py
│
└── README.md
```

*Note: This is a simplified overview of the main project components.*

## 6. Technologies and Dependencies

The project is implemented using Python and deep learning technologies.

**Core technologies:**

- Python
- PyTorch
- Torchvision
- Generative Adversarial Networks (GANs)
- CycleGAN
- Unsupervised Domain Adaptation
- FAISS for similarity search and patch retrieval

**Required Python packages:**

```text
torch>=0.4.1
torchvision>=0.2.1
dominate>=2.3.1
visdom>=0.1.8.3
faiss
numpy
Pillow
```

The original dependency requirements are inherited from the Art2Real implementation. Actual compatibility should be checked against the Python and PyTorch versions used for the research environment.

## 7. Installation

Clone the repository:

```bash
git clone https://github.com/nipunnadeen/Unfolding-the-Artwork-s-Reality-via-Unsupervised-Domain-Adaptation.git
```

Navigate to the project directory:

```bash
cd Unfolding-the-Artwork-s-Reality-via-Unsupervised-Domain-Adaptation/art2real
```

Install the dependencies:

```bash
pip install torch torchvision dominate visdom faiss-cpu numpy Pillow
```

For GPU-based execution, install a compatible CUDA-enabled PyTorch and FAISS configuration.

## 8. Dataset Preparation

The project works with two visual domains:

- **Source Domain:** Artistic images and paintings.
- **Target Domain:** Real-world photographs.

For the CycleGAN data pipeline, datasets should follow the unaligned image-to-image translation structure:

```text
datasets/
│
└── artwork2photo/
    ├── trainA/
    ├── trainB/
    ├── testA/
    └── testB/
```

Where:

- `trainA`: Source-domain training images.
- `trainB`: Target-domain training images.
- `testA`: Source-domain testing images.
- `testB`: Target-domain testing images.

The feature-adaptation data loader also expects preprocessed source and target tensors with their associated labels. Its current implementation uses specific Google Colab directory paths, which must be configured for the intended execution environment.

## 9. Model Training

The training pipeline is implemented in `train.py`.

It initializes the dataset, translation model, feature extraction network, adaptation data loader, and training visualizer.

The training process incorporates the image translation objectives and additional feature-level adaptation loss.

The main training configuration is defined in:

```text
options/train_options.py
```

Important configurable parameters include:

| Parameter | Description |
|---|---|
| `--lr` | Generator and discriminator learning rate |
| `--niter` | Number of initial training epochs |
| `--niter_decay` | Learning-rate decay period |
| `--lambda_A` | Forward cycle-consistency weight |
| `--lambda_B` | Backward cycle-consistency weight |
| `--lambda_identity` | Identity mapping loss weight |
| `--patch_size_1` | First contextual patch scale |
| `--patch_size_2` | Second contextual patch scale |
| `--patch_size_3` | Third contextual patch scale |
| `--contextual_weight` | Contextual loss contribution |
| `--k` | Number of nearest-neighbor patches |
| `--which_mem_bank` | Real-image patch memory bank directory |

The training entry point is:

```bash
python train.py
```

Training requires the appropriate dataset, pretrained resources where applicable, contextual memory-bank data, and a compatible configuration.

## 10. Image Translation and Testing

The testing pipeline is implemented in `test.py`.

It loads the trained model, processes source-domain images, performs image translation, and saves generated results into an HTML-based experiment results directory.

The output directory follows this structure:

```text
results/
└── experiment_name/
    └── test_latest/
        ├── images
        └── index.html
```

The testing script supports configurable experiment names, checkpoints, dataset locations, and the number of images to process through its testing options.

## 11. Research Contributions

This project investigates the integration of domain adaptation techniques into an existing artistic image-to-image translation framework.

The main research directions include:

1. Extending the original Art2Real framework with feature-level adversarial domain adaptation.
2. Exploring pixel-level adaptation as an additional domain adaptation approach.
3. Combining contextual patch-based learning with domain adaptation objectives.
4. Investigating the reduction of the visual domain gap between artistic representations and realistic photographs.

## 12. Original Research and References

This implementation builds upon the following research:

**Art2Real: Unfolding the Reality of Artworks via Semantically-Aware Image-to-Image Translation**

Authors: Matteo Tomei, Marcella Cornia, Lorenzo Baraldi, and Rita Cucchiara.

Published at: IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2019.

- [Original Research Paper](https://arxiv.org/abs/1811.10666)
- [Original Art2Real Repository](https://github.com/aimagelab/art2real)

**CycleGAN: Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks**

- [Research Paper](https://arxiv.org/abs/1703.10593)

## 13. Acknowledgements

This project is based on the original Art2Real research implementation developed by the AImageLab research group.

The original authors' work provided the foundation for exploring additional domain adaptation techniques in artwork-to-realistic-image translation.

---

**Research Project:** Unfolding the Artwork's Reality via Unsupervised Domain Adaptation

**Implementation:** Python / PyTorch

**Research Area:** Computer Vision, Generative Adversarial Networks, Image-to-Image Translation, and Unsupervised Domain Adaptation.

