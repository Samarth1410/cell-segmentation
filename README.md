# 🔬 CoNSeP Cell Segmentation with UNet

A UNet built from scratch in PyTorch to perform multi-class semantic segmentation of cell nuclei in H&E stained histopathology images. Trained on the CoNSeP dataset with Focal Loss + Dice Loss to handle severe class imbalance.

---

## 🎯 Overview

This project implements a full UNet architecture from scratch to segment cell nuclei in colorectal cancer tissue images. Each pixel is classified into one of 5 categories — background or one of 4 cell nucleus types. The key challenge is extreme class imbalance, where background pixels dominate at 83.6% and the rarest class (Yellow Cell) represents only 0.2% of all pixels.

Real-world applications include automated cancer grading, cell counting, and computer-aided pathology diagnosis.

---

## 📊 Results

| Metric | Value |
|--------|-------|
| Val Loss | 1.3023 |
| mIoU (excl. background) | 0.309 |
| Dice Score (excl. background) | 0.459 |
| mIoU (incl. background) | 0.410 |
| Dice Score (incl. background) | 0.546 |

### Per-Class Performance

| Class | IoU | Dice Score |
|-------|-----|------------|
| Background | 0.811 | 0.896 |
| Yellow Cell | 0.122 | 0.217 |
| Red Cell | 0.361 | 0.530 |
| Green Cell | 0.449 | 0.619 |
| Blue Cell | 0.306 | 0.469 |

**Note:** Background is excluded from mIoU and Dice Score — including it would inflate metrics since it covers 83.6% of all pixels. Green Cell scored highest as the most common cell type (8.6% of pixels). Yellow Cell improved from near-zero to IoU 0.122 thanks to Focal Loss which specifically focuses on hard, rare examples — a known challenge in medical image segmentation with severe class imbalance.

---

## 🗂️ Dataset — CoNSeP

- **Full name:** Colon Nuclei Identification and Counting
- **Images:** 164 total — 108 train, 56 test (512×512 px, RGB)
- **Masks:** RGB color-coded pixel-level annotations
- **Split:** 90% train / 10% validation (97 / 11 images)
- **Classes:** 5 (Background + 4 cell nucleus types)

### Class Distribution (severe imbalance)

| Class | Color | Pixels | % of Total |
|-------|-------|--------|------------|
| Background | Black | 23,674,656 | 83.6% |
| Green Cell | Green | 2,447,050 | 8.6% |
| Blue Cell | Blue | 1,389,151 | 4.9% |
| Red Cell | Red | 751,641 | 2.7% |
| Yellow Cell | Yellow | 49,054 | 0.2% |

---

## 🏗️ Model Architecture — UNet

UNet consists of three parts:

**Encoder (Contracting Path):** Four convolutional blocks progressively halve the spatial dimensions while doubling the channels — compressing the image into a rich feature representation.

```
Input (3, 512, 512) → 64 channels → 128 → 256 → 512 → Bottleneck (1024, 32, 32)
```

**Bottleneck:** The deepest layer — sees the most abstract, global representation of the entire image.

**Decoder (Expanding Path):** Upsamples back to original resolution using transposed convolutions. At each level, feature maps from the encoder are concatenated via **skip connections** — preserving fine spatial details that would otherwise be lost during downsampling.

```
(1024, 32, 32) → 512 → 256 → 128 → 64 → Output (5, 512, 512)
```

**Final layer:** 1×1 convolution maps 64 channels → 5 class scores per pixel.

| Component | Details |
|-----------|---------|
| Total parameters | 31,037,893 |
| Dropout | 0.1–0.3 (encoder/decoder) |
| Normalization | BatchNorm2d after every conv |
| Upsampling | Learnable ConvTranspose2d |

---

## 🚀 Training Details

| Parameter | Value |
|-----------|-------|
| Epochs | 200 (early stopping at 109) |
| Batch size | 8 |
| Learning rate | 3e-5 |
| Optimizer | Adam (weight decay 1e-4) |
| Scheduler | ReduceLROnPlateau (patience=7, factor=0.5) |
| Early stopping | 20 epochs patience |
| Training hardware | L4 GPU (24GB) |

### Loss Function — Focal Loss + Dice Loss

**Focal Loss** down-weights easy examples (background pixels already predicted correctly) and forces the model to focus on hard, rare examples like Yellow Cell nuclei. The focusing parameter γ=2 is the standard setting.

**Dice Loss** measures overlap between prediction and ground truth per class, treating each class equally regardless of pixel count — naturally handling the severe imbalance.

Combining both gives stable gradients from Focal Loss while ensuring minority classes are not ignored via Dice Loss.

### Data Augmentation (training only)

| Augmentation | Details |
|---|---|
| Horizontal flip | 50% probability |
| Vertical flip | 50% probability |
| Rotation | 0°, 90°, 180°, or 270° |
| Random crop | 70–100% of image, resized back to 512×512 |
| Brightness | ±20% random adjustment |
| Contrast | ±20% random adjustment |

**Important:** Spatial augmentations (flip, rotate, crop) are applied identically to both image and mask. Color augmentations are applied to image only — mask colors encode class labels and must not change.

---

## 📁 Project Structure

```
cell-segmentation/
│
├── notebooks/
│   ├── 00_dataset_exploration.ipynb   # EDA, class distribution, color scan
│   ├── 01_preprocessing.ipynb         # RGB→label conversion, augmentation, DataLoader
│   ├── 02_training.ipynb              # UNet, loss function, training loop
│   └── 03_evaluation.ipynb            # Per-class metrics, visualizations
│
├── src/
│   ├── dataset.py                     # Dataset and DataLoader
│   ├── model.py                       # UNet architecture
│   ├── metrics.py                     # mIoU and Dice Score
│   └── utils.py                       # Helper functions
│
├── results/
│   ├── sample_pairs.png               # Sample image-mask pairs
│   ├── class_distribution.png         # Pixel distribution per class
│   ├── augmentation_examples.png      # Augmentation visualization
│   ├── per_class_metrics.png          # IoU and Dice bar charts
│   ├── predictions.png                # Model predictions vs ground truth
│   └── training_curves.png            # Loss, mIoU, Dice over epochs
│
├── models/
│   └── unet_best.pth                  # Best model checkpoint
│
└── README.md
```

---

## ⚙️ How to Run Locally

**1. Clone the repository**
```bash
git clone https://github.com/Samarth1410/cell-segmentation.git
cd cell-segmentation
```

**2. Install dependencies**
```bash
pip install torch torchvision matplotlib scikit-learn pillow numpy
```

**3. Prepare dataset**

Download the CoNSeP dataset and place it as:
```
cell_dataset/
├── train/
│   ├── train_images/
│   └── train_masks/
└── test/
    └── test_images/
```

**4. Run notebooks in order**
```bash
jupyter notebook notebooks/00_dataset_exploration.ipynb
```

---

## 🔍 Key Observations

- **Dropout regularization** was essential — without it the model overfits quickly on only 97 training images
- **Focal Loss outperformed standard Cross Entropy** — the val loss dropped from 1.21 (CE) to 0.90 (Focal) at equivalent epochs
- **Green Cell scored highest (IoU 0.353)** — it has the most training pixels among cell classes, confirming that class frequency directly impacts segmentation quality
- **Yellow Cell remains nearly unpredicted (IoU 0.001)** — at 0.2% of pixels, even aggressive class weighting struggles without more training data or oversampling
- **Val loss consistently lower than train loss** — a healthy sign that Dropout2d is working as intended, adding noise during training but not evaluation
- **90/10 train-val split** was chosen over standard 80/20 to maximize training data on the small 108-image dataset

---

## 🛠️ Tech Stack

- **Python 3.12**
- **PyTorch** — UNet implementation, training loop, custom loss functions
- **torchvision** — data augmentation transforms
- **NumPy / Pillow** — image loading and mask conversion
- **scikit-learn** — train/val split
- **Matplotlib** — visualizations

---

## 📚 References

- [U-Net: Convolutional Networks for Biomedical Image Segmentation](https://arxiv.org/abs/1505.04597)
- [Focal Loss for Dense Object Detection](https://arxiv.org/abs/1708.02002)
- [CoNSeP Dataset — Graham et al. 2019](https://warwick.ac.uk/fac/cross_fac/tia/data/hovernet/)
- [Dice Loss for Medical Image Segmentation](https://arxiv.org/abs/1707.03237)

---

## 👤 Author

**Samarth Agrawal**
[GitHub](https://github.com/Samarth1410)

---

## 🎓 Acknowledgement

This project was developed as part of a course assignment under **Prof. Arambam James** at the **Yardi School of AI, IIT Delhi**.