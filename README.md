# Cartoon-to-Realistic Face Translation using CycleGAN

This project implements a CycleGAN-based model that converts cartoon faces into realistic human faces and vice versa. The model learns bidirectional mappings between two domains without requiring paired training data.

---

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Requirements](#requirements)
- [Installation](#installation)
- [Dataset Preparation](#dataset-preparation)
  - [Download Datasets](#download-datasets)
  - [Preprocessing Pipeline](#preprocessing-pipeline)
- [Project Structure](#project-structure)
- [Model Architecture](#model-architecture)
- [Training](#training)
- [Testing](#testing)
- [Pretrained Models](#pretrained-models)
- [Results](#results)
- [References](#references)

---

## Overview

This project uses CycleGAN (Cycle-Consistent Generative Adversarial Networks) to perform unpaired image-to-image translation between cartoon faces and realistic human faces. The model consists of two generators and two discriminators that work together to learn mappings in both directions while maintaining cycle consistency.

## Features

- **Unpaired Image Translation**: No need for paired cartoon-realistic face datasets
- **Bidirectional Mapping**: Convert cartoon → realistic and realistic → cartoon
- **Cycle Consistency**: Ensures that translating from one domain to another and back returns similar images
- **Identity Loss**: Preserves color composition when possible
- **Diversity Loss**: Optional diversity loss to prevent mode collapse
- **Mixed Precision Training**: Faster training with automatic mixed precision
- **Preprocessing Scripts**: Complete pipeline for cleaning and preparing CelebA dataset

---

## Requirements

```
torch>=1.9.0
torchvision>=0.10.0
numpy>=1.19.0
Pillow>=8.0.0
albumentations>=1.0.0
tqdm>=4.60.0
matplotlib>=3.3.0
facenet-pytorch>=2.5.0
opencv-python>=4.5.0
mediapipe>=0.8.0
```

## Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/Cartoon-to-Realistic-using-CycleGAN.git
   cd Cartoon-to-Realistic-using-CycleGAN
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Create necessary directories**
   ```bash
   mkdir -p cartoon2realistic/{trainA,trainB,testA,testB}
   mkdir -p pretrained
   mkdir -p saved_image/{cartoon,realistic}
   ```

---

## Dataset Preparation

### Download Datasets

#### 1. Cartoon10k Dataset
Download the cartoon face dataset from:
```
https://drive.google.com/drive/folders/1couYgzLUQQdFH2_jL7j9LYD53WXrJQnE?usp=sharing
```
Extract and organize:
- Training images → `cartoon2realistic/trainA/`
- Test images → `cartoon2realistic/testA/`

#### 2. CelebA Dataset (Raw)
Download the original CelebA dataset from:
```
https://drive.google.com/file/d/1KLzyqWXAzca1exX6WEYzv9TpFel0IaG7/view?usp=sharing
```
Extract all images into the `img_align_celeba/` folder.

### Preprocessing Pipeline

The CelebA dataset requires preprocessing to extract frontal faces with white backgrounds. Use the scripts in the `preprocessing_scripts/` folder in the following order:

#### Step 1: Face Detection and Alignment
```bash
python preprocessing_scripts/detec_faces.py
```
**What it does:**
- Detects faces in CelebA images using MTCNN (Multi-task Cascaded Convolutional Networks)
- Crops detected faces with bounding boxes
- Resizes faces to 256×256 pixels
- Adds white background to cropped faces
- Saves results to `faces_aligned_white_bg/` folder

**Input:** `img_align_celeba/` folder with raw CelebA images  
**Output:** `faces_aligned_white_bg/` folder with aligned faces

#### Step 2: Frontal Face Filtering
```bash
python preprocessing_scripts/filter_frontal_face.py
```
**What it does:**
- Uses MediaPipe Face Mesh to detect facial landmarks
- Filters only frontal-facing faces based on:
  - Distance between eyes
  - Nose position relative to eye midpoint
  - Filters out profile or tilted faces
- Copies frontal faces to output directory

**Input:** `faces_aligned_white_bg/` folder  
**Output:** `frontal_faces/` folder with only frontal-facing images

#### Step 3: Train/Test Split
```bash
python preprocessing_scripts/split.py
```
**What it does:**
- Randomly shuffles the frontal face dataset
- Splits into training (9,148 images) and testing (901 images) sets
- Copies images to appropriate directories

**Input:** `frontal_faces/` folder  
**Output:**
- `cartoon2realistic/trainB/` (9,148 training images)
- `cartoon2realistic/testB/` (901 test images)

**Complete Preprocessing Pipeline:**
```bash
# Run all preprocessing steps in order
python preprocessing_scripts/detec_faces.py
python preprocessing_scripts/filter_frontal_face.py
python preprocessing_scripts/split.py
```

---

## Project Structure

```plaintext
Cartoon-to-Realistic-using-CycleGAN/
├── preprocessing_scripts/
│   ├── detec_faces.py              # Face detection and alignment
│   ├── filter_frontal_face.py      # Filter frontal faces only
│   └── split.py                    # Train/test split
│
├── cartoon2realistic/
│   ├── trainA/                     # Cartoon training images
│   ├── trainB/                     # Realistic training images
│   ├── testA/                      # Cartoon test images
│   └── testB/                      # Realistic test images
│
├── pretrained/                     # Pretrained model checkpoints
│   ├── genh.pth                    # Generator: Cartoon → Realistic
│   ├── genz.pth                    # Generator: Realistic → Cartoon
│   ├── critich.pth                 # Discriminator for realistic images
│   └── criticz.pth                 # Discriminator for cartoon images
│
├── saved_image/                    # Generated images during training
│   ├── cartoon/                    # Generated cartoon samples
│   └── realistic/                  # Generated realistic samples
│
├── generator.py                    # Generator network architecture
├── discriminator.py                # Discriminator network architecture
├── train.py                        # Standard training script
├── train_with_diversity_loss.py   # Training with diversity loss
├── test_realistic_to_cartoon.py   # Testing/inference script
├── README.md                       # This file
└── .gitignore                      # Git ignore file
```

---

## Model Architecture

### Generator Network
The generator uses a ResNet-based architecture with:
- **Encoder**: 2 downsampling convolutional layers (64→128→256 channels)
- **Transformer**: 9 residual blocks with instance normalization and dropout
- **Decoder**: 2 upsampling transpose convolutional layers (256→128→64→3 channels)
- **Activation**: ReLU for hidden layers, Tanh for output
- **Normalization**: Instance normalization for better style transfer

**Key Components:**
- Reflection padding to reduce border artifacts
- Skip connections in residual blocks for better gradient flow
- Dropout (0.5) in residual blocks to prevent overfitting

### Discriminator Network
PatchGAN discriminator with:
- 5 convolutional layers with stride-2 downsampling
- LeakyReLU (0.2) activation
- Instance normalization
- No fully connected layers (fully convolutional)
- Output shape: Single value per patch

**Additional Components:**
- **ImageBuffer**: Stores generated images for stable discriminator training
- **TensorBuffer**: Circular buffer for diversity loss computation

---

## Training

### Configuration

Key hyperparameters in `train.py`:

```python
DEVICE = "cuda" if available else "cpu"
BATCH_SIZE = 1
LEARNING_RATE = 0.00001
LAMBDA_IDENTITY = 0.5      # Weight for identity loss
LAMBDA_CYCLE = 10          # Weight for cycle consistency loss
LAMBDA_ADVERSARIAL = 1     # Weight for adversarial loss
NUM_EPOCHS = 200
IMAGE_SIZE = 128×128
```

### Loss Functions

1. **Adversarial Loss** (MSE): Ensures generated images fool the discriminator
2. **Cycle Consistency Loss** (L1): Ensures F(G(x)) ≈ x
3. **Identity Loss** (L1): Preserves color composition
4. **Diversity Loss** (Optional): Prevents mode collapse by encouraging variation

**Total Generator Loss:**
```
L_total = λ_adv × L_adv + λ_cycle × L_cycle + λ_identity × L_identity
```

### Training Commands

**Standard Training:**
```bash
python train.py
```

**Training with Diversity Loss:**
```bash
python train_with_diversity_loss.py
```

**Features:**
- Automatic mixed precision (AMP) for faster training
- Gradient scaling with `torch.cuda.amp.GradScaler`
- Progress bars with `tqdm`
- Periodic image saving every 200 iterations
- Model checkpointing after each epoch
- Supports loading from pretrained checkpoints

### Training Tips

1. **GPU Memory**: Reduce batch size if running out of memory
2. **Learning Rate**: Start with 0.0001-0.00001 for stable training
3. **Image Size**: 128×128 is recommended; larger sizes require more memory
4. **Epochs**: Train for at least 100-200 epochs for good results
5. **Monitoring**: Check `saved_image/` folder during training

---

## Testing

### Running Inference

Use `test_realistic_to_cartoon.py` to test the trained model:

```bash
python test_realistic_to_cartoon.py
```

**What it does:**
- Loads pretrained generator models
- Processes test images from both domains
- Generates translations in both directions
- Computes cycle-consistency reconstruction
- Visualizes results with matplotlib

**Output includes:**
- Original image from domain A
- Translated image to domain B
- Cycle-reconstructed image back to domain A

### Custom Image Testing

To test on your own images:

1. Place images in appropriate test directories:
   - Cartoon faces → `cartoon2realistic/testA/`
   - Realistic faces → `cartoon2realistic/testB/`

2. Update paths in `test_realistic_to_cartoon.py` if needed

3. Run the script:
   ```bash
   python test_realistic_to_cartoon.py
   ```

---

## Pretrained Models

Download pretrained models and place them in the `pretrained/` folder:

| Model | Description | Download Link |
|-------|-------------|---------------|
| **genh.pth** | Generator: Cartoon → Realistic | [Download](https://drive.google.com/file/d/1i_xxYf_u2ze4_S-gDaRswUfj4VboVAln/view?usp=sharing) |
| **genz.pth** | Generator: Realistic → Cartoon | [Download](https://drive.google.com/file/d/15Zc3rNdNS04QuZzULf-c7LA5qOeNKs5T/view?usp=sharing) |
| **critich.pth** | Discriminator for realistic images | [Download](https://drive.google.com/file/d/1uVg-cSdDPhQ1tJzXSO8ARG3zaeYPzyAZ/view?usp=sharing) |
| **criticz.pth** | Discriminator for cartoon images | [Download](https://drive.google.com/file/d/1vqp2pWmyz7HGdpCa0PbFi6VJ7C2yWVCx/view?usp=sharing) |

### Using Pretrained Models

Set `LOAD_MODEL = True` in `train.py` or `test_realistic_to_cartoon.py` to load pretrained weights.

---

## Results

The model achieves:
- **High-quality translations** from cartoon to realistic faces
- **Preservation of identity features** (hairstyle, facial structure, etc.)
- **Natural-looking skin textures** in realistic domain
- **Consistent cartoon styles** when translating to cartoon domain
- **Cycle consistency** with minimal reconstruction error

Sample results can be found in the `saved_image/` directory during training.

---

## References

1. **CycleGAN Paper**: [Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks](https://arxiv.org/abs/1703.10593)  
   Zhu, J. Y., Park, T., Isola, P., & Efros, A. A. (2017)

2. **CelebA Dataset**: [Large-scale CelebFaces Attributes Dataset](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html)  
   Liu, Z., Luo, P., Wang, X., & Tang, X. (2015)

3. **Cartoon10k Dataset**: Custom cartoon face dataset

4. **MTCNN**: [Joint Face Detection and Alignment using Multi-task Cascaded Convolutional Networks](https://arxiv.org/abs/1604.02878)  
   Zhang, K., Zhang, Z., Li, Z., & Qiao, Y. (2016)

5. **MediaPipe**: [MediaPipe Face Mesh](https://google.github.io/mediapipe/solutions/face_mesh.html)  
   Google Research

---

## License

This project is provided for educational and research purposes.

## Acknowledgments

- CycleGAN implementation inspired by the original PyTorch implementation
- Thanks to the creators of CelebA and Cartoon10k datasets
- Face detection powered by MTCNN and MediaPipe

---

## Contact

For questions or issues, please open an issue on GitHub.

---

**Happy Training! 🎨→🖼️**



