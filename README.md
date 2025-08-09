# Cartoon-to-Realistic Face Translation

This project converts cartoon faces into realistic human faces using a cycleGAN.

---

## Project Setup and Usage

### 1. Download Datasets and Pretrained Model

- **Cartoon10k Dataset**  
  Download from:  
  https://drive.google.com/drive/folders/1couYgzLUQQdFH2_jL7j9LYD53WXrJQnE?usp=sharing  
  Extract and place contents into the `cartoonset10k/` folder.

- **Original CelebA Dataset (Raw)**  
  Download from:  
  https://drive.google.com/file/d/1KLzyqWXAzca1exX6WEYzv9TpFel0IaG7/view?usp=sharing  
  Extract and place images into the `img_align_celeba/` folder.


- **Pretrained Realistic Models**
  
  criticz.pth
  https://drive.google.com/file/d/1vqp2pWmyz7HGdpCa0PbFi6VJ7C2yWVCx/view?usp=sharing

  critich.pth
  https://drive.google.com/file/d/1uVg-cSdDPhQ1tJzXSO8ARG3zaeYPzyAZ/view?usp=sharing

  genh.pth
  https://drive.google.com/file/d/1i_xxYf_u2ze4_S-gDaRswUfj4VboVAln/view?usp=sharing

  genz.pth
  https://drive.google.com/file/d/15Zc3rNdNS04QuZzULf-c7LA5qOeNKs5T/view?usp=sharing
---

To clean the celeb dataset use the scripts in the preprocessing_scripts folder
## Project Structure

```plaintext
project/
├── cartoon2realistic/
│   ├── testA/        # Test images from domain A (e.g., cartoons)
│   ├── testB/        # Test images from domain B (e.g., real images)
│   ├── trainA/       # Training images from domain A
│   ├── trainB/       # Training images from domain B
│
├── discriminator.py  # Discriminator network definition
├── generator.py      # Generator network definition
├── train.py          # Training script



