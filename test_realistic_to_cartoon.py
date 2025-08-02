import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from generator import Generator
from torch.utils.data import Dataset,DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
import os
from tqdm import tqdm
import numpy as np
from PIL import Image
from torchvision.utils import save_image

device="cuda" if torch.cuda.is_available() else "cpu"



transforms = A.Compose(
        [
            A.Resize(width=128, height=128),
            A.HorizontalFlip(p=0.5),
            A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], max_pixel_value=255),
            ToTensorV2(),
        ],
        additional_targets = {"image0": "image"},
)


class HorseZebraDataset(Dataset):
    def __init__(self,root_zebra,root_horse,transform = None):
        self.root_zebra = root_zebra
        self.root_horse = root_horse
        self.transform = transform

        self.zebra_images = os.listdir(root_zebra)
        self.horse_images = os.listdir(root_horse)
        self.len_set = max(len(self.zebra_images),len(self.horse_images))
        self.zebra_length, self.horse_length = len(self.zebra_images),len(self.horse_images)
    def __len__(self):
        return self.len_set
    def __getitem__(self, index):
        zebra_path= os.path.join(self.root_zebra,self.zebra_images[index%self.zebra_length])
        horse_path = os.path.join(self.root_horse,self.horse_images[index%self.horse_length])
        zebra_img = np.array(Image.open(zebra_path).convert('RGB'))
        horse_img = np.array(Image.open(horse_path).convert('RGB'))
        if self.transform:
            augmentations = self.transform(image = zebra_img, image0 = horse_img)
            zebra_img = augmentations['image']
            horse_img = augmentations['image0']
        return zebra_img,horse_img


gen_H = Generator().to(device)
gen_Z = Generator().to(device)

checkpoint_h = torch.load('pretrained/genh_2.pth.tar', map_location=device)
checkpoint_z = torch.load('pretrained/genz_2.pth.tar', map_location=device)
print('__loaded the model__')

gen_H.load_state_dict(checkpoint_h['model_state'])
gen_Z.load_state_dict(checkpoint_z['model_state'])


val_dataset = HorseZebraDataset(
    root_horse='C:\\Users\\Apaar\\PycharmProjects\\Cartoon_to_realistic\\.venv\\horse2zebra\\testA',
    root_zebra='C:\\Users\\Apaar\\PycharmProjects\\Cartoon_to_realistic\\.venv\\horse2zebra\\testB',
    transform=transforms,
)

val_loader = DataLoader(
    val_dataset,
    batch_size=1,
    shuffle=True,
    pin_memory=True,
)

loop = tqdm(val_loader, leave=True)
for idx, (zebra, horse) in enumerate(loop):

    zebra=zebra.to(device)
    horse=horse.to(device)

    fake_horse = gen_H(zebra)
    fake_zebra=gen_Z(horse)

    # Save generated images with same style as training
    save_image(horse * 0.5 + 0.5, f'result_image/horses/{idx}.png')
    save_image(fake_zebra * 0.5 + 0.5, f'result_image/zebras/{idx}.png')



