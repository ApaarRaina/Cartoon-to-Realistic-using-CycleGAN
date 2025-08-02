#17

import torch
import torch.nn as nn
import os
from PIL import Image
import numpy as np
from torch.utils.data import Dataset,DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
from torchvision.utils import save_image
import sys
import matplotlib.pyplot as plt
import random
import copy
import torch.multiprocessing
from generator import Generator
from discriminator import Discriminator


if not os.path.isdir('saved_image'):
    os.mkdir('saved_image')
    os.mkdir('saved_image/horses')
    os.mkdir('saved_image/zebras')

class Config:
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    TRAIN_DIR = "horse2zebra"
    VAL_DIR = "horse2zebra"
    BATCH_SIZE = 1
    LEARNING_RATE = 0.0001
    LAMBDA_IDENTITY = 0.5
    LAMBDA_CYCLE = 1
    NUM_WORKERS = 4
    NUM_EPOCHS = 10
    LOAD_MODEL = True
    SAVE_MODEL = True
    CHECKPOINT_GEN_H = "pretrained/genh_2.pth.tar"
    CHECKPOINT_GEN_Z = "pretrained/genz_2.pth.tar"
    CHECKPOINT_CRITIC_H = "pretrained/critich_2.pth.tar"
    CHECKPOINT_CRITIC_Z = "pretrained/criticz_2.pth.tar"
    transforms = A.Compose(
        [
            A.Resize(width=128, height=128),
            A.HorizontalFlip(p=0.5),
            A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], max_pixel_value=255),
            ToTensorV2(),
        ],
        additional_targets={"image0": "image"},
    )

config = Config()


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

class Utils:
    def save_checkpoint(self,model, optimizer, file_name = "Model.pth.tar"):
        print('__saving checkpoint__')
        checkpoint = {
            'model_state': model.state_dict(),
            'optimizer_state': optimizer.state_dict()
        }
        torch.save(checkpoint,file_name)
    def load_checkpoint(self,checkpoint_path,model, optimizer, lr):
        checkpoint = torch.load(checkpoint_path, map_location = config.DEVICE)
        print('__loading checkpoint for model__')
        model.load_state_dict(checkpoint['model_state'])
        print('__loading checkpoint for optimizer__')
        optimizer.load_state_dict(checkpoint['optimizer_state'])
        for param_group  in optimizer.param_groups:
            param_group['lr'] = lr


utils = Utils()
config = Config()


class Trainer:
    def train_epoch(self,disc_H,disc_Z,gen_Z,gen_H, loader,opt_disc, opt_gen,l1,mse,d_scaler,g_scaler):
        loop = tqdm(loader,leave=True)
        for idx, (zebra,horse) in enumerate(loop):
            horse = horse.to(config.DEVICE)
            zebra = zebra.to(config.DEVICE)

            # Train Disc
            with torch.cuda.amp.autocast():

                # Loss for Disc of Zebra
                fake_zebra = gen_Z(horse)
                D_Z_real = disc_Z(zebra)
                D_Z_fake = disc_Z(fake_zebra.detach())
                D_Z_real_loss = mse(D_Z_real,torch.ones_like(D_Z_real))
                D_Z_fake_loss = mse(D_Z_fake,torch.zeros_like(D_Z_fake))
                D_Z_loss = D_Z_real_loss+D_Z_fake_loss

                # loss for Disc of horse
                fake_horse = gen_H(zebra)
                D_H_real = disc_H(horse)
                D_H_fake = disc_H(fake_horse.detach())
                D_H_real_loss = mse(D_H_real,torch.ones_like(D_H_real))
                D_H_fake_loss = mse(D_H_fake,torch.zeros_like(D_H_fake))
                D_H_loss = D_H_real_loss+D_H_fake_loss

                D_loss = (D_H_loss+D_Z_loss)/2
            opt_disc.zero_grad()
            d_scaler.scale(D_loss).backward(retain_graph = True)
            d_scaler.step(opt_disc)
            d_scaler.update()

            # Training the generator
            with torch.cuda.amp.autocast():
                D_H_fake = disc_H(fake_horse)
                D_Z_fake = disc_Z(fake_zebra)

                # adversarial loss
                loss_G_H = mse(D_H_fake, torch.ones_like(D_H_fake))
                loss_G_Z = mse(D_Z_fake, torch.ones_like(D_Z_fake))

                # cycle loss
                cycle_horse = gen_H(fake_zebra)
                cycle_zebra = gen_Z(fake_horse)
                cycle_zebra_loss = l1(zebra,cycle_zebra)
                cycle_horse_loss = l1(horse,cycle_horse)

                # identity loss
                cycle_horse = gen_H(horse)
                cycle_zebra = gen_Z(zebra)
                identity_zebra_loss = l1(zebra,cycle_zebra)
                identity_horse_loss = l1(horse,cycle_horse)

            G_loss = (loss_G_Z+loss_G_H+config.LAMBDA_CYCLE*(cycle_zebra_loss+cycle_horse_loss)+config.LAMBDA_IDENTITY*(identity_zebra_loss+identity_horse_loss))
            opt_gen.zero_grad()
            g_scaler.scale(G_loss).backward()
            g_scaler.step(opt_gen)
            g_scaler.update()
            loop.set_postfix(d_loss = D_loss.item(), g_loss = G_loss.item())
            if idx %200 == 0:
                save_image(fake_horse*0.5+0.5, f'saved_image/horses/{idx}.png')
                save_image(fake_zebra*0.5+0.5, f'saved_image/zebras/{idx}.png')

    def train(self):
        disc_H = Discriminator().to(config.DEVICE)
        disc_Z = Discriminator().to(config.DEVICE)

        gen_Z = Generator().to(config.DEVICE)
        gen_H = Generator().to(config.DEVICE)
        opt_disc = torch.optim.Adam(
            list(disc_H.parameters()) + list(disc_Z.parameters()),
            lr=config.LEARNING_RATE,
            betas=(0.5, 0.999),
        )

        opt_gen = torch.optim.Adam(
            list(gen_Z.parameters()) + list(gen_H.parameters()),
            lr=config.LEARNING_RATE,
            betas=(0.5, 0.999),
        )

        L1 = nn.L1Loss()
        mse = nn.MSELoss()

        if config.LOAD_MODEL:
            utils.load_checkpoint(
                config.CHECKPOINT_GEN_H,
                gen_H,
                opt_gen,
                config.LEARNING_RATE,
            )
            utils.load_checkpoint(
                config.CHECKPOINT_GEN_Z,
                gen_Z,
                opt_gen,
                config.LEARNING_RATE,
            )
            utils.load_checkpoint(
                config.CHECKPOINT_CRITIC_H,
                disc_H,
                opt_disc,
                config.LEARNING_RATE,
            )
            utils.load_checkpoint(
                config.CHECKPOINT_CRITIC_Z,
                disc_Z,
                opt_disc,
                config.LEARNING_RATE,
            )

        dataset = HorseZebraDataset(
            root_horse=config.TRAIN_DIR + "/trainA",  #cartoon
            root_zebra=config.TRAIN_DIR + "/trainB",  #realistic
            transform=config.transforms,
        )
        val_dataset = HorseZebraDataset(
            root_horse=config.VAL_DIR + '/testA',
            root_zebra=config.VAL_DIR + '/testB',
            transform=config.transforms,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=1,
            shuffle=False,
            pin_memory=True,
        )
        loader = DataLoader(
            dataset,
            batch_size=config.BATCH_SIZE,
            shuffle=True,
            num_workers=config.NUM_WORKERS,
            pin_memory=True,
        )
        g_scaler = torch.cuda.amp.GradScaler()
        d_scaler = torch.cuda.amp.GradScaler()

        for epoch in range(config.NUM_EPOCHS):
            self.train_epoch(
                disc_H,
                disc_Z,
                gen_Z,
                gen_H,
                loader,
                opt_disc,
                opt_gen,
                L1,
                mse,
                d_scaler,
                g_scaler,
            )

            if config.SAVE_MODEL:
                utils.save_checkpoint(gen_H, opt_gen, file_name=config.CHECKPOINT_GEN_H)
                utils.save_checkpoint(gen_Z, opt_gen, file_name=config.CHECKPOINT_GEN_Z)
                utils.save_checkpoint(disc_H, opt_disc, file_name=config.CHECKPOINT_CRITIC_H)
                utils.save_checkpoint(disc_Z, opt_disc, file_name=config.CHECKPOINT_CRITIC_Z)

if __name__ == "__main__":
    torch.multiprocessing.freeze_support()  # optional, but safe

    Trainer().train()

    plt.imshow(Image.open('./saved_image/zebras/800.png'))