import torch
import torch.nn as nn
import random

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(3, 64, 5, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(64, 128, 5, stride=2, padding=1),
            nn.InstanceNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(128, 256, 5, stride=2, padding=1),
            nn.InstanceNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(256, 512, 5, stride=2, padding=1),
            nn.InstanceNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(256, 1, 5, padding=0)
        )
    def forward(self, x):
        out = self.main(x)
        out2 = torch.flatten(out)
        return out2


class ImageBuffer():
    def __init__(self):
        self.data = []
        self.max_size = 50

    def push_and_pop(self, data):
        to_return = []
        for el in data.data:
            el = torch.unsqueeze(el, 0)
            if len(self.data) < self.max_size:
                self.data.append(el)
                to_return.append(el)
            else:
                if random.uniform(0, 1) > 0.5:
                    i = random.randint(0, self.max_size - 1)
                    to_return.append(self.data[i].clone())
                    self.data[i] = el
                else:
                    to_return.append(el)
        return torch.cat(to_return)


class TensorBuffer:

    def __init__(self, max_size=25, shape=(3,128,128), device="cuda" if torch.cuda.is_available() else "cpu"):
        self.max_size = max_size
        self.buffer = torch.zeros((max_size, *shape), device=device)
        self.index = 0
        self.is_full = False

    def push(self, value):

        self.buffer[self.index] = value.detach().clone()
        self.index = (self.index + 1) % self.max_size
        if self.index == 0:
            self.is_full = True

    def get_all(self):
        if self.is_full:
            return torch.cat([self.buffer[self.index:], self.buffer[:self.index]], dim=0)
        else:
            return self.buffer[:self.index]
