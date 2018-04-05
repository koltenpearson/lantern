#an implementation of unet in pytorch
import torch
import numpy as np
from torch import nn
import torch.nn.functional as func


class UNet(nn.module) :

    def __init__(self, in_depth=3, out_depth=2, padding=False) :
        relu = nn.ReLU(True)
        self.crop = not padding

        pad = 0
        if padding :
            pad = 1

        self.down1 = nn.Sequential(
            nn.Conv2d(in_depth,64, kernel_size=3, stride=1, padding=pad),relu,
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=pad),relu,
        )

        self.down2 = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Conv2d(64,128, kernel_size=3, stride=1, padding=pad), relu,
            nn.Conv2d(128,128, kernel_size=3, stride=1, padding=pad), relu,
        )

        self.down3 = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Conv2d(128,256, kernel_size=3, stride=1, padding=pad), relu,
            nn.Conv2d(256,256, kernel_size=3, stride=1, padding=pad), relu,
        )

        self.down4 = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Conv2d(256,512, kernel_size=3, stride=1, padding=pad), relu,
            nn.Conv2d(256,512, kernel_size=3, stride=1, padding=pad), relu,
        )

        self.u = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Conv2d(512,1024, kernel_size=3, stride=1, padding=pad), relu,
            nn.Conv2d(1024,1024, kernel_size=3, stride=1, padding=pad), relu,
            nn.ConvTranspose2d(1024,512, kernel_size=2, stride=2, padding=0)
        )

        self.up4 = nn.Sequential(
            nn.Conv2d(1024,512, kernel_size=3, stride=1, padding=pad), relu,
            nn.Conv2d(512,512, kernel_size=3, stride=1, padding=pad), relu,
            nn.ConvTranspose2d(512,256, kernel_size=2, stride=2, padding=0)
        )

        self.up3 = nn.Sequential(
            nn.Conv2d(512,256, kernel_size=3, stride=1, padding=pad), relu,
            nn.Conv2d(256,256, kernel_size=3, stride=1, padding=pad), relu,
            nn.ConvTranspose2d(256,128, kernel_size=2, stride=2, padding=0)
        )

        self.up2 = nn.Sequential(
            nn.Conv2d(256,128, kernel_size=3, stride=1, padding=pad), relu,
            nn.Conv2d(128,128, kernel_size=3, stride=1, padding=pad), relu,
            nn.ConvTranspose2d(128,64, kernel_size=2, stride=2, padding=0)
        )

        self.up2 = nn.Sequential(
            nn.Conv2d(128,64, kernel_size=3, stride=1, padding=pad), relu,
            nn.Conv2d(64,64, kernel_size=3, stride=1, padding=pad), relu,
            nn.Conv2d(64,out_depth, kernel_size=1, stride=1, padding=0)
        )

        self.down = [self.down1, self.down2, self.down3, self.down4]
        self.up = [self.up4, self.up3, self.up2, self.up1]


    def forward(self, inp) :
        stack = [inp]

        for l in self.down :
            stack.append(l(stack[-1]))

        result = self.u(stack[-1])

        for l in self.up :
            skip = stack.pop()

            if self.crop :
                r_crop = (skip.shape[1] - result.shape[2])//2
                c_crop = (skip.shape[2] - result.shape[3])//2
                skip = skip[:,:,r_crop:-r_crop,c_crop:-c_crop]

            result = l(torch.cat((skip,result), 1))

        return result



