import os
import sys
import json
import torch
from torch.utils.data import TensorDataset, DataLoader
import torchvision.models as models
import dataloader10
import numpy as np
import itertools
import cv2
import math

class TrackNet10(torch.nn.Module):
    def __init__(self, input_height=288, input_width=512 ): #input_height = 288, input_width = 512
        super(TrackNet10, self).__init__()

        #Layer1
        self.conv1 = torch.nn.Conv2d(10, 64, kernel_size = 3, stride = 1, padding=(1,1))
        self.nor1 = torch.nn.BatchNorm2d(64)

        #Layer2
        self.conv2 = torch.nn.Conv2d(64, 64, kernel_size = 3, stride = 1, padding=(1,1))
        self.nor2 = torch.nn.BatchNorm2d(64)

        #Layer3
        self.max3 = torch.nn.MaxPool2d(kernel_size = 2, stride = 2)

        #Layer4
        self.conv4 = torch.nn.Conv2d(64, 128, kernel_size = 3, stride = 1, padding=(1,1))
        self.nor4 = torch.nn.BatchNorm2d(128)

        #Layer5
        self.conv5 = torch.nn.Conv2d(128, 128, kernel_size = 3, stride = 1, padding=(1,1))
        self.nor5 = torch.nn.BatchNorm2d(128)

        #Layer6
        self.max6 = torch.nn.MaxPool2d(kernel_size = 2, stride = 2)

        #Layer7
        self.conv7 = torch.nn.Conv2d(128, 256, kernel_size = 3, stride = 1, padding=(1,1))
        self.nor7 = torch.nn.BatchNorm2d(256)

        #Layer8
        self.conv8 = torch.nn.Conv2d(256, 256, kernel_size = 3, stride = 1, padding=(1,1))
        self.nor8 = torch.nn.BatchNorm2d(256)

        #Layer9
        self.conv9 = torch.nn.Conv2d(256, 256, kernel_size = 3, stride = 1, padding=(1,1))
        self.nor9 = torch.nn.BatchNorm2d(256)

        #Layer10
        self.max10 = torch.nn.MaxPool2d(kernel_size = 2, stride = 2)

        #Layer11
        self.conv11 = torch.nn.Conv2d(256, 512, kernel_size = 3, stride = 1, padding=(1,1))
        self.nor11 = torch.nn.BatchNorm2d(512)

        #Layer12
        self.conv12 = torch.nn.Conv2d(512, 512, kernel_size = 3, stride = 1, padding=(1,1))
        self.nor12 = torch.nn.BatchNorm2d(512)

        #Layer13
        self.conv13 = torch.nn.Conv2d(512, 512, kernel_size = 3, stride = 1, padding=(1,1))
        self.nor13 = torch.nn.BatchNorm2d(512)

        #Layer14
        #upsample (2,2) 13 layer output和9 layer的output concat axis =1

        #Layer15
        self.conv15 = torch.nn.Conv2d(768, 256, kernel_size = 3, stride = 1, padding=(1,1))
        self.nor15 = torch.nn.BatchNorm2d(256)

        #Layer16
        self.conv16 = torch.nn.Conv2d(256, 256, kernel_size = 3, stride = 1, padding=(1,1))
        self.nor16 = torch.nn.BatchNorm2d(256)

        #Layer17
        self.conv17 = torch.nn.Conv2d(256, 256, kernel_size = 3, stride = 1, padding=(1,1))
        self.nor17 = torch.nn.BatchNorm2d(256)

        #Layer18
        #upsample (2,2) 17 layer output和5 layer的output concat axis =1

        #Layer19
        self.conv19 = torch.nn.Conv2d(384, 128, kernel_size = 3, stride = 1, padding=(1,1))
        self.nor19 = torch.nn.BatchNorm2d(128)

        #Layer20
        self.conv20 = torch.nn.Conv2d(128, 128, kernel_size = 3, stride = 1, padding=(1,1))
        self.nor20 = torch.nn.BatchNorm2d(128)

        #Layer21
        #upsample (2,2) 20 layer output和2 layer的output concat axis =1

        #Layer22
        self.conv22 = torch.nn.Conv2d(192, 64, kernel_size = 3, stride = 1, padding=(1,1))
        self.nor22 = torch.nn.BatchNorm2d(64)

        #Layer23
        self.conv23 = torch.nn.Conv2d(64, 64, kernel_size = 3, stride = 1, padding=(1,1))
        self.nor23 = torch.nn.BatchNorm2d(64)

        #Layer24
        self.conv24 = torch.nn.Conv2d(64, 10, kernel_size = 1, stride = 1)

        self.upsample = torch.nn.UpsamplingBilinear2d(scale_factor=2)
        self.act = torch.nn.ReLU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.act(x)
        x = self.nor1(x)

        x = self.conv2(x)
        x = self.act(x)
        x1 = self.nor2(x)

        x = self.max3(x1)

        x = self.conv4(x)
        x = self.act(x)
        x = self.nor4(x)

        x = self.conv5(x)
        x = self.act(x)
        x2 = self.nor5(x)

        x = self.max6(x2)

        x = self.conv7(x)
        x = self.act(x)
        x = self.nor7(x)

        x = self.conv8(x)
        x = self.act(x)
        x = self.nor8(x)

        x = self.conv9(x)
        x = self.act(x)
        x3 = self.nor9(x)

        x = self.max10(x3)

        x = self.conv11(x)
        x = self.act(x)
        x = self.nor11(x)

        x = self.conv12(x)
        x = self.act(x)
        x = self.nor12(x)

        x = self.conv13(x)
        x = self.act(x)
        x = self.nor13(x)

        x = torch.cat((self.upsample(x), x3), 1)

        x = self.conv15(x)
        x = self.act(x)
        x = self.nor15(x)

        x = self.conv16(x)
        x = self.act(x)
        x = self.nor16(x)

        x = self.conv17(x)
        x = self.act(x)
        x = self.nor17(x)

        x = torch.cat((self.upsample(x), x2), 1)

        x = self.conv19(x)
        x = self.act(x)
        x = self.nor19(x)

        x = self.conv20(x)
        x = self.act(x)
        x = self.nor20(x)

        x = torch.cat((self.upsample(x), x1), 1)

        x = self.conv22(x)
        x = self.act(x)
        x = self.nor22(x)

        x = self.conv23(x)
        x = self.act(x)
        x = self.nor23(x)

        x = self.conv24(x)
        x = torch.sigmoid(x)

        return x