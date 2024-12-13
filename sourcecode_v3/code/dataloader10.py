import pandas as pd
from torch.utils import data
import numpy as np
import torch
import os
from PIL import Image
import random
import cv2

HEIGHT=288
WIDTH=512
mag = 1
sigma = 2.5

def genHeatMap(w, h, cx, cy, r, mag):
    if cx < 0 or cy < 0:
      return np.zeros((h, w))
    x, y = np.meshgrid(np.linspace(1, w, w), np.linspace(1, h, h))
    heatmap = ((y - (cy + 1))**2) + ((x - (cx + 1))**2)
    heatmap[heatmap <= r**2] = 1
    heatmap[heatmap > r**2] = 0
    return heatmap*mag


def getData(mode):
    if mode == 'train':
        img = pd.read_csv('tracknet_train_list_x_10.csv')
        label = pd.read_csv('tracknet_train_list_y_10.csv')
        fast = pd.read_csv('tracknet_train_list_z_10.csv')
        return np.squeeze(img.values), np.squeeze(label.values), np.squeeze(fast.values)
    elif mode == 'test':
        img = pd.read_csv('tracknet_test_list_x_10.csv')
        label = pd.read_csv('tracknet_test_list_y_10.csv')
        fast = pd.read_csv('tracknet_test_list_z_10.csv')
        return np.squeeze(img.values), np.squeeze(label.values), np.squeeze(fast.values)
    else:
        return None

class TrackNetLoader(data.Dataset):
    def __init__(self, root, mode):
        self.root = root
        self.img_name, self.label_name, self.fast = getData(mode)
        self.mode = mode
        img = Image.open(self.img_name[0][0]).convert('LA')
        w, h = img.size
        self.ratio = h / HEIGHT
        print("> Found %d data..." % (len(self.img_name)))

    def __len__(self):
        """'return the size of dataset"""
        return len(self.img_name)

    def __getitem__(self, index):
        img_path = self.img_name[index]
        label_path = self.label_name[index]
        fast = self.fast[index]
        img_all = []
        label_all = []
        fast_all = []
        for i in range(10):
            x = Image.open(img_path[i]).convert('L')
            x = x.resize((WIDTH, HEIGHT))

            x = np.asarray(x) / 255.0
            img_all.append(x)

            y = Image.open(label_path[i])
            y = np.asarray(y) / 255.0
            label_all.append(y)

            fast_all.append(float(fast[i]))

        img_all = np.asarray(img_all)
        label_all = np.asarray(label_all)
        fast_all = np.asarray(fast_all)
        '''
        if self.mode == 'train':
          if random.random() < 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
        '''
        return img_all, label_all, fast_all

'''
img = pd.read_csv('tracknet_train_list_x.csv')
label = pd.read_csv('tracknet_train_list_y.csv')
img = np.squeeze(img.values)
label = np.squeeze(label.values)

img_path = img[0]
label_path = label[0]
img_all = []
label_all = []

img_all = np.asarray(img_all)
label_all = np.asarray(label_all)
img_all = (img_all/255)
print((img_all))
print((label_all).shape)
'''