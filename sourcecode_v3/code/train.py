import os
import sys
import json
import torch
import argparse
from torch.utils.data import TensorDataset, DataLoader
from TrackNet10 import TrackNet10
import torchvision.models as models
import dataloader10
import numpy as np
from dataloader10 import TrackNetLoader
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import itertools
import cv2
import math
import csv

parser = argparse.ArgumentParser(description = 'Pytorch TrackNet10')
parser.add_argument('--batchsize', type = int, default = 8, help = 'input batch size for training (defalut: 8)')
parser.add_argument('--epochs', type = int, default = 30, help = 'number of epochs to train (default: 30)')
parser.add_argument('--lr', type = float, default = 1, help = 'learning rate (default: 1)')
parser.add_argument('--tol', type = int, default = 4, help = 'tolerance values (defalut: 4)')
parser.add_argument('--optimizer', type = str, default = 'Adadelta', help = 'Adadelta or SGD (default: Adadelta)')
parser.add_argument('--momentum', type = float, default = 0.9, help = 'momentum fator (default: 0.9)')
parser.add_argument('--weight_decay', type = float, default = 5e-4, help = 'weight decay (default: 5e-4)')
parser.add_argument('--seed', type=int, default = 1, help = 'random seed (default: 1)')
parser.add_argument('--load_weight', type = str, default = None, help = 'the weight you want to retrain')
parser.add_argument('--checkpoint', type = str, default = None, help = 'the checkpoint weight')
parser.add_argument('--save_weight', type = str, default = 'TrackNet10', help = 'the weight you want to save') # TrackNet10_1.tar, ...... , TrackNet10_30.tar
#parser.add_argument('--save_fig', type = str, default = 'output', help = 'the figure you want to save') # output_loss.jpg & output_accuracy.jpg
parser.add_argument('--result_csv', type = str, default = None, help = 'the file you want to append the output to')
args = parser.parse_args()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") # Use All GPUs
print('GPU Use : ',torch.cuda.is_available())
train_data = TrackNetLoader('' , 'train')
train_loader = DataLoader(dataset = train_data, batch_size=args.batchsize, shuffle=True)

def outcome(y_pred, y_true, tol):
    n = y_pred.shape[0]
    i = 0
    TP = TN = FP1 = FP2 = FN = 0
    while i < n:
        for j in range(10):
            if torch.max(y_pred[i][j]) == 0 and torch.max(y_true[i][j]) == 0:
                TN += 1
            elif torch.max(y_pred[i][j]) > 0 and torch.max(y_true[i][j]) == 0:
                FP2 += 1
            elif torch.max(y_pred[i][j]) == 0 and torch.max(y_true[i][j]) > 0:
                FN += 1
            elif torch.max(y_pred[i][j]) > 0 and torch.max(y_true[i][j]) > 0:
                h_pred = (y_pred[i][j] * 255).cpu().numpy()
                h_true = (y_true[i][j] * 255).cpu().numpy()
                h_pred = h_pred.astype('uint8')
                h_true = h_true.astype('uint8')
                #h_pred
                (cnts, _) = cv2.findContours(h_pred.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                rects = [cv2.boundingRect(ctr) for ctr in cnts]
                max_area_idx = 0
                max_area = rects[max_area_idx][2] * rects[max_area_idx][3]
                for j in range(len(rects)):
                    area = rects[j][2] * rects[j][3]
                    if area > max_area:
                        max_area_idx = j
                        max_area = area
                target = rects[max_area_idx]
                (cx_pred, cy_pred) = (int(target[0] + target[2] / 2), int(target[1] + target[3] / 2))

                #h_true
                (cnts, _) = cv2.findContours(h_true.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                rects = [cv2.boundingRect(ctr) for ctr in cnts]
                max_area_idx = 0
                max_area = rects[max_area_idx][2] * rects[max_area_idx][3]
                for j in range(len(rects)):
                    area = rects[j][2] * rects[j][3]
                    if area > max_area:
                        max_area_idx = j
                        max_area = area
                target = rects[max_area_idx]
                (cx_true, cy_true) = (int(target[0] + target[2] / 2), int(target[1] + target[3] / 2))
                dist = math.sqrt(pow(cx_pred-cx_true, 2)+pow(cy_pred-cy_true, 2))
                if dist > tol:
                    FP1 += 1
                else:
                    TP += 1
        i += 1
    return (TP, TN, FP1, FP2, FN)

def weighted_loss(y_true):
    n = y_true.shape[0] # 16
    N = 10 # TrackNet 10
    i = 0
    weighted = 0
    while i < n:
        points_x = [np.nan] * N # 0~512
        points_y = [np.nan] * N # 0~288
        last_distance = np.nan
        for j in range(N):
            if torch.max(y_true[i][j]) > 0:
                h_true = (y_true[i][j] * 255).cpu().numpy()
                h_true = h_true.astype('uint8')

                #h_true
                (cnts, _) = cv2.findContours(h_true.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                rects = [cv2.boundingRect(ctr) for ctr in cnts]
                max_area_idx = 0
                max_area = rects[max_area_idx][2] * rects[max_area_idx][3]
                for k in range(len(rects)):
                    area = rects[k][2] * rects[k][3]
                    if area > max_area:
                        max_area_idx = k
                        max_area = area
                target = rects[max_area_idx]
                (cx_true, cy_true) = (int(target[0] + target[2] / 2), int(target[1] + target[3] / 2))
                points_x[j] = cx_true
                points_y[j] = cy_true
        w = 0
        for j in range(1,N):
            if not np.isnan(points_x[j-1]) and not np.isnan(points_x[j]) and not np.isnan(points_y[j-1]) and not np.isnan(points_y[j]):
                last_distance = abs(points_x[j] - points_x[j-1]) + abs(points_y[j]-points_y[j-1])
                w += last_distance
            elif not np.isnan(last_distance):
                w += last_distance
            else:
                w += 10
        w /= 100
        weighted += w
        i += 1
    weighted /= n
    print("Batch Weighted:", weighted)
    return weighted

def evaluation(TP, TN, FP1, FP2, FN):
    try:
        accuracy = (TP + TN) / (TP + TN + FP1 + FP2 + FN)
    except:
        accuracy = 0
    try:
        precision = TP / (TP + FP1 + FP2)
    except:
        precision = 0
    try:
        recall = TP / (TP + FN)
    except:
        recall = 0
    try:
        f1_score = 2 * (precision * recall) / (precision + recall)
    except:
        f1_score = 0
    return (accuracy, precision, recall, f1_score)

def WBCE(y_pred, y_true):
    eps = 1e-7
    loss = ((-1)*(torch.square(1 - y_pred) * y_true * torch.log(torch.clamp(y_pred, eps, 1)) + torch.square(y_pred) * (1 - y_true) * torch.log(torch.clamp(1 - y_pred, eps, 1))))
    # return torch.mean(loss) * torch.mean(fast)
    return torch.mean(loss)

def train(epoch):
    model.train()
    train_loss = 0
    TP = TN = FP1 = FP2 = FN = 0
    for batch_idx, (data, label, fast) in enumerate(train_loader):
        data = data.type(torch.FloatTensor).to(device)
        label = label.type(torch.FloatTensor).to(device)
        # fast = label.type(torch.FloatTensor).to(device)
        optimizer.zero_grad()
        y_pred = model(data)
        loss = WBCE(y_pred, label)
        loss = loss * weighted_loss(label) * torch.mean(fast)
        print('Train Epoch" {} [{}/{} ({:.0f}%)]\tLoss : {:.6f}'.format(epoch, (batch_idx+1) * len(data), len(train_loader.dataset),100.0 * (batch_idx+1) / len(train_loader), loss.data))
        print('Fast means = {}'.format(torch.mean(fast)))
        train_loss += loss.data
        loss.backward()
        optimizer.step()
        if(epoch % 1 == 0):
            y_pred = y_pred > 0.5
            (tp, tn, fp1, fp2, fn) = outcome(y_pred, label, args.tol)
            TP += tp
            TN += tn
            FP1 += fp1
            FP2 += fp2
            FN += fn
    train_loss /= len(train_loader)
    train_loss = train_loss.item() # tensor to float
    if(epoch % 1 == 0):
        display(epoch, TP, TN, FP1, FP2, FN, train_loss)
        savefilename = args.save_weight + '_{}.tar'.format(epoch)
        torch.save({'epoch':epoch,'state_dict':model.state_dict(),'optimizer':optimizer.state_dict(),}, savefilename)
    return train_loss

def display(epoch, TP, TN, FP1, FP2, FN, loss):
    output =  "Epoch: {}\n".format(epoch)
    output += "======================Evaluate=======================\n"
    output += "Number of true positive: {}\n".format(TP)
    output += "Number of true negative: {}\n".format(TN)
    output += "Number of false positive FP1: {}\n".format(FP1)
    output += "Number of false positive FP2: {}\n".format(FP2)
    output += "Number of false negative: {}\n".format(FN)
    output += "Loss: {}\n".format(loss)
    (accuracy, precision, recall, f1_score) = evaluation(TP, TN, FP1, FP2, FN)
    output += "Accuracy: {}\n".format(accuracy)
    output += "Precision: {}\n".format(precision)
    output += "Recall: {}\n".format(recall)
    output += "f1-score: {}\n".format(f1_score)
    output += "=====================================================\n\n"
    print(output)
    if args.result_csv != None:
        with open(args.result_csv, "a") as f:
            writer = csv.writer(f)
            writer.writerow([epoch,accuracy,precision,recall,TP,TN,FP1,FP2,FN,loss])


model = TrackNet10()
model.to(device)
if args.optimizer == 'Adadelta':
    optimizer = torch.optim.Adadelta(model.parameters(), lr=args.lr, rho=0.9, eps=1e-06, weight_decay=0)

    #optimizer = torch.optim.Adam(model.parameters(), lr = args.lr, weight_decay = args.weight_decay)
else:
    optimizer = torch.optim.SGD(model.parameters(), lr = args.lr, weight_decay = args.weight_decay, momentum = args.momentum)

if(args.checkpoint):
    print('=================Continue Training==================')
    checkpoint = torch.load(args.checkpoint)
    model.load_state_dict(checkpoint['state_dict'])
    epoch = checkpoint['epoch']+1
    optimizer.load_state_dict(checkpoint['optimizer'])
elif(args.load_weight):
    print('====================Load Weight=====================')
    checkpoint = torch.load(args.load_weight)
    model.load_state_dict(checkpoint['state_dict'])
    epoch = 1
else:
    epoch = 1

train_loss = []

for i in range(epoch, args.epochs + 1):
    loss = train(i)
    train_loss.append(loss)
#show(train_loss,train_accuracy, train_precision, train_recall)
