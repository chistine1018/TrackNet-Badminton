import torch
import argparse
from torch.utils.data import TensorDataset, DataLoader
import torchvision.models as models
from TrackNet10 import TrackNet10
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
parser.add_argument('--load_weight', type = str, required=True, help = 'the weight you want to test')
parser.add_argument('--info_csv', type = str, default = None, help = 'the file you want to append the output to')
parser.add_argument('--location', type = str, required=True, help = 'the input dataset location')
args = parser.parse_args()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('GPU Use : ',torch.cuda.is_available())
test_data = TrackNetLoader('' , 'test')
test_loader = DataLoader(dataset = test_data, batch_size=args.batchsize, shuffle=False)

def outcome(y_pred, y_true, tol, fast):
    n = y_pred.shape[0]
    i = 0
    TP = TN = FP1 = FP2 = FN = TP_f = TN_f = FP1_f = FP2_f = FN_f = 0
    while i < n:
        for j in range(10):
            if int(fast[i][j]) == 1:
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
            else:
                if torch.max(y_pred[i][j]) == 0 and torch.max(y_true[i][j]) == 0:
                    TN_f += 1
                elif torch.max(y_pred[i][j]) > 0 and torch.max(y_true[i][j]) == 0:
                    FP2_f += 1
                elif torch.max(y_pred[i][j]) == 0 and torch.max(y_true[i][j]) > 0:
                    FN_f += 1
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
                        FP1_f += 1
                    else:
                        TP_f += 1
        i += 1
    return (TP, TN, FP1, FP2, FN, TP_f, TN_f, FP1_f, FP2_f, FN_f)

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
    loss = (-1)*(torch.square(1 - y_pred) * y_true * torch.log(torch.clamp(y_pred, eps, 1)) + torch.square(y_pred) * (1 - y_true) * torch.log(torch.clamp(1 - y_pred, eps, 1)))
    return torch.mean(loss)

def display(TP, TN, FP1, FP2, FN, TP_f, TN_f, FP1_f, FP2_f, FN_f):
    output =  "Weight: {}\n".format(args.load_weight)
    output += "======================Evaluate=======================\n"
    output += "Number of true positive: {}\n".format(TP)
    output += "Number of true negative: {}\n".format(TN)
    output += "Number of false positive FP1: {}\n".format(FP1)
    output += "Number of false positive FP2: {}\n".format(FP2)
    output += "Number of false negative: {}\n".format(FN)
    output += "Number of fast true positive: {}\n".format(TP_f)
    output += "Number of fast true negative: {}\n".format(TN_f)
    output += "Number of fast false positive FP1: {}\n".format(FP1_f)
    output += "Number of fast false positive FP2: {}\n".format(FP2_f)
    output += "Number of fast false negative: {}\n".format(FN_f)
    (accuracy, precision, recall, f1_score) = evaluation(TP, TN, FP1, FP2, FN)
    (accuracy_f, precision_f, recall_f, f1_score_f) = evaluation(TP_f, TN_f, FP1_f, FP2_f, FN_f)
    output += "Accuracy: {}\n".format(accuracy)
    output += "Precision: {}\n".format(precision)
    output += "Recall: {}\n".format(recall)
    output += "f1-score: {}\n".format(f1_score)
    output += "fast Accuracy: {}\n".format(accuracy_f)
    output += "fast Precision: {}\n".format(precision_f)
    output += "fast Recall: {}\n".format(recall_f)
    output += "fast f1-score: {}\n".format(f1_score_f)
    output += "=====================================================\n\n"
    print(output)
    if args.info_csv != None:
        with open(args.info_csv, "a") as f:
            writer = csv.writer(f)
            writer.writerow([args.location,accuracy,precision,recall,TP,TN,FP1,FP2,FN])
            location_fast = str(args.location)+'_fast'
            writer.writerow([location_fast,accuracy_f,precision_f,recall_f,TP_f,TN_f,FP1_f,FP2_f,FN_f])

def test():
    model.eval()
    #print('======================Evaluate=======================')
    TP = TN = FP1 = FP2 = FN = TP_f = TN_f = FP1_f = FP2_f = FN_f = 0
    for batch_idx, (data, label, fast) in enumerate(test_loader):
        data = data.type(torch.FloatTensor).to(device)
        label = label.type(torch.FloatTensor).to(device)
        with torch.no_grad():
            y_pred = model(data)
        y_pred = y_pred > 0.5
        (tp, tn, fp1, fp2, fn, tp_f, tn_f, fp1_f, fp2_f, fn_f) = outcome(y_pred, label, args.tol, fast)
        TP += tp
        TN += tn
        FP1 += fp1
        FP2 += fp2
        FN += fn
        TP_f += tp_f
        TN_f += tn_f
        FP1_f += fp1_f
        FP2_f += fp2_f
        FN_f += fn_f
        print('Test : [{}/{} ({:.0f}%)]'.format((batch_idx+1) * len(data), len(test_loader.dataset),100.0 * (batch_idx+1) / len(test_loader)))
    display(TP, TN, FP1, FP2, FN, TP_f, TN_f, FP1_f, FP2_f, FN_f)

model = TrackNet10()
model.to(device)
if args.optimizer == 'Adadelta':
    optimizer = torch.optim.Adadelta(model.parameters(), lr=args.lr, rho=0.9, eps=1e-06, weight_decay=0)
    #optimizer = torch.optim.Adam(model.parameters(), lr = args.lr, weight_decay = args.weight_decay)
else:
    optimizer = torch.optim.SGD(model.parameters(), lr = args.lr, weight_decay = args.weight_decay, momentum = args.momentum)

if(args.load_weight):
    checkpoint = torch.load(args.load_weight)
    model.load_state_dict(checkpoint['state_dict'])
    epoch = checkpoint['epoch']
test()
