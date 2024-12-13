import matplotlib
matplotlib.use('Agg')
import os
import argparse
import csv
import matplotlib.pyplot as plt
import numpy as np

parser = argparse.ArgumentParser(description = 'Show Figure')
parser.add_argument('--csv', type=str, required=True, help = 'csv file')
parser.add_argument('--type', type=str, required=True, help = 'train / test (csv type, Choose one)')
args = parser.parse_args()

if args.type == 'train':
    epoch = []
    accuracy = []
    precision = []
    recall = []
    loss = []
    with open(args.csv, newline='') as f:
        rows = csv.reader(f)
        flag = False
        for row in rows:
            if flag:
                epoch.append(float(row[0]))
                accuracy.append(float(row[1]))
                precision.append(float(row[2]))
                recall.append(float(row[3]))
                loss.append(float(row[9]))
            if row[0] == 'Epoch':
                flag = True
    fig = plt.figure()
    axAccuracy = fig.add_subplot(1,2,1)
    axAccuracy.set_title('Accuracy, Precision, Recall')
    axAccuracy.set_xlabel('Epoch')
    l1, = axAccuracy.plot(epoch, accuracy, marker=".", color='red')
    l2, = axAccuracy.plot(epoch, precision, marker=".", color='green')
    l3, = axAccuracy.plot(epoch, recall, marker=".", color='blue')
    axAccuracy.legend(handles=[l1,l2,l3],labels=['accuracy', 'precision', 'recall'],loc='best')

    axLoss = fig.add_subplot(1,2,2)
    axLoss.set_title('Loss')
    axLoss.set_xlabel('Epoch')
    axLoss.plot(epoch, loss, color='red')
    plt.savefig(os.path.splitext(args.csv)[0] + '.png')

elif args.type == 'test':
    locations = []
    accuracy = []
    precision = []
    recall = []
    with open(args.csv, newline='') as f:
        rows = csv.DictReader(f)
        for row in rows:
            locations.append(row['Location'])
            accuracy.append(float(row['Accuracy']))
            precision.append(float(row['Precision']))
            recall.append(float(row['Recall']))
    x = np.arange(len(locations))
    width = 0.02
    plt.bar(x,accuracy, width, color='red', label='Accuracy')
    plt.bar(x+width,precision, width, color='green', label='Precision')
    plt.bar(x+width*2,recall, width, color='blue', label='Recall')
    plt.xticks(x + width , locations)
    plt.legend(bbox_to_anchor=(1,1), loc='best')
    plt.savefig(os.path.splitext(args.csv)[0] + '.png')