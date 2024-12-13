import cv2 
import csv
from glob import glob
import numpy as np
import os
import random
import pandas as pd
import argparse
import csv

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

parser = argparse.ArgumentParser(description = 'Preprocess')
parser.add_argument('--dataset_folder', type=str, required=True, help = 'Dataset Folder (Include Different Locations)')
parser.add_argument('--locations', type=str, nargs='+', required=True, help = 'Example: --location EC234 nctu_old_gym profession_dataset (Use 3 locations)')
parser.add_argument('--info_csv', type = str, default = None, help = 'the file you want to append the output info to')
parser.add_argument('--result_csv', type = str, default = None, help = 'the file you want to append the output to')
parser.add_argument('--test', type=int,default=0,help = '0 means train data (default), N means Nth test data')
args = parser.parse_args()

dataset = args.dataset_folder

location = []
location_frame_cnt = []

train_x = []
train_y = []
train_z = []
for l in args.locations:
    frame_cnt = 0
    date_list = []
    if os.path.isdir(os.path.join(dataset, l)):
        train_path = glob(os.path.join(dataset, l, 'frame', '*'))
        frame_cnt += len(glob(os.path.join(dataset, l, 'frame', '*','*.png')))
        for i in range(len(train_path)):
            train_path[i] = os.path.basename(train_path[i])
        for p in train_path:
            # print(p)
            video_heatmap_folder = os.path.join(dataset,l , 'heatmap', p)
            if not os.path.exists(video_heatmap_folder):
                os.makedirs(video_heatmap_folder)

            labelPath = os.path.join(dataset, l, 'csv', p+'_ball.csv')
            a = cv2.imread(os.path.join(dataset, l, 'frame', p, '0.png'))
            ratio_h = a.shape[0] / HEIGHT
            ratio_w = a.shape[1] / WIDTH
            data = pd.read_csv(labelPath)
            no = data['Frame'].values
            v = data['Visibility'].values
            x = data['X'].values
            y = data['Y'].values
            fast = data['Fast'].values
            num = no.shape[0]
            r = os.path.join(dataset, l, 'frame', p)
            r2 = os.path.join(dataset, l, 'heatmap', p)
            x_data_tmp = []
            y_data_tmp = []
            for i in range(num-9):
                unit = []
                for j in range(10):
                    target=str(no[i+j])+'.png'
                    png_path = os.path.join(r, target)
                    unit.append(png_path)
                train_x.append(unit)
                unit = []
                for j in range(10):
                    target=str(no[i+j])+'.png'
                    heatmap_path = os.path.join(r2, target)
                    unit.append(heatmap_path)
                    if not os.path.isfile(heatmap_path):
                        if v[i+j] == 0:
                            heatmap_img = genHeatMap(WIDTH, HEIGHT, -1, -1, sigma, mag)
                        else:
                            heatmap_img = genHeatMap(WIDTH, HEIGHT, int(x[i+j]/ratio_w), int(y[i+j]/ratio_h), sigma, mag)
                        heatmap_img *= 255
                        cv2.imwrite(heatmap_path,heatmap_img)
                train_y.append(unit)
                unit = []
                for j in range(10):
                    if int(fast[i+j]) == 1:
                        unit.append('5')
                    elif int(fast[i+j]) == 0:
                        unit.append('1')
                train_z.append(unit)

    location.append(l)
    location_frame_cnt.append(frame_cnt)

if args.info_csv != None:
    with open(args.info_csv, "a") as f:
        writer = csv.writer(f)
        writer.writerow(['Location','Frames'])
        for i in range(len(location)):
            row = []
            row.append(location[i])
            row.append(location_frame_cnt[i])
            writer.writerow(row)

if args.result_csv != None:
    with open(args.result_csv, "a") as f:
        writer = csv.writer(f)
        if args.test == 1:
            writer.writerow(['Location','Accuracy','Precision','Recall','TP','TN','FP1','FP2','FN'])
        elif args.test == 0:
            writer.writerow(['Epoch','Accuracy','Precision','Recall','TP','TN','FP1','FP2','FN','Loss'])

if args.test != 0:
    outputfile_name = 'tracknet_test_list_x_10.csv'
elif args.test == 0:
    outputfile_name = 'tracknet_train_list_x_10.csv'
with open(outputfile_name,'w') as outputfile:
    for i in range(len(train_x)):
        outputfile.write("%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n"%(train_x[i][0], train_x[i][1], train_x[i][2], train_x[i][3], train_x[i][4], train_x[i][5], train_x[i][6], train_x[i][7], train_x[i][8], train_x[i][9]))

if args.test != 0:
    outputfile_name = 'tracknet_test_list_y_10.csv'
elif args.test == 0:
    outputfile_name = 'tracknet_train_list_y_10.csv'
with open(outputfile_name,'w') as outputfile:
    for i in range(len(train_x)):
        outputfile.write("%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n"%(train_y[i][0], train_y[i][1], train_y[i][2], train_y[i][3], train_y[i][4], train_y[i][5], train_y[i][6], train_y[i][7], train_y[i][8], train_y[i][9]))

if args.test != 0:
    outputfile_name = 'tracknet_test_list_z_10.csv'
elif args.test == 0:
    outputfile_name = 'tracknet_train_list_z_10.csv'
with open(outputfile_name,'w') as outputfile:
    for i in range(len(train_x)):
        outputfile.write("%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n"%(train_z[i][0], train_z[i][1], train_z[i][2], train_z[i][3], train_z[i][4], train_z[i][5], train_z[i][6], train_z[i][7], train_z[i][8], train_z[i][9]))

print('finish')
