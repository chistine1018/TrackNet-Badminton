import cv2
import csv
import os
import sys
import shutil
from glob import glob
import argparse

parser = argparse.ArgumentParser(description = 'Frame Generator')
parser.add_argument('--dataset_folder', type=str, required=True, help = 'Dataset Folder (Include Different Locations)')
parser.add_argument('--locations', type=str, nargs='+', required=True, help = 'Example: --location EC234 nctu_old_gym profession_dataset (Use 3 locations)')
args = parser.parse_args()

game_list = []
for location in args.locations:
    if os.path.isdir(os.path.join(args.dataset_folder,location)):
        game_list.append(os.path.join(args.dataset_folder,location))

for game in game_list:
    p = os.path.join(game, 'video', '*')
    video_list = glob(p)
    frame_folder = os.path.join(game, 'frame')
    if not os.path.isdir(frame_folder):
        os.makedirs(frame_folder)
    for videoName in video_list:
        rallyName = os.path.splitext(os.path.basename(videoName))[0]
        # print(rallyName)
        video_frame_folder = os.path.join(frame_folder, rallyName)
        if not os.path.isdir(video_frame_folder):
            os.makedirs(video_frame_folder)
            cap = cv2.VideoCapture(videoName)
            success, count = True, 0
            success, image = cap.read()
            while success:
                cv2.imwrite(os.path.join(video_frame_folder, '{}.png'.format(count)) , image)
                count += 1
                success, image = cap.read()

print('finish')