import pandas as pd
import argparse
import os
from glob import glob

parser = argparse.ArgumentParser(description = 'Detect_fast')
parser.add_argument('--dataset_folder', type=str, required=True, help = 'Dataset Folder (Include Different Locations)')
parser.add_argument('--locations', type=str, nargs='+', required=True, help = 'Example: --location EC234 nctu_old_gym profession_dataset (Use 3 locations)')
args = parser.parse_args()

dataset = args.dataset_folder
location = []

for l in args.locations:
    if os.path.isdir(os.path.join(dataset, l)):
        train_path = glob(os.path.join(dataset, l, 'frame', '*'))
        for i in range(len(train_path)):
            train_path[i] = os.path.basename(train_path[i])
        for p in train_path:
            labelPath = os.path.join(dataset, l, 'csv', p+'_ball.csv')
            df = pd.read_csv(labelPath)
            if 'Fast' in df.columns:
                df = df.drop(['Fast'], axis=1)
            Fast = []
            for i in range(len(df)):
                if i == 0 or (int(df.loc[i, 'X'])==0 and int(df.loc[i, 'Y'])==0):
                    Fast.append(0)
                    continue
                if abs((int(df.loc[i, 'X'])-int(df.loc[i-1, 'X']))**2 + (int(df.loc[i, 'Y'])-int(df.loc[i-1, 'Y']))**2) > 40000 and int(df.loc[i-1, 'X'])!=0 and int(df.loc[i-1, 'Y'])!=0:
                    Fast[i-1] = 1
                    Fast.append(1)
                else:
                    Fast.append(0)
            df_f = pd.DataFrame(Fast, columns = ['Fast'])
            df = df.join(df_f)
            df.to_csv(labelPath, index = False)