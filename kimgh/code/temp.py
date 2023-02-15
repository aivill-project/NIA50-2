import pandas as pd
import numpy as np
import os
import open3d as o3d
import pickle as pkl
import re
import json
import shutil
import glob
from collections import Counter
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import subprocess


def xyxy2xywhn(x, w=1920, h=1200, clip=False, eps=0.0):
    # Convert nx4 boxes from [x1, y1, x2, y2] to [x, y, w, h] normalized where xy1=top-left, xy2=bottom-right
    if clip:
        clip_boxes(x, (h - eps, w - eps))  # warning: inplace clip
    x = np.array(x).reshape(1, -1)
    y = np.copy(x)
    y[:, 0] = ((x[:, 0] + x[:, 2]) / 2) / w  # x center
    y[:, 1] = ((x[:, 1] + x[:, 3]) / 2) / h  # y center
    y[:, 2] = (x[:, 2] - x[:, 0]) / w  # width
    y[:, 3] = (x[:, 3] - x[:, 1]) / h  # height
    y = list(y.reshape(-1))
    return y

# rotation matrix
def roty(t, Rx=90/180*np.pi):
    ''' Rotation about the y-axis. '''
    c = np.cos(t)
    s = np.sin(t)
    
    X = np.array([[1, 0, 0],
                    [0, np.cos(Rx), -np.sin(Rx)],
                    [0, np.sin(Rx), np.cos(Rx)]])

    Z = np.array([[c, -s, 0],
                    [s, c, 0],
                    [0, 0, 1]])
    
    return np.matmul(Z, X)

def xyz2xyxy(x, y, z, l, w, h, rot_y, extrinsic, intrinsic):
    R = roty(rot_y)
   
    x_corners = [l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2];
    y_corners = [h / 2, h / 2, h / 2, h / 2, -h / 2, -h / 2, -h / 2, -h / 2];
    z_corners = [w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2];
    
    corners_3d = np.dot(R, np.vstack([x_corners, y_corners, z_corners]))
    corners_3d[0, :] = corners_3d[0, :] + x  # x
    corners_3d[1, :] = corners_3d[1, :] + y  # y
    corners_3d[2, :] = corners_3d[2, :] + z  # z
    corners_3d = np.vstack([corners_3d, [1, 1, 1, 1, 1, 1, 1, 1]])
    
    point2d = np.matmul(intrinsic, np.matmul(extrinsic, corners_3d))
    pointx = np.around(point2d/point2d[2])[0]
    pointy = np.around(point2d/point2d[2])[1]

    return min(pointx), min(pointy), max(pointx), max(pointy)


def main():
    dp = pd.read_csv('/data/NIA50/50-2/data/nia50_final/data_info.csv', index_col=0, dtype={'frame':object})
    # class 통합
    dp.loc[(dp['class']=='Car') | (dp['class']=='Light_Car') | (dp['class']=='Small_Car'), 'class'] = 'Car'
    dp.loc[(dp['class']=='SUV') | (dp['class']=='Van'), 'class'] = 'SUV_&_Van'
    dp.loc[(dp['class']=='Adult') | (dp['class']=='Kid') | (dp['class']=='Kickboard'), 'class'] = 'Person'
    dp.loc[(dp['class']=='Small_Truck') | (dp['class']=='Medium_Truck') | (dp['class']=='Large_Truck'), 'class'] = 'Truck'
    dp.loc[(dp['class']=='Mini_Bus') | (dp['class']=='Bus'), 'class'] = 'Bus'

    qt = dp[['class', 'l', 'w', 'h']].groupby('class').quantile(0.99)
    dp.loc[(dp['class']=='Special_Vehicle') & (dp['l']<=qt.loc['SUV_&_Van']['l']) & (dp['w']<=qt.loc['SUV_&_Van']['w']) & (dp['h']<=qt.loc['SUV_&_Van']['h']), 'class'] = 'SUV_&_Van'
    dp.loc[(dp['class']=='Special_Vehicle') & (dp['l']>qt.loc['SUV_&_Van']['l']) & (dp['w']>qt.loc['SUV_&_Van']['w']) & (dp['h']>qt.loc['SUV_&_Van']['h'])
            & (dp['l']<=qt.loc['Truck']['l']) & (dp['w']<=qt.loc['Truck']['w']) & (dp['h']<=qt.loc['Truck']['h']), 'class'] = 'Truck'

    src = '/data/NIA50/50-2/data/nia50_final/raw'
    dst = '/data/NIA50/50-2/data/nia50_final/pvrcnn_integ'

    os.makedirs(f'{dst}/labels', exist_ok=True)
    os.makedirs(f'{dst}/points', exist_ok=True)
    os.makedirs(f'{dst}/ImageSets', exist_ok=True)

    scenes = dp['scene'].unique()
    for scene in tqdm(scenes[2161:]):
        # dat_typs.append(re.findall('[a-zA-Z]+_[A-Z]_[A-Z]', scene)[0])

        # with open(f'{src}/{scene}/calib/camera/camera_0.json', 'r') as f:
        #     calib = json.load(f)

        # for typ in ['typ1', 'typ2', 'typ3', 'typ4']:
        #     if calib['extrinsic'] == calib_typ[typ]['calib']:
        #         mov_zpoint = calib_typ[typ]['mov_zpoint']

        frames = dp.loc[dp['scene']==scene, 'frame'].unique()
        for frame in frames:

            # make labels
            frame_data = dp.loc[(dp['scene']==scene) & (dp['frame']==frame)].copy()
            frame_data[['point_x', 'point_y', 'point_z', 'l', 'w', 'h', 'rot_y', 'class']].to_csv(f'{dst}/labels/{scene}_{frame}.txt', header=False, index=False, sep=' ')


    os.chdir('/data/NIA50/50-2/models/OpenPCDet_aivill')
    dataset = '/data/NIA50/50-2/data/nia50_final/pvrcnn_integ/nia50_pvrcnn_integ_data.yaml'
    make_info = f'python -m pcdet.datasets.custom.custom_dataset create_custom_infos {dataset}'
    subprocess.call(make_info, shell=True)

if __name__=='__main__':
    main()