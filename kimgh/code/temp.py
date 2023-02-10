import open3d as o3d
import numpy as np
import cv2
import torch
import pandas as pd
import re
import pickle as pkl
import json
import glob
import shutil
import os
from collections import Counter
from sklearn.model_selection import train_test_split
from scipy.spatial.transform import Rotation as R


# 문자열 숫자리스트로 바꾸는 함수
def str2list(txt):
    txt = txt.replace('\n', '').split(',')
    txt = list(map(float, txt))
    
    return txt


# 리스트를 문자열로 바꾸는 함수
def list2str(list):
    list = ' '.join(map(str, list))
    
    return list


# alpha 구하는 공식
import math

def normalizeAngle(angle):
    result = angle % (2*math.pi)
    if result < -math.pi:
        result += 2*math.pi
    elif result > math.pi:
        result -= 2*math.pi
    return result

def cal_alpha_ori(x, z, ry):  
    angle = ry
    angle -= -math.atan2(z, x) -1.5*math.pi 
    alpha = normalizeAngle(angle)
    return alpha # -1.818032754845337
# cal_alpha_ori(2.5702, 9.7190, -1.5595)


# convert 2d coordinate
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
    # return  np.array([[c, 0, s],
    #                 [0, 1, 0],
    #                 [-s, 0, c]])

    X = np.array([[1, 0, 0],
                    [0, np.cos(Rx), -np.sin(Rx)],
                    [0, np.sin(Rx), np.cos(Rx)]])

    # X = np.eye(3)

    Z = np.array([[c, -s, 0],
                    [s, c, 0],
                    [0, 0, 1]])
    
    return np.matmul(Z, X)


# Z축 이동을 위해서 calib와 매칭하여 이동범위 지정
calibs = sorted(glob.glob('/data/NIA50/50-2/data/NIA50/nia50_all/raw/*/calib/camera/camera_0.json'))

calib_ls = []
scenes = []
for calib in calibs:
    scene = re.findall('[a-zA-Z0-9_]+', calib)[-5]
    with open(calib, 'r') as f:
        calib = json.load(f)
    if calib['extrinsic'] not in calib_ls:
        calib_ls.append(calib['extrinsic'])
        scenes.append(scene)
    
calib_typ = {'typ1': {'calib': calib_ls[0], 'mov_zpoint': 14},
             'typ2': {'calib': calib_ls[1], 'mov_zpoint': 13},
             'typ3': {'calib': calib_ls[2], 'mov_zpoint': 0},
             'typ4': {'calib': calib_ls[3], 'mov_zpoint': -20}}


# yolov5, pvrcnn inference 완료한 데이터
with open('/data/NIA50/50-2/data/NIA50/nia50_all/pvrcnn_allcat/ImageSets/test.txt', 'r') as f:
    scenes = [re.sub('\n', '', i)[:-5] for i in f.readlines()]
    scenes = sorted(list(set(scenes)))

source_path = '/data/NIA50/50-2/data/NIA50/nia50_all/raw'
save_path = '/data/NIA50/50-2/data/NIA50/nia50_all/deepfusionmot_allcat/'


# 2d_yolov5
save_2d_label = f'{save_path}/2D_yolov5/'
os.makedirs(save_2d_label, exist_ok=True)

w = 1920
h = 1200
for scene in scenes:
    labels = sorted(glob.glob(f'/data/NIA50/50-2/result/yolov5_allcat/test/exp/labels/{scene}*.txt'))

    label_df = pd.DataFrame()
    for label in labels:
        # with open(label, 'r') as f:
        #     bbox = [re.sub('\n', '', j) for j in f.readlines()]
        frame_df = pd.read_csv(label, header=None, sep=' ')
        frame_df.columns = ['frame', 'x', 'y', 'w', 'h', 'conf']
        frame_df['frame'] = int(re.findall('[0-9]+', label)[-1])
        frame_df['xmin'] = (frame_df['x'] - frame_df['w']/2) * w
        frame_df['ymin'] = (frame_df['y'] - frame_df['h']/2) * h
        frame_df['xmax'] = (frame_df['x'] + frame_df['w']/2) * w
        frame_df['ymax'] = (frame_df['y'] + frame_df['h']/2) * h

        label_df = pd.concat((label_df, frame_df[['frame', 'xmin', 'ymin', 'xmax', 'ymax', 'conf']]), axis=0)
    label_df.to_csv(save_2d_label+f'{scene}.txt', index=None, header=None, sep=',')


# 3d_pvrcnn
save_3d_label = save_path + '/3D_pvrcnn/'
os.makedirs(save_3d_label, exist_ok=True)

with open('/data/NIA50/50-2/result/pvrcnn_allcat/test/eval/result.pkl', 'rb') as f:
    results = pkl.load(f)

for scene in scenes:

    label_df = pd.DataFrame()
    for result in results:
        frame_df = pd.DataFrame(columns=['frame_id', 'type', 'x1', 'y1', 'x2', 'y2', 'score', 'h', 'w', 'l', 'x', 'y', 'z', 'rot_y'])
        if scene in result['frame_id']:
            frame_df['type'] = result['pred_labels']
            frame_df['score'] = result['score']
            frame_df['h'] = result['boxes_lidar'][:, 5]
            frame_df['w'] = result['boxes_lidar'][:, 4]
            frame_df['l'] = result['boxes_lidar'][:, 3]
            frame_df['x'] = result['boxes_lidar'][:, 0]
            frame_df['y'] = result['boxes_lidar'][:, 1]
            frame_df['z'] = result['boxes_lidar'][:, 2]
            frame_df['rot_y'] = result['boxes_lidar'][:, 6]
            frame_df['frame_id'] = result['frame_id']

        label_df = pd.concat([label_df, frame_df], axis=0)

    label_df['scene'] = label_df['frame_id'].apply(lambda x: x[:-5])
    label_df['frame'] = label_df['frame_id'].apply(lambda x: int(x[-4:]))
    # label_df = label_df.loc[(-75.2 < label_df['x']) & (label_df['x'] < 75.2) & (-150.4 < label_df['y']) & (label_df['y'] < 0) & (-4 < label_df['z']) & (label_df['z'] < 8)]

    with open(save_path+f'calib/{scene}.txt') as f:
        calib = [re.sub('\n', '', i) for i in f.readlines()]
    intrinsic = np.asarray(calib[2].split(' ')[1:], dtype=float).reshape(3, 4)
    extrinsic = np.asarray(calib[5].split(' ')[1:], dtype=float).reshape(3, 4)
    extrinsic = np.vstack([extrinsic, [0, 0, 0, 1]])
    # scene_df = label_df.loc[label_df['scene']==scene]

    minx = []
    miny = []
    maxx = []
    maxy = []
    for i in np.arange(len(label_df)):
        obj = label_df.iloc[i]

        R = roty(obj['rot_y'])
        x = obj['x']
        y = obj['y']
        z = obj['z']
        l = obj['l']
        w = obj['w']
        h = obj['h']
        
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
        pointx[pointx > 1920] = 1920
        pointx[pointx < 0] = 0
        pointy[pointy > 1200] = 1200
        pointy[pointy < 0] = 0

        minx.append(min(pointx))
        miny.append(min(pointy))
        maxx.append(max(pointx))
        maxy.append(max(pointy))

    label_df[['x1', 'y1', 'x2', 'y2']] = np.asarray((minx, miny, maxx, maxy)).T
    label_df['alpha'] = 0
    # label_df = label_df.loc[(label_df['x2']-label_df['x1']!=0) & (label_df['y2']-label_df['y1']!=0)]
    label_df = label_df[['frame', 'type', 'x1', 'y1', 'x2', 'y2', 'score', 'h', 'w', 'l', 'x', 'y', 'z', 'rot_y', 'alpha']]
    label_df.to_csv(save_3d_label+f'{scene}.txt', index=None, header=None, sep=',')

    # scene_df[['x1', 'y1', 'x2', 'y2']] = np.asarray((minx, miny, maxx, maxy)).T
    # scene_df['alpha'] = 0
    # scene_df = scene_df[['frame', 'type', 'x1', 'y1', 'x2', 'y2', 'score', 'h', 'w', 'l', 'x', 'y', 'z', 'rot_y', 'alpha']]
    # scene_df.to_csv(save_3d_label+f'{scene}.txt', index=None, header=None, sep=',')