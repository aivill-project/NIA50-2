import pandas as pd
import pathlib as pl
import json
import numpy as np
from glob import glob
import os
import argparse
from tqdm import tqdm
from collections import Counter
import sys

# 2D
# import glob
# import json
# import shutil
# import numpy as np
import re
import shutil
from sklearn.model_selection import train_test_split

# 3D
# import re
# import shutil
# import glob
# import numpy as np
# import pandas as pd
import open3d as o3d
import cv2
import pickle as pkl
# from sklearn.model_selection import train_test_split
# from scipy.spatial.transform import Rotation as R


CLASSES = ["Car", "Small_Car", "Light_Car", "SUV", "Van", "Small_Truck", "Medium_Truck", "Large_Truck", "Bus", "Mini_Bus", "Special_Vehicle","Two_Wheeler", "Kickboard", "Adult", "Kid"]

# check the python version. if the python version is under 3.11, print the warning message
if int(str(sys.version_info[0]) + str(sys.version_info[1])) < 311:
    # print("Warning: The python version is under 3.11. The program works faster if the python version is >= 3.11")
    print("경고: 파이썬 버전이 3.11보다 낮습니다. 파이썬 버전이 3.11 이상이면 프로그램이 더 빠르게 작동합니다.")

def get_json_data_3d(json_files_path: list, save_data_info: str = None, save_stat_to: str = None):
    """if there is a file on save_data_info or save_stat_to ask the user if they want to overwrite the file

    Args:
        json_files_path (list): 레이블 파일 상위 경로
        save_data_info (str, optional): 데이터 객체 수 등 저장 경로. Defaults "/info.csv".
        save_stat_to (str, optional): 데이터 객체 별 통계 csv 저장 경로. Defaults to "/stat.csv".

    Returns:
        _type_: dp_stat, class_statistic
    """
    
    #클래스 종류 및 개수
    def get_class_simple_statistic(dp_stat: pd.DataFrame):
        classes = Counter(dp_stat["class"]) # e.g. {"Car": 19000,, "SUV": 5000}
        print(classes)
        # print f"classes: {classes}" for every 3 classes and goes to the next line
        print(f"classes: {', '.join([f'{key}: {value}' for key, value in classes.items()])}")
        return classes
    
    if os.path.isfile(save_data_info):
        answer_info = input(f"Warning: The file {save_data_info} already exists. Do you want to overwrite it? (y/n):")
    if os.path.isfile(save_stat_to):
        answer_stat = input(f"Warning: The file {save_stat_to} already exists. Do you want to overwrite it? (y/n): ")
    
    if answer_info.lower() != "n":
        # dp = pd.DataFrame(columns=["class", "position x", "position y", "position z", "rotation", "scale l", "scale w", "scale h", "file_name"])
        print("Collecting 3d label data...")
        print("3D 레이블 데이터 수집 중...")

        dp_ls = []
        for j, file in enumerate(tqdm(json_files_path)):
            globals()[f'dp{j}'] = pd.DataFrame()

            with open(file, 'r') as f:
                json_data = json.load(f)
                for i in range(len(json_data)):
                    file_path = file.split("/")
                    # file_name is among the file_path if the attribute contains "_Suwon" or "_Pangyo" add this attribute and the last attribute with underbar "_" without extension
                    file_name = "_" + file_path[-2] + "_" + file_path[-1].split(".")[0] if "_Suwon" in file_path or "_Pangyo" in file_path else file_path[-1].split(".")[0]
                    row = [json_data[i]["obj_type"], 
                        json_data[i]["psr"]["position"]["x"], 
                        json_data[i]["psr"]["position"]["y"], 
                        json_data[i]["psr"]["position"]["z"], 
                        json_data[i]["psr"]["rotation"]['z'], 
                        json_data[i]["psr"]["scale"]["x"], 
                        json_data[i]["psr"]["scale"]["y"], 
                        json_data[i]["psr"]["scale"]["z"], 
                        file_name]
                    frame_data = pd.DataFrame([row], columns=["class", "position x", "position y", "position z", "rotation", "scale l", "scale w", "scale h", "file_name"])
                    globals()[f'dp{j}'] = pd.concat([globals()[f'dp{j}'], frame_data], axis=0)
            dp_ls.append(globals()[f'dp{j}'])
            
        dp = pd.concat(dp_ls)

        # make the dp's index to descending order
        dp = dp.reset_index(drop=True)
        dp.to_csv(save_data_info, index=False)
        # print(f"Saved the data info to '{save_data_info}'")
        print(f"데이터 정보를 '{save_data_info}'에 저장하였습니다.")
    else:
        dp_stat = pd.read_csv(save_data_info)
    
    classes = get_class_simple_statistic(dp_stat)
            
    if answer_stat.lower() != "n":
        dp_stat = dp.copy()
        
        # remove the rows if class is not in the CLASSES
        dp_stat = dp_stat[dp_stat["class"].isin(CLASSES)]

        # remove the rows if scale l, w is over 20
        dp_stat = dp_stat[(dp_stat["scale l"] <= 20) & (dp_stat["scale w"] <= 20) & (dp_stat["scale h"] <= 20)]
        dp_stat = dp_stat.reset_index(drop=True)

        # if the value is under 0, change it to positive value
        # if the value is 0, remove it
        dp_stat["scale l"] = dp_stat["scale l"].apply(lambda x: abs(x) if x != 0 else np.nan)
        dp_stat["scale w"] = dp_stat["scale w"].apply(lambda x: abs(x) if x != 0 else np.nan)
        dp_stat["scale h"] = dp_stat["scale h"].apply(lambda x: abs(x) if x != 0 else np.nan)
        dp_stat = dp_stat.dropna(axis=0)
        dp_stat = dp_stat.reset_index(drop=True)


        get_class_simple_statistic(dp_stat)
        
        # average of scale l, w, h for each class. Make this to pandas dataframe
        # the row's index is class name, and the column's name is average of scale l, w, h
        class_scal_avg = pd.DataFrame(columns=["scale l", "scale w", "scale h"])
        for c in classes.keys():
            class_scal_avg.loc[c] = [dp_stat[dp_stat["class"] == c]["scale l"].mean(), dp_stat[dp_stat["class"] == c]["scale w"].mean(), dp_stat[dp_stat["class"] == c]["scale h"].mean()]
        
        # min & max of position x, y, z and scale l, w, h for each class. Make this to pandas dataframe
        # the row's index is class name, and the column's name is min & max of position x, y, z and scale l, w, h
        # each values is .3f
        class_statistic = pd.DataFrame(columns=["count", "position x min", "position x max", "position y min", "position y max", "position z min", "position z max", "scale l min", "scale l max", "scale w min", "scale w max", "scale h min", "scale h max", "scale l avg", "scale w avg", "scale h avg"])
        for c in classes.keys():
            class_statistic.loc[c] = [classes[c],
                                    f"{min(dp_stat[dp_stat['class'] == c]['position x']):.3f}", 
                                    f"{max(dp_stat[dp_stat['class'] == c]['position x']):.3f}", 
                                    f"{min(dp_stat[dp_stat['class'] == c]['position y']):.3f}", 
                                    f"{max(dp_stat[dp_stat['class'] == c]['position y']):.3f}", 
                                    f"{min(dp_stat[dp_stat['class'] == c]['position z']):.3f}", 
                                    f"{max(dp_stat[dp_stat['class'] == c]['position z']):.3f}", 
                                    f"{min(dp_stat[dp_stat['class'] == c]['scale l']):.3f}", 
                                    f"{max(dp_stat[dp_stat['class'] == c]['scale l']):.3f}", 
                                    f"{min(dp_stat[dp_stat['class'] == c]['scale w']):.3f}", 
                                    f"{max(dp_stat[dp_stat['class'] == c]['scale w']):.3f}", 
                                    f"{min(dp_stat[dp_stat['class'] == c]['scale h']):.3f}", 
                                    f"{max(dp_stat[dp_stat['class'] == c]['scale h']):.3f}", 
                                    f"{class_scal_avg.loc[c]['scale l']:.3f}", 
                                    f"{class_scal_avg.loc[c]['scale w']:.3f}", 
                                    f"{class_scal_avg.loc[c]['scale h']:.3f}"]
            
        # save stat to csv file
        class_statistic.to_csv(save_stat_to)
        print(f"save stat to '{save_stat_to}'")
    else:
        class_statistic = pd.read_csv(save_stat_to)
        get_class_simple_statistic(dp_stat)
        
    
    return dp_stat, class_statistic


## 2D (yolov5)
# 라벨 좌표 형태 변경 함수
def xywh2xyxy(x):
    # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    x = np.array(x, dtype=float).reshape(1, -1)
    y = np.copy(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
    y = list(y.reshape(-1))
    return y

def xyxy2xywhn(x, w=1920, h=1200, clip=False, eps=0.0):
    # Convert nx4 boxes from [x1, y1, x2, y2] to [x, y, w, h] normalized where xy1=top-left, xy2=bottom-right
    if clip:
        clip_boxes(x, (h - eps, w - eps))  # warning: inplace clip
    x = np.array(x, dtype=float).reshape(1, -1)
    y = np.copy(x)
    y[:, 0] = ((x[:, 0] + x[:, 2]) / 2) / w  # x center
    y[:, 1] = ((x[:, 1] + x[:, 3]) / 2) / h  # y center
    y[:, 2] = (x[:, 2] - x[:, 0]) / w  # width
    y[:, 3] = (x[:, 3] - x[:, 1]) / h  # height
    y = list(y.reshape(-1))
    return y

# labels 생성
def mk_2d_labels(from_path: str, to_path: str):
    os.makedirs(f'{to_path}/labels', exist_ok=True)

    class_ls = {'Small_Car': 0,
                'Light_Car': 1,
                'Car': 2,
                'Van': 3,
                'SUV': 4,
                'Small_Truck': 5,
                'Medium_Truck': 6,
                'Large_Truck': 7,
                'Mini_Bus': 8,
                'Bus': 9,
                'Special_Vehicle': 10,
                'Two_Wheeler': 11,
                'Kickboard': 12,
                'Adult': 13,
                'Kid': 14}

    scenes = sorted(os.listdir(from_path))
    for scene in tqdm(scenes, desc='2D labels 생성 중'):
        labels = os.listdir(f'{from_path}/{scene}/2d_label')

        for label in labels:
            with open(f'{from_path}/{scene}/2d_label/{label}') as f:
                label_js = json.load(f)

            frame = label_js['filename'][:4]
            div_w, div_h = label_js['width']/1920, label_js['height']/1200
            objs = label_js['objects']
        
            labeling_ls = []
            for obj in objs:
                if obj['class'] in class_ls.keys():
                    cls_num = class_ls[obj['class']]
                    bbox_3d = obj['front'] + obj['back']

                    x = []
                    y = []
                    for cordi in bbox_3d:
                        x.append(cordi['x'])
                        y.append(cordi['y'])

                    x = np.asarray(x, dtype=float) / div_w
                    y = np.asarray(y, dtype=float) / div_h

                    x[x > 1920] = 1920
                    x[x < 0] = 0
                    y[y > 1200] = 1200
                    y[y < 0] = 0

                    xmin, ymin, xmax, ymax = min(x), min(y), max(x), max(y)
                    
                    bbox_2d = [xmin, ymin, xmax, ymax]
                    bbox_2d = xyxy2xywhn(bbox_2d)               

                    if bbox_2d[2]==0 or bbox_2d[3]==0:
                        continue
            
                    labeling = ' '.join(map(str, [cls_num] + bbox_2d))
                    labeling_ls.append(labeling)

            with open(f'{to_path}/labels/{scene}_{frame}'+'.txt', 'w') as f:
                f.write('\n'.join(labeling_ls))

# images 생성
def mk_2d_images(from_path: str, to_path: str):
    os.makedirs(f'{to_path}/images', exist_ok=True)

    image_ls = sorted(glob(f'{from_path}/*/camera/camera_0/*.jpg'))
    for image in tqdm(image_ls, desc='2D images 폴더 생성 중'):
        scene = re.findall('[a-zA-Z0-9_]+', image)[-5]
        frame = re.findall('[0-9]+', image)[-1]
        img_name = scene + '_' + frame + '.jpg'

        shutil.copyfile(image, f'{to_path}/images/{img_name}')

def mk_2d_imagesets(from_path: str, to_path: str):
    if os.path.isdir(f'{to_path}/images')==False:
        print('to_path 위치에 images 폴더가 존재하지 않습니다')
    else:
        os.makedirs(f'{to_path}/ImageSets', exist_ok=True)
        dat_typs = set([re.search('[A-Z]_[A-Z]', j).group() for j in os.listdir(from_path)])
        scenes = sorted(os.listdir(from_path))
        images = sorted(glob(f'{to_path}/images/*.jpg'))

        train_ls = []
        val_ls = []
        test_ls = []
        for dat_typ in tqdm(dat_typs, desc='2D ImageSets 생성 중'):
            scenes_typ = [scene for scene in scenes if dat_typ in scene]
            
            train_val, test = train_test_split(scenes_typ, test_size=0.2, random_state=44)
            train, val = train_test_split(train_val, test_size=0.2, random_state=44)

            for j in train:
                for image in images:
                    if j in image:
                        train_ls.append(image)

            for j in val:
                for image in images:
                    if j in image:
                        val_ls.append(image)

            for j in test:
                for image in images:
                    if j in image:
                        test_ls.append(image)

        with open(f'{to_path}/ImageSets/train.txt', 'w') as f:
            f.write('\n'.join(sorted(train_ls)))
            
        with open(f'{to_path}/ImageSets/val.txt', 'w') as f:
            f.write('\n'.join(sorted(val_ls)))

        with open(f'{to_path}/ImageSets/test.txt', 'w') as f:
            f.write('\n'.join(sorted(test_ls)))


## 3D (pvrcnn)
# Z축 이동을 위해서 calib와 매칭하여 이동범위 지정
def calib_check(from_path: str):
    calibs = sorted(glob(f'{from_path}/*/calib/camera/camera_0.json'))

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
    
    return calib_typ

# points 생성
def mk_3d_points(from_path: str, to_path: str):
    os.makedirs(f'{to_path}/points', exist_ok=True)

    points = sorted(glob(f'{from_path}/*/lidar/*.pcd'))

    for point in tqdm(points, desc='3D points 생성 중'):
        scene = re.findall('[a-zA-Z0-9_]+', point)[-4]
        frame = re.findall('[0-9]+', point)[-1]
        
        with open(f'{from_path}/{scene}/calib/camera/camera_0.json', 'r') as f:
            calib = json.load(f)
        
        pcd = o3d.t.io.read_point_cloud(point)
        positions = pcd.point.positions.numpy()
        intensity = pcd.point.intensity.numpy()

        calib_typ = calib_check(from_path)

        if calib['extrinsic'] == calib_typ['typ1']['calib']: positions[:, 2] += calib_typ['typ1']['mov_zpoint']
        elif calib['extrinsic'] == calib_typ['typ2']['calib']: positions[:, 2] += calib_typ['typ2']['mov_zpoint']
        elif calib['extrinsic'] == calib_typ['typ3']['calib']: positions[:, 2] += calib_typ['typ3']['mov_zpoint']
        elif calib['extrinsic'] == calib_typ['typ4']['calib']: positions[:, 2] += calib_typ['typ4']['mov_zpoint']

        pcd = np.concatenate((positions, intensity), axis = 1)

        np.save(f'{to_path}/points/{scene}_{frame}.npy', pcd)

# labels 생성
def mk_3d_labels(from_path: str, to_path:str):
    os.makedirs(f'{to_path}/labels', exist_ok=True)

    labels = sorted(glob(f'{from_path}/*/label/*.json'))

    x = []
    y = []
    z = []
    for label in tqdm(labels, desc='3D labels 생성 중'):
        scene = re.findall('[a-zA-Z_0-9_]+', label)[-4]
        frame = re.findall('[0-9]+', label)[-1]

        with open(f'{from_path}/{scene}/calib/camera/camera_0.json', 'r') as f:
            calib = json.load(f)
        
        with open(label, 'r') as f:
            label = json.load(f)
        
        labeling_ls = []
        for i in np.arange(len(label)):
            try:
                xyz = list(label[i]['psr']['position'].values())

                calib_typ = calib_check(from_path)

                if calib['extrinsic'] == calib_typ['typ1']['calib']: xyz[2] += calib_typ['typ1']['mov_zpoint']
                elif calib['extrinsic'] == calib_typ['typ2']['calib']: xyz[2] += calib_typ['typ2']['mov_zpoint']
                elif calib['extrinsic'] == calib_typ['typ3']['calib']: xyz[2] += calib_typ['typ3']['mov_zpoint']
                elif calib['extrinsic'] == calib_typ['typ4']['calib']: xyz[2] += calib_typ['typ4']['mov_zpoint']

                lwh = list(label[i]['psr']['scale'].values())
                rotation_y = [label[i]['psr']['rotation']['z']]
                class_ = label[i]['obj_type']
                # rename_class = [class_ls[class_]]
                # label_format = ' '.join(map(str, xyz + lwh + rotation_y + rename_class))
                label_format = ' '.join(map(str, xyz + lwh + rotation_y + [class_]))
                labeling_ls.append(label_format)

                # if xyz[0] < -130:
                #     print('x축', scene, frame)

                # if xyz[1] < -90:
                #     print('y축', scene, frame)

                # if xyz[2] > 8:
                #     print('z축', scene, frame)

                x.append(xyz[0])
                y.append(xyz[1])
                z.append(xyz[2])
            except:
                continue

        with open(f'{to_path}/labels/{scene}_{frame}.txt', 'w') as f:
            f.write('\n'.join(labeling_ls))

    #  객체 포인트 범위 확인
    print(f'x축 범위: {min(x)} ~ {max(x)}')
    print(f'y축 범위: {min(y)} ~ {max(y)}')
    print(f'z축 범위: {min(z)} ~ {max(z)}')

# imagesets 생성
def mk_3d_imagesets(from_path: str, to_path: str):
    if os.path.isdir(f'{to_path}/points')==False:
        print('pvrcnn data의 points 폴더가 존재하지 않습니다')
    else:
        os.makedirs(f'{to_path}/ImageSets', exist_ok=True)
        dat_typs = set([re.search('[A-Z]_[A-Z]', j).group() for j in os.listdir(from_path)])
        scenes = sorted(os.listdir(from_path))
        points = sorted(glob(f'{to_path}/points/*.npy'))

        train_ls = []
        val_ls = []
        test_ls = []
        for dat_typ in tqdm(dat_typs, desc='3D ImageSets 생성 중'):
            # images_typ = [image for image in images if dat_typ in image]
            scenes_typ = [scene for scene in scenes if dat_typ in scene]
            
            train_val, test = train_test_split(scenes_typ, test_size=0.2, random_state=44)
            train, val = train_test_split(train_val, test_size=0.2, random_state=44)

            for j in train:
                for point in points:
                    if j in point:
                        point = re.findall('[a-zA-Z0-9_]+', point)[-2]
                        train_ls.append(point)

            for j in val:
                for point in points:
                    if j in point:
                        point = re.findall('[a-zA-Z0-9_]+', point)[-2]
                        val_ls.append(point)

            for j in test:
                for point in points:
                    if j in point:
                        point = re.findall('[a-zA-Z0-9_]+', point)[-2]
                        test_ls.append(point)

        with open(f'{to_path}/ImageSets/train.txt', 'w') as f:
            f.write('\n'.join(sorted(train_ls)))
            
        with open(f'{to_path}/ImageSets/val.txt', 'w') as f:
            f.write('\n'.join(sorted(val_ls)))

        with open(f'{to_path}/ImageSets/test.txt', 'w') as f:
            f.write('\n'.join(sorted(test_ls)))
        


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--from_path', '-f', type=str, help='데이터셋 폴더 위치')
    parser.add_argument('--to_path', '-t', type=str, help='데이터를 옮길 폴더 위치', default=None)

    parser.add_argument('--all', '-a', action='store_true', help='모든 레이블, 이미지, PCD파일을 변환')

    parser.add_argument('--pcd', '-p', action='store_true', help='PCD 파일을 변환')
    parser.add_argument('--label_3d', '-l3', action='store_true', help='3D 레이블을 json에서 OpenPCD에 맞는 txt로 변환')
    parser.add_argument('--imagesets_3d', '-is3', action='store_true', help='3D 데이터를 train, val, test 세트로 분할')

    parser.add_argument('--img', '-i', action='store_true', help='이미지를 변환(이름만 변경)')
    parser.add_argument('--label_2d', '-l2', action='store_true', help='2D 레이블을 json에서 YOLOv5에 맞는 txt로 변환')
    parser.add_argument('--imagesets_2d', '-is2', action='store_true', help='2D 데이터를 train, val, test 세트로 분할')

    parser.add_argument('--stat_3d', '-s3', action='store_true', help='3D 레이블의 통계 정보 확인')
    parser.add_argument('--stat_output_3d', '-so3', type=str, help='3D 통계 정보 저장 위치', default="./stat_3d.csv")
    parser.add_argument('--stat_2d', '-s2', action='store_true', help='2D 레이블의 통계 정보 확인')
    parser.add_argument('--stat_output_2d', '-so2', type=str, help='2D 통계 정보 저장 위치', default="./stat_2d.csv")
    parser.add_argument('--info_output_3d', '-n3', type=str, help='3D 데이터 정보 저장 위치', default="./info_3d.csv")
    parser.add_argument('--info_output_2d', '-n2', type=str, help='2D 데이터 정보 저장 위치', default="./info_2d.csv")
    
    args = parser.parse_args()
    print(f"Dataset path: {args.from_path}")
    
    if args.all:
        # get 2D data
        mk_2d_images(args.from_path, args.to_path+'/yolov5')
        mk_2d_labels(args.from_path, args.to_path+'/yolov5')
        mk_2d_imagesets(args.from_path, args.to_path+'/yolov5')
        # get 3D data
        mk_3d_points(args.from_path, args.to_path+'/pvrcnn')
        mk_3d_labels(args.from_path, args.to_path+'/pvrcnn')
        mk_3d_imagesets(args.from_path, args.to_path+'/pvrcnn')

    if args.from_path is None:
        args.from_path = "./dataset"
    if args.to_path is None:
        args.to_path = "./output"

    # make the output folder
    if args.to_path is not None:
        os.makedirs(args.to_path, exist_ok=True)

    # get the file list of the dataset folder
    print("Collecting the 3d label file list of the dataset")
    json_paths = sorted([str(p) for p in pl.Path(args.from_path).rglob("*.json") if "calib" not in str(p) and "2d_label" not in str(p)])
    
    if args.stat_3d is True:
        # get the statistic of the dataset
        print("Collecting the statistic of the dataset")
        dp_info, class_statistic = get_json_data_3d(json_paths, save_data_info=args.info_output_3d, save_stat_to=args.stat_output_3d)
        print("Dataset statistic:")
        print(dp_info)
        print("Class statistic:")
        print(class_statistic)

    # get 2D data
    if args.img:
        mk_2d_images(args.from_path, args.to_path)

    if args.label_2d:
        mk_2d_labels(args.from_path, args.to_path)

    if args.imagesets_2d:
        mk_2d_imagesets(args.from_path, args.to_path)


    # get 3D data
    if args.pcd:
        mk_3d_points(args.from_path, args.to_path)

    if args.label_3d:
        mk_3d_labels(args.from_path, args.to_path)

    if args.imagesets_3d:
        mk_3d_imagesets(args.from_path, args.to_path)