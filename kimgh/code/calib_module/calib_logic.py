import numpy as np
import os
import cv2
import re

import calib_parser
import visualizer

PATTERN = r'[^0-9]'


##//core logic =============================

def rotate_translate_pcd(pcd_arr, rmat, tvec):
    ##// pcd 데이터 회전, 평행이동
    ##// pcd_arr: (N, 3) pcd array
    ##// rmat: (3, 3) rotation matrix
    ##// tvec: (1, 3) translation vector
    ret = []
    for pt in pcd_arr:
        v = rmat @ pt + tvec
        ret.append(v)
    ret = np.asarray(ret)
    return ret


def quaternion_rotation_matrix(Q):
    #// ./calib.txt의 Quaternion 값을 rotation matrix로 역산
    q0 = Q[0]
    q1 = Q[1]
    q2 = Q[2]
    q3 = Q[3]
    
    r00 = 2 * (q0 * q0 + q1 * q1) - 1
    r01 = 2 * (q1 * q2 - q0 * q3)
    r02 = 2 * (q1 * q3 + q0 * q2)
    
    r10 = 2 * (q1 * q2 + q0 * q3)
    r11 = 2 * (q0 * q0 + q2 * q2) - 1
    r12 = 2 * (q2 * q3 - q0 * q1)
     
    r20 = 2 * (q1 * q3 - q0 * q2)
    r21 = 2 * (q2 * q3 + q0 * q1)
    r22 = 2 * (q0 * q0 + q3 * q3) - 1
     
    rot_matrix = np.array([[r00, r01, r02],
                           [r10, r11, r12],
                           [r20, r21, r22]])
    return rot_matrix



##//overlay 함수 =============================

def overlay_lidar(conf_pth, lidar_dir, img_dir, out_dir, direction, rot_dscr_lidar, start, end, show):
    K, D, rvec_lidar, tvec_lidar = calib_parser.parse_conf_lidar(conf_pth, rot_dscr_lidar)
    
    #img_dir = img_dir + "/" + str(direction).zfill(2) + "/"
    lidar_files = os.listdir(lidar_dir)
    img_files = os.listdir(img_dir)
    
    for i in range(start, end):
        lidar_pth = f'{lidar_dir}/{lidar_files[i]}' 
        
        lidar_np = calib_parser.parse_pcd(lidar_pth)
        
        lidar_np = visualizer.cut_pcd(lidar_np, direction)
        print(f'rvec: {rvec_lidar}, tvec: {tvec_lidar}')
        
        res_loc,_ = cv2.projectPoints(lidar_np, rvec_lidar, tvec_lidar, K, D)
        depth = visualizer.get_depth(lidar_np)
        
        name = re.sub(PATTERN, '', lidar_files[i])
        img_pth = f'{img_dir}/{img_files[i]}'
        out_pth = f'{out_dir}/{name}_'
        visualizer.make_img_lidar(img_pth, out_pth, res_loc, depth, show)


def overlay_radar(lidar_conf_pth, radar_conf_pth, radar_dir, img_dir, out_dir, rot_dscr_lidar, rot_dscr_radar, start, end, show=False, to_calib=False):
    K, D, rvec_lidar, tvec_lidar = calib_parser.parse_conf_lidar(lidar_conf_pth, rot_dscr_lidar)
    rmat_radar, tvec_radar = calib_parser.parse_conf_radar(radar_conf_pth, rot_dscr_radar)
    
    #img_dir = img_dir + "/" + str(1).zfill(2) + "/"
    radar_files = os.listdir(radar_dir)
    img_files = os.listdir(img_dir)
    
    for i in range(start,end):
        
        radar_pth = f'{radar_dir}/{radar_files[i]}'
        radar_np = calib_parser.parse_pcd(radar_pth)
        if to_calib: radar_np = rotate_translate_pcd(radar_np, rmat_radar, tvec_radar)
        print(f'lidar rvec: {rvec_lidar}, lidar tvec: {tvec_lidar}')
        print(f'radar rmat: {rmat_radar}, radar tvec: {tvec_radar}')
        res_loc,_ = cv2.projectPoints(radar_np, rvec_lidar, tvec_lidar, K, D)
        depth = visualizer.get_depth(radar_np)
        
        name = re.sub(PATTERN, '', radar_files[i])
        img_pth = f'{img_dir}/{img_files[i]}'
        out_pth = f'{out_dir}/{name}_'
        visualizer.make_img_radar(img_pth, out_pth, res_loc, depth, show)
    

def overlay_fusion(lidar_conf_pth, radar_conf_pth, lidar_dir, radar_dir, img_dir, out_dir, direction, rot_dscr_lidar, rot_dscr_radar, start, end, show=False, to_calib=False):
    K, D, rvec_lidar, tvec_lidar = calib_parser.parse_conf_lidar(lidar_conf_pth, rot_dscr_lidar)
    rmat_radar, tvec_radar = calib_parser.parse_conf_radar(radar_conf_pth, rot_dscr_radar)
    
    #img_dir = img_dir + "/" + str(1).zfill(2) + "/"
    lidar_files = os.listdir(lidar_dir)
    radar_files = os.listdir(radar_dir)
    img_files = os.listdir(img_dir)
    
    
    
    for i in range(start, end):
        lidar_pth = f'{lidar_dir}/{lidar_files[i]}' 
        lidar_np = calib_parser.parse_pcd(lidar_pth)
        lidar_np = visualizer.cut_pcd(lidar_np, direction)
        res_loc,_ = cv2.projectPoints(lidar_np, rvec_lidar, tvec_lidar, K, D)
        depth = visualizer.get_depth(lidar_np)
        
        radar_pth = f'{radar_dir}/{radar_files[i]}'
        radar_np = calib_parser.parse_pcd(radar_pth)
        
        print(f'lidar rvec: {rvec_lidar}, lidar tvec: {tvec_lidar}')
        print(f'radar rmat: {rmat_radar}, radar tvec: {tvec_radar}')
        
        if to_calib: radar_np = rotate_translate_pcd(radar_np, rmat_radar, tvec_radar)
        res_loc_radar,_ = cv2.projectPoints(radar_np, rvec_lidar, tvec_lidar, K, D)
        
        name = re.sub(PATTERN, '', lidar_files[i])
        img_pth = f'{img_dir}/{img_files[i]}'
        out_pth = f'{out_dir}/{name}_'
        visualizer.make_img_fusion(img_pth, out_pth, res_loc, res_loc_radar, depth, show)


def overlay_cuboid(conf_pth, img_pth, cube_list, out_dir, direction, rot_dscr_lidar):
    K, D, rvec_lidar, tvec_lidar = calib_parser.parse_conf_lidar(conf_pth, rot_dscr_lidar)
    
    #img_dir = img_dir + "/" + str(direction).zfill(2) + "/"
        
    projected_cubes = []
    for cube in cube_list:
        X, Y, Z = cube.get_cuboid_arr()
        cube_arr = np.concatenate((X, Y, Z), axis=0)
        #cube_arr = visualizer.cut_pcd(cube_arr, direction)
        res_loc,_ = cv2.projectPoints(cube_arr, rvec_lidar, tvec_lidar, K, D)
        projected_cubes.append(res_loc)
        
    # X, Y, Z = visualizer.get_cuboid_arr(center, size)
    # cube_arr = np.concatenate((X, Y, Z), axis=0)
    # cube_arr = visualizer.cut_pcd(cube_arr, direction)
    # res_loc,_ = cv2.projectPoints(cube_arr, rvec_lidar, tvec_lidar, K, D)
        
    
    visualizer.make_img_cube(img_pth, out_dir, projected_cubes)


def make_2d_pts(lidar_np, direction, rvec_lidar, tvec_lidar, K, D):
    
    lidar_np = visualizer.cut_pcd(lidar_np, direction)
    
    res_loc,_ = cv2.projectPoints(lidar_np, rvec_lidar, tvec_lidar, K, D)
    
    return res_loc
