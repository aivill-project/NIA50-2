import calib_logic as calib
import calib_parser
import utils
import visualizer
import open3d as o3d
import cuboid

"""
    ## !pip install opencv-python==3.4.17.61
    ## !pip install open3d

    ## core_dependencies: 
        ## python ver.__3.7.13 
        ## opencv ver.__3.4.17 
        ## open3d ver.__0.15.1 
        ## scipy ver.__1.7.3

    direction:
        1: x>0
        2: x<0
        3: y>0
        4: y<0
"""

## ==================CONFIG PATHS==================
lidar_conf_pth1 = "./calib_conf\config01.yaml"  ## config 파일 라이다  
radar_conf_pth = "calib_conf\config_radar.yaml" ## config 파일 레이다 

## ==================FILE DIRECTORIES==================
lidar_dir = "./dataset/lidar/"                  ## 라이다 pcd파일 디렉토리
radar_dir = "./dataset/radar/"                  ## 레이다 pcd파일 디렉토리
img_dir = "./dataset/camera/01/"                ## 이미지 파일 디렉토리

## ==================OUTPUT DIRECTORY==================
out_dir = "./output/camera/01/"                 ## output 디렉토리



## ==================CONSTANTS==================
DIRECTION = 4                                   ## cut_pcd 함수를 활용하여 알아내야 함
LIDAR_ROT = 'quat'                              ## config 파일 상 라이다 rotation matrix의 표현 방식
RADAR_ROT = 'euler_deg'                         ## config 파일 상 레이다 rotation matrix의 표현 방식


## ==================VARIABLES==================
START = 0                                       ## overlay할 파일 시작 번호
END = 1                                         ## overlay할 파일 끝 번호




# ##// 라이다만 오버레이
# calib.overlay_lidar(lidar_conf_pth1, lidar_dir, img_dir, out_dir, 
#                     direction=DIRECTION, rot_dscr_lidar='quat', start=START, end=END, show=False)

##// 레이다만 오버레이
#calib.overlay_radar(lidar_conf_pth1, radar_conf_pth, radar_dir, img_dir, out_dir, 'quat', 'euler_deg', 0, 1, show=False, to_calib=True)

##// 라이다 레이다 둘 다 오버레이
# calib.overlay_fusion(lidar_conf_pth1, radar_conf_pth, lidar_dir, radar_dir, img_dir, out_dir,
#                     DIRECTION, LIDAR_ROT, RADAR_ROT, START, END, show=True, to_calib=True)

##// 오버레이 한 이미지들로 동영상 만들기
# utils.make_vid("./output/vid_fusion.mp4","./output/camera/01/",0,600)


lidar_pth = r"C:\aivill_sey_ryeong\labeling\라벨링툴\클립_예시\48_1\Normal\Day\Clip_00035\Lidar/481_ND_00035_LR_001.pcd"

radar_pth = r"C:\aivill_sey_ryeong\labeling\라벨링툴\클립_예시\48_1\Normal\Day\Clip_00035\Radar\RadarFront/481_ND_00035_RF_001.pcd"
radar_conf_pth = r"C:\aivill_sey_ryeong\labeling\라벨링툴\클립_예시\48_1\Normal\Day\Clip_00035\calib\Lidar_radar_calib/conf.yaml" ## config 파일 레이다 
# # radar_conf_pth = r"C:\aivill_sey_ryeong\labeling\라벨링툴\클립_예시\48_1\Normal\Day\Clip_00035\calib\Lidar_radar_calib/conf_mbt.yaml" ## config 파일 레이다 
# radar_conf_pth = r"C:\aivill_sey_ryeong\labeling\라벨링툴\클립_예시\48_1\Normal\Day\Clip_00035\calib\Lidar_radar_calib/conf_srs.yaml" ## config 파일 레이다 


LIDAR_ROT = 'quat'                              ## config 파일 상 라이다 rotation matrix의 표현 방식
RADAR_ROT = 'euler_deg'                         ## config 파일 상 레이다 rotation matrix의 표현 방식

# utils.show_lidar(r"")

# utils.show_fusion(lidar_pth, radar_pth, radar_conf_pth, RADAR_ROT, to_calib=True)

# utils.show_fusion(lidar_pth, radar_pth, conf_dir, 'euler_deg', True)
# out_dir = "./output/radar_calibed/"
# before = calib_parser.parse_pcd(r"dataset_sample\radar\000599.pcd")
# utils.make_calib_radar(radar_conf_pth, radar_dir, out_dir, 'euler_deg', 0, 50)
# after = calib_parser.parse_pcd(r"output\radar_calibed\calibed_000599.pcd")

# print(before)
# print("\n")
# print(after)


lidar_pth = r"dataset_sample\lidar\000599.pcd"
img_pth = r"dataset_sample\camera\01\000599.jpg"


# # center0 = [2.7, -5 , -1.7]
center0 = [4, -4 , 0]
size0 = [4, 4, 4]
cube0 = cuboid.Cuboid(center0, size0)
rot = [0.0, 0.0, 0.0]
cube0.set_rotation(rot)


# center1 = [-0.5, -27.0 , -1.7]
# size1 = [1.6, 4, 1.3]


# cube1 = cuboid.Cuboid(center1, size1)



cube_list = [cube0]
# utils.show_cuboid(lidar_pth, cube_list)

calib.overlay_cuboid(lidar_conf_pth1, img_pth, cube_list, out_dir, 
                     DIRECTION, LIDAR_ROT)





# bin_pth = r"C:\Users\Admin\Downloads/1_1_1_20210906_019_00000000.bin"

# bin_pcd = calib_parser.parse_pcd(bin_pth)

# bin_pcd = visualizer.cut_pcd(bin_pcd, 2)

# point_cloud = o3d.geometry.PointCloud()
# point_cloud.points = o3d.utility.Vector3dVector(bin_pcd) # array_of_points.shape = (N,3)
# #point_cloud.colors = o3d.utility.Vector3dVector(lidar_color) # array_of_colors.shape = (N,3)
# o3d.visualization.draw_geometries([point_cloud])        



"""
디버깅: 화면 범위 벗어날 때: cv2 텍스트, 사각형 넣을 때 문제
cv2.error: OpenCV(3.4.17) D:\a\opencv-python\opencv-python\opencv\modules\calib3d\src\calibration.cpp:552: error: (-5:Bad argument) 
One of required arguments is not a valid matrix in function 'cvProjectPoints2'
"""