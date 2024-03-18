import numpy as np
import os
import struct
from itertools import cycle
import open3d as o3d
import random
from pandas import DataFrame
from pyntcloud import PyntCloud



#     点云数组
def read_velodyne_bin(path):
    '''
    :param path:
    :return: homography matrix of the point cloud, N*3
    '''
    pc_list = []
    with open(path, 'rb') as f:
        content = f.read()
        pc_iter = struct.iter_unpack('ffff', content)
        for idx, point in enumerate(pc_iter):
            pc_list.append([point[0], point[1], point[2]])
    return np.asarray(pc_list, dtype=np.float32)


def point_cloud_to_instance(data):
    origin_points_df = DataFrame(data, columns=['x', 'y', 'z'])
    point_cloud_pynt = PyntCloud(origin_points_df)  # 将points的数据 存到结构体中
    point_cloud_o3d = point_cloud_pynt.to_instance("open3d", mesh=False)  # 实例化
    return point_cloud_o3d

def visulaize(scenes):
    color = cycle([[0, 0, 1], [0, 1, 0], [1, 0, 0]])

    for scene, c in zip(scenes, color):
        scene.paint_uniform_color(c)

    o3d.visualization.draw_geometries(scenes)


def visual_all(root_dir,data_dir):

    cat1 = os.listdir(root_dir)
    cat1 = cat1[0:]
    iteration_num1 = len(cat1)
    scenes = []
    color1 = cycle([[0, 0, 1], [0, 1, 0]])

    for cat_1,c1 in zip(cat1,color1):
        filename = os.path.join(root_dir, cat_1)
        # print('visualing file:', filename)
        origin_points = read_velodyne_bin(filename)  # 读取数据点
        # print("origin_points:", origin_points,origin_points.shape)
        point_cloud_o3d = point_cloud_to_instance(origin_points)
        point_cloud_o3d.paint_uniform_color(c1)
        # o3d.visualization.draw_geometries([point_cloud_o3d])  # 显示原始点云
        scenes.append(point_cloud_o3d)

    cat2 = os.listdir(data_dir)
    cat2 = cat2[0:]
    iteration_num2 = len(cat2)
    color2 = cycle([[1, 1, 0],[1, 0, 1], [0, 1, 1], [1, 0, 0]])

    for cat_2,c2 in zip(cat2,color2):
        filename = os.path.join(data_dir, cat_2)
        # print('visualing file:', filename)
        origin_points = read_velodyne_bin(filename)  # 读取数据点
        # print("origin_points:", origin_points,origin_points.shape)
        point_cloud_o3d = point_cloud_to_instance(origin_points)
        point_cloud_o3d.paint_uniform_color(c2)
        # o3d.visualization.draw_geometries([point_cloud_o3d])  # 显示原始点云
        scenes.append(point_cloud_o3d)

    print("scenes:",scenes)
    o3d.visualization.draw_geometries(scenes)

if __name__ == '__main__':
    visual_all('./data/demo_data/original/','./data/demo_data/interpolated/')