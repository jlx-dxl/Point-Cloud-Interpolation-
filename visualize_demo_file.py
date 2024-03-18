import numpy as np
import os
from itertools import cycle
import open3d as o3d

from Utils.Visualize import PcdsVisualizer

rootdir = './Demos/demo_0519/test2/Inputs/'
files = os.listdir(rootdir)
print(files)

visualizer = PcdsVisualizer(if_save=True, if_show=False, if_down_sample=True, npoints=30000,
                            view_point_json_file='zoom-test2.json',point_size=4.0)

for file in files:
    if file == 'key_1.bin':
        filename = os.path.join(rootdir, file)
        pcd = visualizer.read_bin_pc_fps_5(filename)
        visualizer.add_to_vis(pcd, [0, 0.8, 0])   # 绿色
    if file == 'key_2.bin':
        filename = os.path.join(rootdir, file)
        pcd = visualizer.read_bin_pc_fps_5(filename)
        visualizer.add_to_vis(pcd, [0, 0, 0.8])   # 蓝色

# 时序：红，橙，黄，紫
color_list = [[0.8, 0, 0], [0.8, 0.4, 0], [0.6, 0.6, 0],[0.6, 0, 0.6]]

# # see GTs
#
# for file in files:
#     for i, time in enumerate(['0.2', '0.4', '0.6', '0.8']):
#         if file == ('gt_' + time + '.bin'):
#             filename = os.path.join(rootdir, file)
#             print(filename)
#             pcd = visualizer.read_bin_pc_fps_5(filename,jitter=0.03)
#             visualizer.add_to_vis(pcd, color_list[i])

# # check fwd & bwd
# for file in files:
#     for i, time in enumerate(['1', '2', '3']):
#         if file == ('forward_' + time + '.bin'):
#             filename = os.path.join(rootdir, file)
#             print(filename)
#             pcd = visualizer.read_bin_pc_fps_5(filename)
#             visualizer.add_to_vis(pcd, color_list[i])
#         elif file == ('backward_' + time + '.bin'):
#             filename = os.path.join(rootdir, file)
#             print(filename)
#             pcd = visualizer.read_bin_pc_fps_5(filename)
#             visualizer.add_to_vis(pcd, color_list[i])

#
# see results

# field = 0

# rootdir = './Demos/demo_0519/test2/result_field_0_10/'
# files = os.listdir(rootdir)
# print(files)
#
# for file in files:
#     for i, time in enumerate(['0.2', '0.4', '0.6', '0.8']):
#         if file == ('result_' + time + '.bin'):
#             filename = os.path.join(rootdir, file)
#             print(filename)
#             pcd = visualizer.read_bin_pc_fps_3(filename)
#             visualizer.add_to_vis(pcd, color_list[i])

#
# # field = 1
#
# rootdir = './Demos/demo_0519/test2/result_field_1_1/'
# files = os.listdir(rootdir)
# print(files)
#
# for file in files:
#     for i, time in enumerate(['0.2', '0.4', '0.6', '0.8']):
#         if file == ('result_' + time + '.bin'):
#             filename = os.path.join(rootdir, file)
#             print(filename)
#             pcd = visualizer.read_bin_pc_fps_3(filename)
#             visualizer.add_to_vis(pcd, color_list[i])

# # field = 2
#
# rootdir = './Demos/demo_0519/test1/result_field_2/'
# files = os.listdir(rootdir)
# print(files)
#
# for file in files:
#     for i, time in enumerate(['0.2', '0.4', '0.6', '0.8']):
#         if file == ('result_' + time + '.bin'):
#             filename = os.path.join(rootdir, file)
#             print(filename)
#             pcd = visualizer.read_bin_pc_fps_3(filename)
#             visualizer.add_to_vis(pcd, color_list[i])

# # field = 3
#
# rootdir = './Demos/demo_0519/test1/result_field_3/'
# files = os.listdir(rootdir)
# print(files)
#
# for file in files:
#     for i, time in enumerate(['0.2', '0.4', '0.6', '0.8']):
#     # for i, time in enumerate(['0.4']):
#         if file == ('result_' + time + '.bin'):
#             filename = os.path.join(rootdir, file)
#             print(filename)
#             pcd = visualizer.read_bin_pc_fps_3(filename)
#             visualizer.add_to_vis(pcd, color_list[i])

# pointinet

rootdir = './Demos/demo_0519/test2/result_pointinet/'
files = os.listdir(rootdir)
print(files)

for file in files:
    for i, time in enumerate(['0.2', '0.4', '0.6', '0.8']):
    # for i, time in enumerate(['0.4']):
        if file == ('result_' + time + '.bin'):
            filename = os.path.join(rootdir, file)
            print(filename)
            pcd = visualizer.read_bin_pc_fps_3(filename)
            visualizer.add_to_vis(pcd, color_list[i])
#
visualizer.show_and_save('./zoomtest-pointinet.png')
visualizer.clear()
