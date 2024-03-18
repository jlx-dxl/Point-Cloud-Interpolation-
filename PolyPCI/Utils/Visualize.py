import numpy as np
import os
from itertools import cycle
import open3d as o3d


class PcdsVisualizer():
    def __init__(self, view_point_json_file, if_save=True, if_show=True, if_down_sample=True, npoints=16384 * 3,
                 window_name="Iterpolation-Result-Show",
                 background_color=np.array([0, 0, 0]), point_size=3.0):
        self.npoints = npoints
        self.window_name = window_name
        self.background_color = background_color
        self.point_size = point_size
        self.view_point_json_file = view_point_json_file
        self.if_save = if_save
        self.if_show = if_show
        self.if_down_sample = if_down_sample
        self.create_vis()

    def convert_to_o3d_from_numpy(self, points):
        # 将array格式的点云转换为open3d的点云格式,若直接用open3d打开的点云数据，则不用转换
        pcd = o3d.geometry.PointCloud()  # 传入3d点云格式
        pcd.points = o3d.utility.Vector3dVector(points)  # 转换格式
        return pcd

    def convert_to_o3d_from_tensor(self, points):
        points = points.clone().detach().cpu().numpy().reshape(-1, 3)
        # 将array格式的点云转换为open3d的点云格式,若直接用open3d打开的点云数据，则不用转换
        pcd = o3d.geometry.PointCloud()  # 传入3d点云格式
        pcd.points = o3d.utility.Vector3dVector(points)  # 转换格式
        return pcd

    def read_bin_pc_fps_5(self, path):
        points = np.fromfile(path, dtype=np.float32).reshape(-1, 5)
        points = points[:, :3]  # open3d 只需xyz 与pcl不同

        # 将array格式的点云转换为open3d的点云格式,若直接用open3d打开的点云数据，则不用转换
        pcd = o3d.geometry.PointCloud()  # 传入3d点云格式
        pcd.points = o3d.utility.Vector3dVector(points)  # 转换格式
        if self.if_down_sample == True:
            # fps downsample
            pcd = pcd.farthest_point_down_sample(self.npoints)
        return pcd

    def read_bin_pc_fps_3(self, path):
        points = np.fromfile(path, dtype=np.float32).reshape(-1, 3)

        # 将array格式的点云转换为open3d的点云格式,若直接用open3d打开的点云数据，则不用转换
        pcd = o3d.geometry.PointCloud()  # 传入3d点云格式
        pcd.points = o3d.utility.Vector3dVector(points)  # 转换格式
        if self.if_down_sample == True:
            # fps downsample
            pcd = pcd.farthest_point_down_sample(self.npoints)
        return pcd

    def create_vis(self):
        # 创建窗口对象
        self.vis = o3d.visualization.Visualizer()
        # 创建窗口,设置窗口名称
        if self.if_show:
            self.vis.create_window(window_name=self.window_name)
        else:
            self.vis.create_window(window_name=self.window_name, visible=False)
        # vis.set_full_screen(True)
        # 设置点云渲染参数
        self.opt = self.vis.get_render_option()
        # 设置背景色（这里为白色）
        self.opt.background_color = self.background_color
        # 设置渲染点的大小
        if self.view_point_json_file is not None:
            self.opt.point_size = self.point_size
            self.ctr = self.vis.get_view_control()
            self.parameters = o3d.io.read_pinhole_camera_parameters(self.view_point_json_file)

    def add_to_vis(self, pcd, color):
        # 设置颜色 只能是0 1 如[1,0,0]代表红色
        pcd.paint_uniform_color(color)
        # 添加点云
        self.vis.add_geometry(pcd)
        if self.view_point_json_file is not None:
            self.ctr.convert_from_pinhole_camera_parameters(self.parameters)

    def screen_shot(self, result_dir):
        self.vis.capture_screen_image(result_dir, do_render=True)

    def show_and_save(self, result_dir):
        if self.if_show:
            self.vis.run()
            if self.if_save:
                self.screen_shot(result_dir)
            self.vis.destroy_window()
        else:
            if self.if_save:
                self.screen_shot(result_dir)

    def clear(self):
        self.vis.clear_geometries()


if __name__ == '__main__':
    visualizer = PcdsVisualizer(view_point_json_file=None)

    # rootdir1 = './Dataset/demo_data/original/'
    # cat1 = os.listdir(rootdir1)
    # cat1 = cat1[0:]
    # color1 = cycle([[0, 0, 1], [0, 1, 0]])
    #
    # for cat, c in zip(cat1, color1):
    #     filename = os.path.join(rootdir1, cat)
    #     pcd = visualizer.read_bin_pc_fps_4(filename)  # 读取数据点
    #     print('pcd found:', filename)
    #     visualizer.add_to_vis(pcd, c)

    rootdir2 = './Dataset/demo_data/interpolated/'
    cat2 = os.listdir(rootdir2)
    cat2 = cat2[0:]
    color2 = cycle([[1, 1, 0], [1, 0, 1], [0, 1, 1], [1, 0, 0]])
    for cat, c in zip(cat2, color2):
        filename = os.path.join(rootdir2, cat)
        pcd = visualizer.read_bin_pc_fps_4(filename)  # 读取数据点
        print('pcd found:', filename)
        visualizer.add_to_vis(pcd, c)
