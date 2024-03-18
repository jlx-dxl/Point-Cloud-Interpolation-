import torch
from torch.utils.data import Dataset, DataLoader
import open3d as o3d
import numpy as np
import os, sys
import copy

sys.path.append("..")
from tqdm import tqdm
from Utils.Visualize import PcdsVisualizer


class NuscenesDataset(Dataset):
    def __init__(self, root, scenes_list, scene_split_lib='./Dataset/scene-split/', field=1,
                 npoints=int(20000), interval=10, if_random=False, random_times = 1):
        # 文件目录相关
        self.root = root  # 放置点云data的目录
        self.scenes = self.read_scene_list(scenes_list)  # 从设定的scenelist中读出scenes的名称
        self.timestamp_list, self.fns_list = self.load_scene(self.scenes,
                                                             scene_split_lib)  # 根据读出的scenes到scene_split_lib中中读取时间戳和帧名

        # 模型参数相关
        self.npoints = npoints  # 读出时把点降采样到相同的个数，默认=20000
        self.interval = interval  # 插帧间隔默认=10
        self.field = field  # 向前后看几帧

        # 读取数据集的方式（遍历/随机选）
        self.if_random = if_random
        self.random_times = random_times

        # 组织数据集（元素为帧的名称）
        self.forward_frame_lists, self.key_frame_lists, self.backward_frame_lists, self.t_list, self.gt_frame_list = self.make_dataset()

    # 把scenelist里的scenes列为python-list
    def read_scene_list(self, scenes_list):
        scenes = []
        with open(scenes_list, 'r') as f:
            for line in f.readlines():
                line = line.strip('\n')
                scenes.append(line)
        return scenes

    def load_scene(self, scenes, scene_split_lib):
        timestamp_list = []
        fns_list = []

        for scene in scenes:
            scene_file = os.path.join(scene_split_lib, scene + '.txt')  # scene_file就是列有该scene下属的所有点云名的文件
            times = []
            fns = []
            with open(scene_file) as f:
                for line in f.readlines():
                    line = line.strip('\n').split(' ')  # 以“ ”为分隔符将每行分为两个元素
                    fn = line[0]  # 第一个元素是frame_name
                    timestamp = float(line[1])  # 第二个元素是timestamp
                    fns.append(fn)  # 将一个scene里的所有framename累积成表
                    times.append(timestamp)  # 将一个scene里的所有timestamp累积成表
            timestamp_list.append(times)  # 将所有scene里的所有timestamp累积成表
            fns_list.append(fns)  # 将所有scene里的所有framename累积成表
        return timestamp_list, fns_list

    def make_dataset(self):
        forward_frame_lists = []  # 用于储存前瞻点，元素：list，长度：self.field
        backward_frame_lists = []  # 用于储存后瞻点，元素：list，长度：self.field
        key_frame_lists = []  # 用于储存关键点（即插帧位置前后两帧），元素：list，长度：2
        t_list = []  # 时间戳，元素：float
        gt_frame_list = []  # 每个时间戳对应一个gt帧，元素：frame_name

        for i in range(len(self.timestamp_list)):  # 遍历所有scene
            times = self.timestamp_list[i]  # 当前scene里的所有timestamp列表
            fns = self.fns_list[i]  # 当前scene里的所有framename列表
            max_ind = len(times)
            key_frame_front_ind = self.field * self.interval
            key_frame_back_ind = self.field * self.interval + self.interval

            while (key_frame_back_ind + self.field * self.interval < max_ind):

                if self.if_random == False:
                    for bias in range(1, self.interval):
                        forw = []
                        backw = []

                        for j in range(1, self.field + 1):
                            forw.append(fns[key_frame_front_ind - self.interval * j])
                            # print("forward_frame_ind:", key_frame_front_ind - self.interval * j)
                        forward_frame_lists.append(forw)

                        keys = [fns[key_frame_front_ind], fns[key_frame_back_ind]]
                        key_frame_lists.append(keys)
                        # print("key_frame_ind:", key_frame_front_ind,key_frame_back_ind)

                        for j in range(1, self.field + 1):
                            backw.append(fns[key_frame_back_ind + self.interval * j])
                            # print("backward_frame_ind:", key_frame_back_ind + self.interval * j)
                        backward_frame_lists.append(backw)

                        t = (times[key_frame_front_ind + bias] - times[key_frame_front_ind]) / (
                                times[key_frame_back_ind] - times[key_frame_front_ind])
                        t_list.append(t)
                        # print("t:", t)

                        gt_frame_list.append(fns[key_frame_front_ind + bias])
                        # print("gt_frame_ind:", key_frame_front_ind + bias)

                    key_frame_front_ind = key_frame_back_ind
                    key_frame_back_ind = key_frame_back_ind + self.interval

                else:
                    forw = []
                    backw = []
                    bias = np.random.randint(1, self.interval, self.random_times)
                    
                    for k in range(self.random_times):
                        for j in range(1, self.field + 1):
                            forw.append(fns[key_frame_front_ind - self.interval * j])
                            # print("forward_frame:", key_frame_front_ind - self.interval * j)
                        forward_frame_lists.append(forw)

                        keys = [fns[key_frame_front_ind], fns[key_frame_back_ind]]
                        key_frame_lists.append(keys)
                        # print("key_frame:", key_frame_front_ind)

                        for j in range(1, self.field + 1):
                            backw.append(fns[key_frame_back_ind + self.interval * j])
                            # print("backward_frame:", key_frame_front_ind + self.interval * j)
                        backward_frame_lists.append(backw)

                        t = (times[key_frame_front_ind + bias[k]] - times[key_frame_front_ind]) / (
                                times[key_frame_back_ind] - times[key_frame_front_ind])
                        t_list.append(t)
                        # print("t:", t)

                        gt_frame_list.append(fns[key_frame_front_ind + bias[k]])
                        # print("gt_frame:", key_frame_front_ind + bias)

                    key_frame_front_ind = key_frame_back_ind
                    key_frame_back_ind = key_frame_back_ind + self.interval

        return forward_frame_lists, key_frame_lists, backward_frame_lists, t_list, gt_frame_list

    def get_lidar(self, fn):
        scan_in = np.fromfile(fn, np.float32).reshape(-1, 5)  # 变成五列
        scan = scan_in[:, :3]
        pcd = o3d.geometry.PointCloud()  # 传入3d点云格式
        pcd.points = o3d.utility.Vector3dVector(scan)  # 转换格式
        pcd_down = pcd.farthest_point_down_sample(self.npoints)  # FPS降采样
        return np.asarray(pcd_down.points)

    def __getitem__(self, index):
        forward_pcds = []
        backward_pcds = []
        key_pcds = []

        for fn in self.forward_frame_lists[index]:
            f = self.get_lidar(os.path.join(self.root, fn))
            forward_pcds.append(torch.from_numpy(f).t().to(torch.float32))

        for fn in self.backward_frame_lists[index]:
            b = self.get_lidar(os.path.join(self.root, fn))
            backward_pcds.append(torch.from_numpy(b).t().to(torch.float32))

        for fn in self.key_frame_lists[index]:
            k = self.get_lidar(os.path.join(self.root, fn))
            key_pcds.append(torch.from_numpy(k).t().to(torch.float32))

        t = self.t_list[index]

        gt = self.get_lidar(os.path.join(self.root, self.gt_frame_list[index]))
        gt = torch.from_numpy(gt).t().to(torch.float32)

        ini_feature = torch.from_numpy(np.zeros([self.npoints, 3])).t().to(torch.float32)

        return forward_pcds, key_pcds, backward_pcds, t, gt, ini_feature

    def __len__(self):
        return len(self.t_list)


if __name__ == '__main__':
    # 数据集测试代码,可以去make_dataset函数中将print index打开，检查读取的index
    dataset = NuscenesDataset(root='../../ISAPCI/Dataset/Subsets/Subset_01/LIDAR_TOP/',
                              scenes_list='../../ISAPCI/Dataset/Subsets/Subset_01/try_list.txt',
                              scene_split_lib='../../ISAPCI/Dataset/scene-split/')
    # print(len(dataset.forward_frame_lists), len(dataset.key_frame_lists), len(dataset.backward_frame_lists),
    #       len(dataset.t_list), len(dataset.gt_frame_list))

    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    pbar = tqdm(enumerate(dataloader))
    print(len(dataloader))
    #

    # 存图集
    visualizer = PcdsVisualizer(if_save=True, if_show=False,
                                view_point_json_file="../Utils/ScreenCamera_2023-05-19-13-09-46.json", point_size=1.0)
    for i, data in pbar:
        forward_pcds, key_pcds, backward_pcds, t, gt, ini_feature = data
        # print(str(i), ":", forward_pcds, key_pcds, backward_pcds, t, gt, ini_feature)

        color1 = [[1, 0, 0], [0.8, 0, 0.2]]
        for pcd, c in zip(forward_pcds, color1):
            pcd_o3d = visualizer.convert_to_o3d_from_tensor(pcd.permute(0,2,1).squeeze(0))
            visualizer.add_to_vis(pcd_o3d, c)

        color2 = [[0.6, 0, 0.4], [0.4, 0, 0.6]]
        for pcd, c in zip(key_pcds, color2):
            pcd_o3d = visualizer.convert_to_o3d_from_tensor(pcd.permute(0,2,1).squeeze(0))
            visualizer.add_to_vis(pcd_o3d, c)

        color3 = [[0.2, 0, 0.8], [0, 0, 1]]
        for pcd, c in zip(backward_pcds, color3):
            pcd_o3d = visualizer.convert_to_o3d_from_tensor(pcd.permute(0,2,1).squeeze(0))
            visualizer.add_to_vis(pcd_o3d, c)

        gt_o3d = visualizer.convert_to_o3d_from_tensor(gt.permute(0,2,1).squeeze(0))
        visualizer.add_to_vis(gt_o3d, [1, 1, 1])
        visualizer.show_and_save("./check/scene-0001-"+str(i)+".png")
        visualizer.clear()


    # # 找+存视角
    # visualizer = PcdsVisualizer(if_save=False, if_show=True,
    #                             view_point_json_file=None, point_size=1.0)
    #
    # for i, data in pbar:
    #     forward_pcds, key_pcds, backward_pcds, t, gt, ini_feature = data
    #
    #     color1 = [[1, 0, 0], [0.8, 0, 0.2]]
    #     for pcd, c in zip(forward_pcds, color1):
    #         pcd_o3d = visualizer.convert_to_o3d_from_tensor(pcd.permute(0,2,1).squeeze(0))
    #         visualizer.add_to_vis(pcd_o3d, c)
    #
    #     color2 = [[0.6, 0, 0.4], [0.4, 0, 0.6]]
    #     for pcd, c in zip(key_pcds, color2):
    #         pcd_o3d = visualizer.convert_to_o3d_from_tensor(pcd.permute(0,2,1).squeeze(0))
    #         visualizer.add_to_vis(pcd_o3d, c)
    #
    #     color3 = [[0.2, 0, 0.8], [0, 0, 1]]
    #     for pcd, c in zip(backward_pcds, color3):
    #         pcd_o3d = visualizer.convert_to_o3d_from_tensor(pcd.permute(0,2,1).squeeze(0))
    #         visualizer.add_to_vis(pcd_o3d, c)
    #
    #     gt_o3d = visualizer.convert_to_o3d_from_tensor(gt.permute(0,2,1).squeeze(0))
    #     visualizer.add_to_vis(gt_o3d, [1, 1, 1])
    #
    #     visualizer.show_and_save(None)

    # 存特定帧的截图
    # forward_pcds, key_pcd, backward_pcds, t, gt, ini_feature, [forward_pcds_for_vis, key_pcd_for_vis,
    #                                                            backward_pcds_for_vis, gt_for_vis] = dataset[10]
    # print([forward_pcds_for_vis, key_pcd_for_vis, backward_pcds_for_vis, gt_for_vis])
    # visualizer.add_to_vis(key_pcd_for_vis, [1, 0, 0])
    # color1 = [[0,1,0],[0.5,0.5,0]]
    # for pcd,c in zip(forward_pcds_for_vis,color1):
    #     visualizer.add_to_vis(pcd,c)
    # color2 = [[0.5,0,0.5],[0,0,0.5]]
    # for pcd,c in zip(backward_pcds_for_vis,color2):
    #     visualizer.add_to_vis(pcd,c)
    # visualizer.add_to_vis(gt_for_vis, [1, 1, 1])
    # visualizer.show_and_save("./demo_data/view_dataloader/Subset_01"+str(10)+"_.png")
