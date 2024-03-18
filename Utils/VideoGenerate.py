import os
import cv2
from tqdm import tqdm     # python 进度条库

image_folder_dir = "../Demos/demo_0519/pngs/"
fps = 4     # fps: frame per seconde 每秒帧数，数值可根据需要进行调整
size = (1920,1080)     # (width, height) 数值可根据需要进行调整
fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')     # 编码为 mp4v 格式，注意此处字母为小写，大写会报错
video = cv2.VideoWriter("../demo2.mp4", fourcc, fps, size, isColor=True)

image_list = [name for name in os.listdir(image_folder_dir) if name.endswith('.png')]     # 获取文件夹下所有格式为 jpg 图像的图像名，并按时间戳进行排序
for i in tqdm(range(len(image_list))):     # 遍历 image_list 中所有图像并添加进度条

		image_full_path = os.path.join(image_folder_dir, image_list[i])     # 获取图像的全路经
		image = cv2.imread(image_full_path)     # 读取图像
		video.write(image)     # 将图像写入视频

video.release()
cv2.destroyAllWindows()
