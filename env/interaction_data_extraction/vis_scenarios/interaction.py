import os
import sys
current_dir = sys.path[0].replace("\\", "/")
project_dir = os.sep.join(current_dir.split('/')[:-2]).replace("\\", "/")
sys.path.append(project_dir)
from NGSIM_env.envs.interaction_env import InterActionEnv
import os
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset-path', type=str, default="data/interaction_data/trajectory_set", help="ngsim dataset path")
    parser.add_argument('--save-path', type=str, default="data/interaction")
    args = parser.parse_args()

    vehicle_dir = f'{project_dir}/{args.dataset_path}'  # 拼接数据集目录的绝对路径
    vehicle_names = os.listdir(vehicle_dir)[:]  # 列出数据集目录下的所有文件名
    for i, vn in enumerate(vehicle_names[:]):
        print("\r", end="")
        print("progress: {} %. ".format((i + 1) / len(vehicle_names[:]) * 100), end="")
        vn = "vehicle_tracks_001_trajectory_set_P2.json"
        path = os.path.join(vehicle_dir, vn)  # 拼接此车辆数据文件的绝对路径
        vehicle_id = vn.split('.')[0].split('_')[-1]  # 解析文件名中的车辆ID
        video_save_name = vn.split('.')[0]  # 解析文件名中的车辆ID
        period = 0
        env = InterActionEnv(path=path, vehicle_id=vehicle_id, IDM=False, render=True, save_video=True)  # 创建交互环境实例
        length = len(env.vehicle.ngsim_traj)  # 获取该车辆的轨迹长度
        env.reset()  # 重置环境
        expert_traj = []  # 存储该车辆的轨迹信息
        env.reset(reset_time=1, video_save_name=video_save_name)  # 重置环境，并指定重置时间
        obs, reword, terminated, info = env.step(action=None)  # 执行一步交互，并获取结果信息
        break


