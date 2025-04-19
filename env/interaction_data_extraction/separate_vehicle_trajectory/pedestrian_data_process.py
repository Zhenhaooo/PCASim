import math
import os
import sys
import json
import time
from utils.math import line_segments_intersect
from interaction_data_extraction.separate_vehicle_trajectory.abstract import DataProcess
from road_network_generation import InterAction
import pandas as pd
import numpy as np
import argparse

project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
sys.path.append(project_dir)

start_time = time.time()


class InterActionDP(DataProcess):
    def __init__(self, config: dict):

        super(InterActionDP, self).__init__(config)

    def read_data(self):
        # 读取csv文件数据
        data = pd.read_csv(self.config["dataset_path"])
        out_df = None
        # 根据配置文件中的代理类型分别处理
        for i, name in enumerate(self.config['agent_type']):
            # 如果是第一个代理类型，则直接赋值给out_df
            if i == 0:
                out_df = self.separate_data_by_key_value(data, self.config['agent_type_name'], name)
            else:
                out_df = pd.concat([out_df,
                                    self.separate_data_by_key_value(data, self.config['agent_type_name'], name)])

        # 将数据按照帧ID排序并重新索引
        self.pd_data = out_df.copy().sort_values(by=self.config['frame_id_name']).reset_index(drop=True)
        # 如果需要处理行人数据
        if self.config['process_pedestrian']:
            # 读取行人数据
            ped_data = pd.read_csv(self.config["dataset_path"].replace('vehicle', 'pedestrian'))
            # 将行人数据按照帧ID排序并重新索引
            self.ped_data = ped_data.copy().sort_values(by=self.config['frame_id_name']).reset_index(drop=True)
        # 删除临时变量out_df
        del out_df


def convert_keys(obj):
    if isinstance(obj, dict):
        new_dict = {}
        for k, v in obj.items():
            # 处理键
            if isinstance(k, np.integer):
                k = int(k)
            elif isinstance(k, np.floating):
                k = float(k)
            elif isinstance(k, np.bool_):
                k = bool(k)
            else:
                k = str(k)
            # 递归处理值
            new_dict[k] = convert_keys(v)
        return new_dict
    elif isinstance(obj, list):
        return [convert_keys(item) for item in obj]
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.integer, np.int64, np.int32, np.int16, np.uint8)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32, np.float16)):
        return float(obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    else:
        return obj


# 通过速度来区别行人和自行车
def apart_person_bicycle(ego_speed_list):
    avg_speed = sum(np.abs(ego_speed_list)) / ego_speed_list.size if ego_speed_list.size > 0 else 0.0
    if avg_speed <= 1.3:
        return 'pedestrian'
    else:
        return 'bicycle'


if __name__ == '__main__':
    print(project_dir)
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset-path', type=str, default="/home/chuan/work/TypicalScenarioExtraction/data/datasets",
                        help="ngsim dataset path")
    parser.add_argument('--save-path', type=str, default="data/ngsim")
    parser.add_argument('--ped_traffic_flow', action='store_true', help="generate pedestrian traffic flow")
    args = parser.parse_args()
    # 设置配置文件
    cf = {
        "dataset_path": f"{args.dataset_path}/Interaction",
        "output_path": f"{project_dir}/data/interaction_data",
        "save_road_kwargs": {
        }
    }
    # 实例化InterAction对象
    ia = InterAction(cf)
    ia.parse_dataset_path()  # 解析数据集路径 ia.classify_path()
    ia.classify_path()  # 对数据集路径进行分类
    if not os.path.exists(f"{project_dir}/data/pedestrian_data/trajectory_set"):
        os.makedirs(f"{project_dir}/data/pedestrian_data/trajectory_set")
    # count = 0
    for i, p in enumerate(ia.class_path.get(1)):
        cf = {
            "dataset_path": p,
            "coordinate_x_name": "x",
            "coordinate_y_name": "y",
            "speed_x_name": "vx",
            "speed_y_name": "vy",
            "time_name": "timestamp_ms",
            "frame_id_name": "frame_id",
            "vehicle_id_name": "track_id",
            "agent_type_name": "agent_type",
            'length_name': 'length',
            'width_name': 'width',
            'heading_name': 'psi_rad',
            "agent_type": ["car", "pedestrian/bicycle"],
            # "agent_type": ["car"],
            'process_pedestrian': True,  # True则需要处理行人数据
        }
        dataset = InterActionDP(cf)
        dataset()  # 调用__call__方法,执行数据处理
        print("process data in {}.".format(p))
        # 行人交通流
        for ped_id in dataset.get_ped_set(dataset.ped_data):
            # print(vh_id)
            trajectory_set = dataset.build_trajecotry(0, ped_id)
            ego_trajectory = np.array(trajectory_set['ego']['trajectory'])
            trajectory_set['ego']['agent_type'] = apart_person_bicycle(ego_trajectory[:, 2])
            # if trajectory_set['ego']['agent_type'] == 'pedestrian':
            #     count += 1
            trajectory_set_converted = convert_keys(trajectory_set)
            # 将轨迹集合保存为JSON文件
            with open(
                    f'{project_dir}/data/pedestrian_data/trajectory_set/{p.split("/")[-1].split(".")[0]}_trajectory_set_{ped_id}.json',
                    'w', encoding='utf-8') as f:
                json.dump(trajectory_set_converted, f, ensure_ascii=False)
    # print(count)
    end_time = time.time()
    total_time = end_time - start_time
    print(f"整个脚本的执行时间: {total_time:.2f}秒")
