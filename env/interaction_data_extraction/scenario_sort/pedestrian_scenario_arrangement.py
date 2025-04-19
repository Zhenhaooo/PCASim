import os
import sys
import json
import time
import shutil

project_dir = os.sep.join(os.path.abspath(__file__).split(os.sep)[:-3])
sys.path.append(project_dir)

import random
import pandas as pd
import argparse
import numpy as np
import simulation_environment
import gymnasium as gym
from gymnasium.wrappers import RecordVideo
from utils.config import MapPath
import matplotlib.pyplot as plt
from matplotlib import font_manager
import warnings

# 允许覆盖
warnings.filterwarnings("ignore", category=UserWarning, module='gymnasium')
font_properties = font_manager.FontProperties(family='SimSun', size=15)  # Adjusted size to 10 for better fit
font_properties2 = font_manager.FontProperties(family='Times New Roman', size=15)  # Adjusted size to 10 for better fit

start_time = time.time()


def visualize_trajectory(file_path, video_save_path, video_save_name, bv_id=None):
    env = gym.make(
        'interaction-v0',
        render_mode="rgb_array",
        data_path=file_path,
        osm_path=MapPath,
        bv_ids=bv_id,
    )
    env = RecordVideo(
        env, video_folder=video_save_path, episode_trigger=lambda e: True,
        name_prefix=video_save_name,
    )
    env.unwrapped.set_record_video_wrapper(env)

    obs, info = env.reset()
    done = False
    while not done:
        obs, reward, terminated, truncated, info = env.step(None)
        env.render()
        done = terminated or truncated
    env.close()


def make_scenario_dir(scenarios_level_name: dict):
    for k, v in scenarios_level_name.items():
        for k1, v1 in v.items():
            for k2,v2 in v1.items():
                if not os.path.exists(f'{project_dir}/data/pedestrian_data/scenario_data/{k}/{k1}/{k2}'):
                    os.makedirs(f'{project_dir}/data/pedestrian_data/scenario_data/{k}/{k1}/{k2}')
                    os.makedirs(f'{project_dir}/data/pedestrian_data/scenario_data/videos/{k}/{k1}/{k2}')

    if not os.path.exists(f'{project_dir}/data/pedestrian_data/scenario_data/statistics_data'):
        os.makedirs(f'{project_dir}/data/pedestrian_data/scenario_data/statistics_data')


def save_trajectory(trajectory_data: dict, save_path: str):
    with open(save_path, 'w', encoding='utf-8') as file:
        json.dump(trajectory_data, file, ensure_ascii=False)


def scenario_classification(base_save_path: str, file_name, scenarios_level_name: dict,
                            trajectory_data: dict,
                            intersection_direction: list = ['crossing', 'U_turn', 'stay', 'appear_sudden']):

    agent_type = trajectory_set['ego']['agent_type']
    if agent_type in ['bicycle', 'pedestrian']:
        scenario_info = trajectory_data['ego']['scenario_info'].copy()
        trajectory_data['ego'].pop('scenario_info')
        # intersection
        for k, v in scenario_info['intersection_info'].items():
            pet = v['PET']
            min_pet = float('inf')
            if pet < min_pet:
                min_pet_id = k
            for k1, v1 in v['bv_direction_info'].items():
                pet_list = []
                if v1 and pet < 8:
                    for dir in intersection_direction:
                        if scenario_info['travel_characteristics'][dir] and min_pet_id == k:
                            new_scenario_info = {
                                'travel_characteristics': scenario_info['travel_characteristics'],
                                'intersection': {k: v}
                            }
                            trajectory_data['ego']['scenario_info'] = new_scenario_info
                            save_trajectory(trajectory_data, os.path.join(base_save_path, agent_type, dir, 'bv_' + k1, file_name))
                            trajectory_data['ego'].pop('scenario_info')
                            pet_list.extend(scenario_info['risk_metrics']['PET_list'])
                            scenarios_level_name[agent_type][dir]['bv_' + k1]['total_file_path'].append(file_name)
                            scenarios_level_name[agent_type][dir]['bv_' + k1]['statistics'].append({
                                'PET': scenario_info['risk_metrics']['PET'],
                                'PET_list': pet_list,
                                'bv_id': k,
                            })
    elif agent_type == 'motorcycle':
        if not os.path.exists(f'{project_dir}/data/pedestrian_data/scenario_data/motorcycle'):
            os.makedirs(f'{project_dir}/data/pedestrian_data/scenario_data/motorcycle')
        motor_bath_path = f'{project_dir}/data/pedestrian_data/scenario_data/motorcycle'
        save_trajectory(trajectory_data, os.path.join(motor_bath_path, file_name))


def statistics(scenarios_level_name: dict, statistics_name_list: list = ['TTC', 'PET', 'TTC_list', 'PET_list']):
    statistics_data = {}
    for at, at_v in scenarios_level_name.items():
        statistics_data[at] = {}
        for k, v in at_v.items():
            statistics_data[at][k] = {}
            for k1, v1 in v.items():
                statistics_data[at][k][k1] = {
                    'scenario_num': len(v1['total_file_path']),
                }
                if len(v1['statistics']) == 0:
                    continue

                for statistics_data_name, statistics_data_value in v1['statistics'][0].items():
                    if statistics_data_name not in statistics_name_list:
                        continue

                    statistics_data[at][k][k1][statistics_data_name] = [sts[statistics_data_name] for sts in v1['statistics']]

    return statistics_data


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset-path', type=str, default="data/pedestrian_data/Processed_trajectory_set",
                        help="interaction dataset path")
    parser.add_argument('--map-path', type=str,
                        default="/home/chuan/work/TypicalScenarioExtraction/data/datasets/Interaction/maps/DR_USA_Intersection_MA.osm")
    parser.add_argument('--save-video', action='store_true', help="save video or not", default=True)
    parser.add_argument('--video-num', type=int, default=10, help="number of video to save")
    args = parser.parse_args()

    vehicle_dir = f'{project_dir}/{args.dataset_path}'  # 拼接数据集目录的绝对路径
    vehicle_names = os.listdir(vehicle_dir)[:]  # 列出数据集目录下的所有文件名
    scenarios_level_name = {
        'pedestrian': {
            'crossing': {
                'bv_go_straight': {
                    'total_file_path': [],
                    'statistics': [],
                },
                'bv_turn_left': {
                    'total_file_path': [],
                    'statistics': [],
                },
                'bv_turn_right': {
                    'total_file_path': [],
                    'statistics': [],
                },
                'bv_turn_around': {
                    'total_file_path': [],
                    'statistics': [],
                },
            },
            'U_turn': {
                'bv_go_straight': {
                    'total_file_path': [],
                    'statistics': [],
                },
                'bv_turn_left': {
                    'total_file_path': [],
                    'statistics': [],
                },
                'bv_turn_right': {
                    'total_file_path': [],
                    'statistics': [],
                },
                'bv_turn_around': {
                    'total_file_path': [],
                    'statistics': [],
                },
            },
            'stay': {
                'bv_go_straight': {
                    'total_file_path': [],
                    'statistics': [],
                },
                'bv_turn_left': {
                    'total_file_path': [],
                    'statistics': [],
                },
                'bv_turn_right': {
                    'total_file_path': [],
                    'statistics': [],
                },
                'bv_turn_around': {
                    'total_file_path': [],
                    'statistics': [],
                },
            },
            'appear_sudden': {
                'go_straight': {
                    'total_file_path': [],
                    'statistics': [],
                },
                'turn_left': {
                    'total_file_path': [],
                    'statistics': [],
                },
                'turn_right': {
                    'total_file_path': [],
                    'statistics': [],
                },
                'bv_turn_around': {
                    'total_file_path': [],
                    'statistics': [],
                },
            },
        },
        'bicycle': {
            'crossing': {
                'bv_go_straight': {
                    'total_file_path': [],
                    'statistics': [],
                },
                'bv_turn_left': {
                    'total_file_path': [],
                    'statistics': [],
                },
                'bv_turn_right': {
                    'total_file_path': [],
                    'statistics': [],
                },
                'bv_turn_around': {
                    'total_file_path': [],
                    'statistics': [],
                },
            },
            'U_turn': {
                'bv_go_straight': {
                    'total_file_path': [],
                    'statistics': [],
                },
                'bv_turn_left': {
                    'total_file_path': [],
                    'statistics': [],
                },
                'bv_turn_right': {
                    'total_file_path': [],
                    'statistics': [],
                },
                'bv_turn_around': {
                    'total_file_path': [],
                    'statistics': [],
                },
            },
            'stay': {
                'bv_go_straight': {
                    'total_file_path': [],
                    'statistics': [],
                },
                'bv_turn_left': {
                    'total_file_path': [],
                    'statistics': [],
                },
                'bv_turn_right': {
                    'total_file_path': [],
                    'statistics': [],
                },
                'bv_turn_around': {
                    'total_file_path': [],
                    'statistics': [],
                },
            },
            'appear_sudden': {
                'go_straight': {
                    'total_file_path': [],
                    'statistics': [],
                },
                'turn_left': {
                    'total_file_path': [],
                    'statistics': [],
                },
                'turn_right': {
                    'total_file_path': [],
                    'statistics': [],
                },
                'bv_turn_around': {
                    'total_file_path': [],
                    'statistics': [],
                },
            },
        },
    }
    make_scenario_dir(scenarios_level_name)
    # 创建保存路径（如果不存在）
    base_save_path = f'{project_dir}/data/pedestrian_data/scenario_data'

    for i, vn in enumerate(vehicle_names[:]):
        print("\r", end="")
        print("progress: {} %. ".format((i + 1) / len(vehicle_names[:]) * 100), end="")
        path = os.path.join(vehicle_dir, vn)  # 拼接此车辆数据文件的绝对路径
        with open(path, "r") as f:
            trajectory_set = json.load(f)
        scenario_classification(base_save_path, vn, scenarios_level_name, trajectory_set)

    print("\n数据处理完成！")

    statistics_data = statistics(scenarios_level_name)
    statistics_data_save = {}
    statistics_data_save['statistics'] = statistics_data
    statistics_data_save.update(scenarios_level_name)
    # 使用 exist_ok=True 简化目录创建
    statistics_data_dir = f'{project_dir}/data/pedestrian_data/scenario_data/statistics_data'
    os.makedirs(statistics_data_dir, exist_ok=True)

    statistics_data_path = os.path.join(statistics_data_dir, 'statistics.json')

    # 检查是否是目录
    if os.path.isdir(statistics_data_path):
        print(f"Error: {statistics_data_path} is a directory. Removing it.")
        shutil.rmtree(statistics_data_path)

    try:
        with open(statistics_data_path, 'w', encoding='utf-8') as file:
            json.dump(statistics_data_save, file, ensure_ascii=False)
        print("完成了数据的整体采集")
    except Exception as e:
        print(f"Error writing statistics.json: {e}")

    if args.save_video:
        for at, at_v in scenarios_level_name.items():
            for k, v in at_v.items():
                vehicle_list = []
                for k1, v1 in v.items():
                    if len(v1['total_file_path']) > 0:
                        sample_index = list(range(len(v1['total_file_path'])))
                        sample_index = np.random.choice(sample_index, min(args.video_num, len(v1['total_file_path'])), replace=False)
                        for s_idx in sample_index.tolist():
                            vd = v1['total_file_path'][s_idx]
                            st_data = v1['statistics'][s_idx]
                            path = os.path.join(vehicle_dir, vd)
                            video_save_path = f'{project_dir}/data/pedestrian_data/scenario_data/videos/{at}/{k}/{k1}'
                            video_save_name = vd.split('.')[0]
                            vehicle_id = video_save_name.split('_')[-1]
                            try:
                                visualize_trajectory(path, video_save_path, video_save_name, bv_id=st_data['bv_id'])
                            except:
                                pass
    end_time = time.time()
    total_time = end_time - start_time
    print(f"整个脚本的执行时间: {total_time:.2f}秒")
