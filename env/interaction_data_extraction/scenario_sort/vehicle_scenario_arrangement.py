import os
import sys
import json
import time

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
        if k not in single_scenario_names:
            for k1, v1 in v.items():
                if not os.path.exists(f'{project_dir}/data/interaction_data/scenario_data/{k}/{k1}'):
                    os.makedirs(f'{project_dir}/data/interaction_data/scenario_data/{k}/{k1}')
                    os.makedirs(f'{project_dir}/data/interaction_data/scenario_data/videos/{k}/{k1}')
        else:
            if not os.path.exists(f'{project_dir}/data/interaction_data/scenario_data/{k}'):
                os.makedirs(f'{project_dir}/data/interaction_data/scenario_data/{k}')
                os.makedirs(f'{project_dir}/data/interaction_data/scenario_data/videos/{k}')

    if not os.path.exists(f'{project_dir}/data/interaction_data/scenario_data/statistics_data'):
        os.makedirs(f'{project_dir}/data/interaction_data/scenario_data/statistics_data')


def save_trajectory(trajectory_data: dict, save_path: str):
    with open(save_path, 'w', encoding='utf-8') as file:
        json.dump(trajectory_data, file, ensure_ascii=False)


def scenario_classification(base_save_path: str, file_name, scenarios_level_name: dict,
                            trajectory_data: dict,
                            intersection_direction: list = ['go_straight', 'turn_left', 'turn_right']):
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
            if v1 and 'turn_round' not in k1 and pet < 8:
                for dir in intersection_direction:
                    if scenario_info['travel_characteristics'][dir] and min_pet_id == k:
                        new_scenario_info = {
                            'travel_characteristics': scenario_info['travel_characteristics'],
                            'intersection': {k: v}
                        }
                        trajectory_data['ego']['scenario_info'] = new_scenario_info
                        save_trajectory(trajectory_data, os.path.join(base_save_path, dir, 'bv_' + k1, file_name))
                        trajectory_data['ego'].pop('scenario_info')
                        pet_list.extend(scenario_info['risk_metrics']['PET_list'])
                        scenarios_level_name[dir]['bv_' + k1]['total_file_path'].append(file_name)
                        scenarios_level_name[dir]['bv_' + k1]['statistics'].append({
                            'PET': scenario_info['risk_metrics']['PET'],
                            'PET_list': pet_list,
                            'bv_id': k,
                        })

    # lane_change
    if scenario_info['travel_characteristics']['lane_change']:
        lane_change_info = scenario_info['laneChanging_info']
        new_scenario_info = {
            'travel_characteristics': scenario_info['travel_characteristics'],
            'laneChanging_info': lane_change_info
        }
        trajectory_data['ego']['scenario_info'] = new_scenario_info
        save_trajectory(trajectory_data,
                        os.path.join(base_save_path, 'lane_change', f'towards_the_{lane_change_info["towards"]}',
                                     file_name))
        trajectory_data['ego'].pop('scenario_info')
        scenarios_level_name['lane_change'][f'towards_the_{lane_change_info["towards"]}']['total_file_path'].append(
            file_name)
        scenarios_level_name['lane_change'][f'towards_the_{lane_change_info["towards"]}']['statistics'].append({
            'TTC': scenario_info['risk_metrics']['TTC'],
            'TTC_list': scenario_info['risk_metrics']['TTC_list'],
            'bv_id': None,
        })

    # following brake
    for sc in ['following', 'brake']:
        ttc_list = []
        if scenario_info['travel_characteristics'][sc]:
            info = scenario_info[f'{sc}_info']
            scenario_list = []
            for item in info:
                angle_diff = abs(item[f'{sc}_ego_cumulative_angles'] - item[f'{sc}_bv_cumulative_angles'])
                if angle_diff > np.radians(5):
                    continue
                # 设置容差
                tolerance = 0.1  # 弧度
                # 检测掉头事件
                u_turns = np.abs(item[f'{sc}_ego_cumulative_angles']) >= (np.pi * 2 / 3 - tolerance)
                # 设置阈值范围 (60° 到 120°) 转弯角度
                low_threshold = np.radians(20)
                up_threshold = np.radians(120)
                # 判断是否发生左转或右转
                left_turns = (item[f'{sc}_ego_cumulative_angles'] >= low_threshold) & (
                        item[f'{sc}_ego_cumulative_angles'] <= up_threshold)
                right_turns = (item[f'{sc}_ego_cumulative_angles'] <= -low_threshold) & (
                        item[f'{sc}_ego_cumulative_angles'] >= -up_threshold)
                if u_turns:
                    front_vehicle = 'turn_round'
                elif np.sum(left_turns) > 0:
                    front_vehicle = 'turn_left'
                elif np.sum(right_turns) > 0:
                    front_vehicle = 'turn_right'
                else:
                    front_vehicle = 'go_straight'
                corrected_value = front_vehicle
                new_info = {f'{sc}_vehicle_id': item[f'{sc}_vehicle_id'], f'{sc}_start': item[f'{sc}_start'],
                            f'{sc}_end': item[f'{sc}_end'], f'{sc}_duration': item[f'{sc}_duration'],
                            f'{sc}_direction': corrected_value}
                scenario_list.append(new_info)
            new_scenario_info = {
                'travel_characteristics': scenario_info['travel_characteristics'],
                f'{sc}_info': scenario_list
            }
            for dir in ['go_straight', 'turn_left', 'turn_right', 'turn_round']:
                trajectory_data['ego']['scenario_info'] = trajectory_data['ego'].get('scenario_info', {})
                trajectory_data['ego']['scenario_info'].update({"travel_characteristics": new_scenario_info[
                    'travel_characteristics']})
                all_sc_list = []
                if not scenario_info['risk_metrics']['TTC_list']:
                    continue
                else:
                    ttc_list.extend(scenario_info['risk_metrics']['TTC_list'])
                for i, value in enumerate(new_scenario_info[f'{sc}_info']):
                    if value[f'{sc}_direction'] == dir:
                        all_sc_list.append(value)
                        trajectory_data['ego']['scenario_info'][f'{sc}_info'] = all_sc_list
                        save_trajectory(trajectory_data, os.path.join(base_save_path, sc, dir, file_name))
                        trajectory_data['ego']['scenario_info'].pop(f'{sc}_info')
                        scenarios_level_name[sc][dir]['total_file_path'].append(file_name)
                        scenarios_level_name[sc][dir]['statistics'].append({
                            'TTC': scenario_info['risk_metrics']['TTC'],
                            'TTC_list': ttc_list,
                            'bv_id': [inf[f'{sc}_vehicle_id'] for inf in info],
                        })

    # turn round
    if scenario_info['travel_characteristics']['turn_round']:
        new_scenario_info = {
            'travel_characteristics': scenario_info['travel_characteristics']
        }
        trajectory_data['ego']['scenario_info'] = new_scenario_info
        save_trajectory(trajectory_data, os.path.join(base_save_path, 'turn_round', file_name))
        trajectory_data['ego'].pop('scenario_info')
        scenarios_level_name['turn_round']['total_file_path'].append(file_name)
        scenarios_level_name['turn_round']['statistics'].append({
            'TTC': scenario_info['risk_metrics']['TTC'],
            'TTC_list': scenario_info['risk_metrics']['TTC_list'],
            'bv_id': None,
        })


def statistics(scenarios_level_name: dict, statistics_name_list: list = ['TTC', 'PET', 'TTC_list', 'PET_list']):
    statistics_data = {}
    for k, v in scenarios_level_name.items():
        if k not in single_scenario_names:
            statistics_data[k] = {}
            for k1, v1 in v.items():
                statistics_data[k][k1] = {
                    'scenario_num': len(v1['total_file_path']),
                }
                if len(v1['statistics']) == 0:
                    continue

                for statistics_data_name, statistics_data_value in v1['statistics'][0].items():
                    if statistics_data_name not in statistics_name_list:
                        continue
                    statistics_data[k][k1][statistics_data_name] = [sts[statistics_data_name] for sts in v1['statistics']]
        else:
            statistics_data[k] = {
                'scenario_num': len(v['total_file_path']),
            }
            if len(v['statistics']) == 0:
                continue

            for statistics_data_name, statistics_data_value in v['statistics'][0].items():
                if statistics_data_name not in statistics_name_list:
                    continue

                statistics_data[k][statistics_data_name] = [sts[statistics_data_name] for sts in v['statistics']]

    return statistics_data


def plot_scenario_distribution(statistics_data, save_path, save_name):
    bar_width = 0.3
    names = ['', '直行', '左转', '右转', '制动', '跟车', '变道', '掉头']
    fig, ax = plt.subplots(figsize=(4, 10))
    go_straight = statistics_data['go_straight']
    turn_left = statistics_data['turn_left']
    turn_right = statistics_data['turn_right']
    brake = statistics_data['brake']
    following = statistics_data['following']
    lane_change = statistics_data['lane_change']
    turn_round = statistics_data['turn_round']

    density_1 = [go_straight['bv_go_straight']['scenario_num'], turn_left['bv_go_straight']['scenario_num'],
                 turn_right['bv_go_straight']['scenario_num'], brake['go_straight']['scenario_num'],
                 following['go_straight']['scenario_num'], lane_change['towards_the_left']['scenario_num'],
                 turn_round['scenario_num']]
    density_2 = [go_straight['bv_turn_left']['scenario_num'], turn_left['bv_turn_left']['scenario_num'],
                 turn_right['bv_turn_left']['scenario_num'], brake['turn_left']['scenario_num'],
                 following['turn_left']['scenario_num'], lane_change['towards_the_right']['scenario_num']]
    density_3 = [go_straight['bv_turn_right']['scenario_num'], turn_left['bv_turn_right']['scenario_num'],
                 turn_right['bv_turn_right']['scenario_num'], brake['turn_right']['scenario_num'],
                 following['turn_right']['scenario_num']]

    plt.bar(np.arange(len(density_1)) - bar_width, np.array(density_1) + 1, width=bar_width, color='r')
    plt.bar(np.arange(len(density_2)), np.array(density_2) + 1, width=bar_width, color='g')
    plt.bar(np.arange(len(density_3)) + bar_width, np.array(density_3) + 1, width=bar_width, color='b')
    ax.set_xticklabels(names, fontproperties=font_properties)
    ax.set_ylabel('场景数', fontproperties=font_properties)
    plt.yticks(fontproperties=font_properties2)
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, save_name + '.svg'), format='svg')


if __name__ == '__main__':
    print(project_dir)
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset-path', type=str, default="data/interaction_data/Processed_trajectory_set",
                        help="interaction dataset path")
    # parser.add_argument('--save-path', type=str, default="data/interaction_data/videos")
    parser.add_argument('--map-path', type=str,
                        default="/home/chuan/work/TypicalScenarioExtraction/data/datasets/Interaction/maps/DR_USA_Intersection_MA.osm")
    parser.add_argument('--save-video', action='store_true', help="save video or not", default=True)
    parser.add_argument('--video-num', type=int, default=10, help="number of video to save")
    args = parser.parse_args()

    vehicle_dir = f'{project_dir}/{args.dataset_path}'  # 拼接数据集目录的绝对路径
    vehicle_names = os.listdir(vehicle_dir)[:]  # 列出数据集目录下的所有文件名
    single_scenario_names = ['turn_round']
    scenarios_level_name = {
        'go_straight': {
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
        },
        'turn_left': {
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
        },
        'turn_right': {
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
        },
        'turn_round': {
            'total_file_path': [],
            'statistics': [],
        },
        'brake': {
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
        },
        'following': {
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
        },
        'lane_change': {
            'towards_the_left': {
                'total_file_path': [],
                'statistics': [],
            },
            'towards_the_right': {
                'total_file_path': [],
                'statistics': [],
            },
        }
    }

    make_scenario_dir(scenarios_level_name)
    # 创建保存路径（如果不存在）
    base_save_path = f'{project_dir}/data/interaction_data/scenario_data'

    for i, vn in enumerate(vehicle_names[:]):
        print("\r", end="")
        print("progress: {} %. ".format((i + 1) / len(vehicle_names[:]) * 100), end="")
        path = os.path.join(vehicle_dir, vn)  # 拼接此车辆数据文件的绝对路径
        with open(path, "r") as f:
            trajectory_set = json.load(f)
        scenario_classification(base_save_path, vn, scenarios_level_name, trajectory_set)

    print("\n数据处理完成！")

    statistics_data = statistics(scenarios_level_name)
    # plot_scenario_distribution(statistics_data, f'{project_dir}/data/interaction_data/scenario_data/statistics_data',
    #                            'scenario_distribution')
    statistics_data_save = {}
    statistics_data_save['statistics'] = statistics_data
    statistics_data_save.update(scenarios_level_name)
    with open(f'{project_dir}/data/interaction_data/scenario_data/statistics_data/statistics.json', 'w',
              encoding='utf-8') as file:
        json.dump(statistics_data_save, file, ensure_ascii=False)

    print("完成了数据的整体采集")

    if args.save_video:
        for k, v in scenarios_level_name.items():
            vehicle_list = []
            if k not in single_scenario_names:
                for k1, v1 in v.items():
                    if len(v1['total_file_path']) > 0:
                        sample_index = list(range(len(v1['total_file_path'])))
                        sample_index = np.random.choice(sample_index, min(args.video_num, len(v1['total_file_path'])),
                                                        replace=False)
                        for s_idx in sample_index.tolist():
                            vd = v1['total_file_path'][s_idx]
                            st_data = v1['statistics'][s_idx]
                            path = os.path.join(vehicle_dir, vd)
                            video_save_path = f'{project_dir}/data/interaction_data/scenario_data/videos/{k}/{k1}'
                            video_save_name = vd.split('.')[0]
                            vehicle_id = video_save_name.split('_')[-1]
                            try:
                                visualize_trajectory(path, video_save_path, video_save_name, bv_id=st_data['bv_id'])
                            except:
                                pass
            else:
                if len(v) > 0:
                    sample_index = list(range(len(v['total_file_path'])))
                    sample_index = np.random.choice(sample_index, min(args.video_num, len(v['total_file_path'])),
                                                    replace=False)

                    for s_idx in sample_index.tolist():
                        vd = v['total_file_path'][s_idx]
                        st_data = v['statistics'][s_idx]
                        path = os.path.join(vehicle_dir, vd)
                        video_save_path = f'{project_dir}/data/interaction_data/scenario_data/videos/{k}'
                        video_save_name = vd.split('.')[0]
                        vehicle_id = video_save_name.split('_')[-1]
                        try:
                            visualize_trajectory(path, video_save_path, video_save_name, bv_id=st_data['bv_id'])
                        except:
                            pass
    end_time = time.time()
    total_time = end_time - start_time
    print(f"整个脚本的执行时间: {total_time:.2f}秒")
