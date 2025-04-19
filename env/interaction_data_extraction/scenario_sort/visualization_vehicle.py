import json
import os
import itertools
import numpy as np
import argparse
from matplotlib import pyplot as plt

from utils.config import statistics_path, pic_save_path


def draw_mutual_scenario(statistics_path, pic_save_path):
    statistics_path_new = f'{statistics_path}/statistics.json'
    with open(statistics_path_new, 'r') as file:
        json_data = json.load(file)

    # 遍历一级目录
    for first_level_key, first_level_value in json_data['statistics'].items():
        if first_level_key == 'turn_round':
            continue  # 如果是 'turn_round'，跳过该目录
        # if first_level_key == 'following':
        #     print()

        # 遍历二级目录
        for second_level_key, second_level_value in first_level_value.items():
            # 记录一级目录 + 二级目录的名字
            col_name = f"{first_level_key}->{second_level_key}"

            # 如果 'scenario_num' 等于 0，跳过该二级目录
            if second_level_value.get('scenario_num') <= 2:
                continue

            for key in second_level_value.keys():
                if key in ['PET', 'pet', 'TTC', 'ttc']:
                    # 获取该 key 对应的所有 value
                    data = second_level_value.get(key, None)
                    data = [x for x in data if x is not None and isinstance(x, (int, float)) and x <= 50.0 and x != 0]
                    max_value = max(data) if data else 50.0
                    max_value = int(max_value)
                    bins = np.linspace(0, max_value, max_value + 5)
                    fig, axs = plt.subplots(1, 1)
                    hist = np.histogram(data, bins=bins, density=True)
                    widths = np.diff(hist[1])
                    norm_counts = hist[0] / (widths * hist[0]).sum()
                    norm_counts = norm_counts / np.sum(norm_counts)
                    axs.bar(hist[1][:-1], norm_counts, width=widths, color='#FF8C00', alpha=1.0, label='ego_vehicle')
                    axs.set_xlim(left=0)
                    axs.set_ylabel('Density')
                    axs.set_xlabel(col_name.upper())
                    plt.legend()
                    plt.tight_layout()
                    plt.grid(True, linestyle='--', alpha=0.6)
                    save_path = f'{pic_save_path}/mutual'
                    if not os.path.exists(save_path):
                        os.makedirs(save_path)
                    plt.savefig(f'{save_path}/{col_name.upper()}.svg', format='svg')


def draw_list_scenario(statistics_path, pic_save_path):
    statistics_path_new = f'{statistics_path}/statistics.json'
    with open(statistics_path_new, 'r') as file:
        json_data = json.load(file)

    # 遍历一级目录
    for first_level_key, first_level_value in json_data['statistics'].items():
        if first_level_key == 'turn_round':
            continue  # 如果是 'turn_round'，跳过该目录

        # 遍历二级目录
        for second_level_key, second_level_value in first_level_value.items():
            # 记录一级目录 + 二级目录的名字
            col_name = f"{first_level_key}->{second_level_key}"
            # 如果 'scenario_num' 等于 0，跳过该二级目录
            if second_level_value.get('scenario_num') <= 2:
                continue

            for key in second_level_value.keys():
                if key in ['PET_list', 'TTC_list']:
                    # 获取该 key 对应的所有 value
                    data = second_level_value.get(key, None)
                    if data is None or not isinstance(data, list) or all(not sublist for sublist in data):
                        print(f"{key} 为空或者其中所有子列表都为空")
                        continue
                    flattened_data = []
                    for sublist in data:
                        if sublist is not None:
                            flattened_data.extend(sublist)
                    # 现在 flattened_data 就是展平后的数据，继续进行后续操作
                    data = flattened_data
                    data = [x for x in data if x is not None and isinstance(x, (int, float)) and x <= 50.0 and x != 0]
                    max_value = max(data) if data else 50.0
                    max_value = int(max_value)
                    bins = np.linspace(0, max_value, max_value + 5)
                    fig, axs = plt.subplots(1, 1)
                    hist = np.histogram(data, bins=bins, density=True)
                    widths = np.diff(hist[1])
                    norm_counts = hist[0] / (widths * hist[0]).sum()
                    norm_counts = norm_counts / np.sum(norm_counts)
                    axs.bar(hist[1][:-1], norm_counts, width=widths, color='#FF8C00', alpha=0.8, label='ego_vehicle')
                    axs.set_xlim(left=0)
                    axs.set_ylabel('Density')
                    axs.set_xlabel(col_name.upper())
                    plt.legend()
                    plt.tight_layout()
                    plt.grid(True, linestyle='--', alpha=0.6)
                    save_path = f'{pic_save_path}/per'
                    if not os.path.exists(save_path):
                        os.makedirs(save_path)
                    plt.savefig(f'{save_path}/{col_name.upper()}.svg', format='svg')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--draw-min', action='store_true', help="save every min pet or ttc", default=True)
    parser.add_argument('--draw-all', action='store_true', help="save pet or ttc for every scenario", default=True)
    args = parser.parse_args()
    if args.draw_min:
        draw_mutual_scenario(statistics_path, pic_save_path)
    if args.draw_all:
        draw_list_scenario(statistics_path, pic_save_path)
