import json
import os
import itertools
import numpy as np
import argparse
from matplotlib import pyplot as plt

from utils.config import statistics_path, pic_save_path


def draw_mutual_scenario(statistics_path, pic_save_path):
    # 读取新的 statistics.json
    with open(f'{statistics_path}/statistics.json', 'r') as file:
        json_data = json.load(file)

    # 读取旧的 statistics_old.json
    with open(f'{statistics_path}/statistics_old.json', 'r') as file:
        json_data_old = json.load(file)

    # 遍历一级目录
    for first_level_key in json_data['statistics'].keys():
        if first_level_key == 'turn_round':
            continue  # 如果是 'turn_round'，跳过该目录

        # 遍历二级目录
        for second_level_key in json_data['statistics'][first_level_key].keys():
            # 检查两个文件中是否都有该二级目录
            if second_level_key in json_data_old['statistics'].get(first_level_key, {}):
                # 获取新旧数据
                new_data = json_data['statistics'][first_level_key][second_level_key]
                old_data = json_data_old['statistics'][first_level_key][second_level_key]

                # 处理新数据
                if new_data.get('scenario_num', 0) > 0 and new_data.get('PET', []):
                    data_new = [x for x in new_data.get('PET', []) if
                                x is not None and isinstance(x, (int, float)) and x <= 50.0]
                    data_old = [x for x in old_data.get('PET', []) if
                                x is not None and isinstance(x, (int, float)) and x <= 50.0]
                elif new_data.get('scenario_num', 0) > 0 and new_data.get('TTC', []):
                    data_new = [x for x in new_data.get('TTC', []) if
                                x is not None and isinstance(x, (int, float)) and x <= 50.0]
                    data_old = [x for x in old_data.get('TTC', []) if
                                x is not None and isinstance(x, (int, float)) and x <= 50.0]
                else:
                    continue
                # 绘图
                max_value_new = max(data_new) if data_new else 50.0
                max_value_old = max(data_old) if data_old else 50.0
                max_value = int(max(max_value_new, max_value_old))

                bins = np.linspace(0, max_value + 5, max_value + 5)
                fig, axs = plt.subplots(1, 1)

                # 计算新数据的直方图
                hist_new = np.histogram(data_new, bins=bins, density=True)
                widths_new = np.diff(hist_new[1])
                norm_counts_new = hist_new[0] / (widths_new * hist_new[0]).sum()
                norm_counts_new = norm_counts_new / np.sum(norm_counts_new)

                # 计算旧数据的直方图
                hist_old = np.histogram(data_old, bins=bins, density=True)
                widths_old = np.diff(hist_old[1])
                norm_counts_old = hist_old[0] / (widths_old * hist_old[0]).sum()
                norm_counts_old = norm_counts_old / np.sum(norm_counts_old)

                # 绘制新数据的直方图
                axs.bar(hist_new[1][:-1], norm_counts_new, width=widths_new, color='#DC143C', alpha=1.0,
                        label='New Data (Ego Vehicle)')
                # 绘制旧数据的直方图
                axs.bar(hist_old[1][:-1], norm_counts_old, width=widths_old, color='#32CD32', alpha=0.7,
                        label='Old Data (Ego Vehicle)')

                axs.set_ylabel('Density')
                axs.set_xlabel(f"{first_level_key}->{second_level_key}".upper())
                plt.legend()
                plt.tight_layout()
                plt.grid(True, linestyle='--', alpha=0.6)

                # 保存图像
                save_path = f'{pic_save_path}/compare_mutual'
                if not os.path.exists(save_path):
                    os.makedirs(save_path)
                plt.savefig(f'{save_path}/{first_level_key}_{second_level_key}.svg', format='svg')
                plt.close(fig)


def draw_list_scenario(statistics_path, pic_save_path):
    with open(f'{statistics_path}/statistics.json', 'r') as file:
        json_data = json.load(file)

    # 读取旧的 statistics_old.json
    with open(f'{statistics_path}/statistics_old.json', 'r') as file:
        json_data_old = json.load(file)

    # 遍历一级目录
    for first_level_key in json_data['statistics'].keys():
        if first_level_key == 'turn_round':
            continue  # 如果是 'turn_round'，跳过该目录

        # 遍历二级目录
        for second_level_key in json_data['statistics'][first_level_key].keys():
            # 检查两个文件中是否都有该二级目录
            if second_level_key in json_data_old['statistics'].get(first_level_key, {}):
                # 获取新旧数据
                new_data = json_data['statistics'][first_level_key][second_level_key]
                old_data = json_data_old['statistics'][first_level_key][second_level_key]
                # col_name = f"{first_level_key}->{second_level_key}"
            # PET list 加载
            if new_data.get('scenario_num', 0) > 0 and new_data.get('PET_list', []):
                data_new = new_data.get('PET_list', [])
                if data_new is None or not isinstance(data_new, list) or all(not sublist for sublist in data_new):
                    continue
                flattened_data_new = []
                for sublist in data_new:
                    if sublist is not None:
                        flattened_data_new.extend(sublist)
                data_new = flattened_data_new
                
                data_old = old_data.get('PET_list', [])
                if data_old is None or not isinstance(data_old, list) or all(not sublist for sublist in data_old):
                    continue
                flattened_data_old = []
                for sublist in data_old:
                    if sublist is not None:
                        flattened_data_old.extend(sublist)
                data_old = flattened_data_old
            # TTC list 加载
            elif new_data.get('scenario_num', 0) > 0 and new_data.get('TTC_list', []):
                data_new = new_data.get('TTC_list', [])
                if data_new is None or not isinstance(data_new, list) or all(not sublist for sublist in data_new):
                    continue
                flattened_data_new = []
                for sublist in data_new:
                    if sublist is not None:
                        flattened_data_new.extend(sublist)
                data_new = flattened_data_new
                
                data_old = old_data.get('TTC_list', [])
                if data_old is None or not isinstance(data_old, list) or all(not sublist for sublist in data_old):
                    continue
                flattened_data_old = []
                for sublist in data_old:
                    if sublist is not None:
                        flattened_data_old.extend(sublist)
                data_old = flattened_data_old
            else:
                continue
            data_new = [x for x in data_new if x is not None and isinstance(x, (int, float)) and x <= 50.0 and x != 0]
            data_old = [x for x in data_old if x is not None and isinstance(x, (int, float)) and x <= 50.0 and x != 0]
            # 绘图
            max_value_new = max(data_new) if data_new else 50.0
            max_value_old = max(data_old) if data_old else 50.0
            max_value = int(max(max_value_new, max_value_old))

            bins = np.linspace(0, max_value + 5, max_value + 5)
            fig, axs = plt.subplots(1, 1)

            # 计算新数据的直方图
            hist_new = np.histogram(data_new, bins=bins, density=True)
            widths_new = np.diff(hist_new[1])
            norm_counts_new = hist_new[0] / (widths_new * hist_new[0]).sum()
            norm_counts_new = norm_counts_new / np.sum(norm_counts_new)

            # 计算旧数据的直方图
            hist_old = np.histogram(data_old, bins=bins, density=True)
            widths_old = np.diff(hist_old[1])
            norm_counts_old = hist_old[0] / (widths_old * hist_old[0]).sum()
            norm_counts_old = norm_counts_old / np.sum(norm_counts_old)

            # 绘制新数据的直方图
            axs.bar(hist_new[1][:-1], norm_counts_new, width=widths_new, color='#DC143C', alpha=1.0,
                    label='New Data (Ego Vehicle)')
            # 绘制旧数据的直方图
            axs.bar(hist_old[1][:-1], norm_counts_old, width=widths_old, color='#32CD32', alpha=0.7,
                    label='Old Data (Ego Vehicle)')

            axs.set_ylabel('Density')
            axs.set_xlabel(f"{first_level_key}->{second_level_key}".upper())
            plt.legend()
            plt.tight_layout()
            plt.grid(True, linestyle='--', alpha=0.6)

            # 保存图像
            save_path = f'{pic_save_path}/compare_per'
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            plt.savefig(f'{save_path}/{first_level_key}_{second_level_key}.svg', format='svg')
            plt.close(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--draw-min', action='store_true', help="save every min pet or ttc", default=True)
    parser.add_argument('--draw-all', action='store_true', help="save pet or ttc for every scenario", default=True)
    args = parser.parse_args()
    if args.draw_min:
        draw_mutual_scenario(statistics_path, pic_save_path)
    if args.draw_all:
        draw_list_scenario(statistics_path, pic_save_path)
