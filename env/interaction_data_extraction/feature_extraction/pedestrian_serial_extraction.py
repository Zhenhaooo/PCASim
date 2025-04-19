from utils.config import MapPath
from multiprocessing import cpu_count, Pool
import math
import os
import sys
import json
import random
import time
import pandas as pd
import argparse
import numpy as np
from shapely.geometry import LineString
from shapely.strtree import STRtree
import ast
import lanelet2
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
from NGSIM_env import utils
from NGSIM_env.road.road import Road, RoadNetwork
from NGSIM_env.road.lane import LineType, StraightLane, PolyLane, PolyLaneFixedWidth

project_dir = os.sep.join(os.path.abspath(__file__).split(os.sep)[:-3])
sys.path.append(project_dir)


def normalize_angles(angles):
    return (angles + np.pi) % (2 * np.pi) - np.pi


def angle_difference(angle1, angle2):
    return ((angle2 - angle1 + np.pi) % (2 * np.pi)) - np.pi


def calc_cumsum_angle_change(trajs):
    trajs = np.array(trajs)
    headings = trajs[:, 3]
    headings = normalize_angles(headings)

    # 计算方向角差值并规范化
    delta_thetas = angle_difference(headings[:-1], headings[1:])
    delta_thetas = normalize_angles(delta_thetas)

    # 累计方向角变化并规范化
    cumulative_angles = np.cumsum(delta_thetas)
    cumulative_angles = normalize_angles(cumulative_angles)

    return cumulative_angles


def search_lc_data(road, lanes_id, trajs):
    ego_trajs = np.array(trajs['ego']['trajectory'])
    ego_trajs = process_raw_trajectory(ego_trajs)
    current_lane_index = None
    last_lane_index = None
    lane_change_index = None
    lane_change_timestamps = None
    lc = False
    last_lane_front_v = last_lane_rear_v = None
    current_lane_front_v = current_lane_rear_v = None
    towards = ''
    start_lane_change_index = 0
    end_lane_change_index = ego_trajs.shape[0]
    for i in range(ego_trajs.shape[0]):
        ego_pos = ego_trajs[i]
        position = ego_pos[:2]
        heading = ego_pos[3]
        lane_index = road.network.get_closest_lane_index(position, heading)
        current_lane_index = f'{lane_index[0]}_{lane_index[1]}'
        if current_lane_index not in list(lanes_id.keys()):
            continue

        if last_lane_index is not None and last_lane_index != current_lane_index:
            if last_lane_index in list(lanes_id.keys()) and current_lane_index in lanes_id[last_lane_index]:
                lc = True
                lane_change_index = i
                lane_change_timestamps = ego_trajs[i][4]
                bv_trajs_set = []
                bv_ids_set = []
                for key, value in trajs.items():
                    if key == "ego":
                        continue
                    bv_tj = value['trajectory'][i]
                    if np.sum(np.array(bv_tj)[:-1]) == 0:
                        continue
                    bv_ids_set.append(key)  # 添加车辆ID
                    bv_trajs_set.append(bv_tj)  # 添加轨迹

                if len(bv_ids_set) == 0:
                    continue
                last_lane_index = tuple(last_lane_index.split('_')) + (0,)
                current_lane_index = tuple(
                    current_lane_index.split('_')) + (0,)
                last_lane = road.network.get_lane(last_lane_index)
                current_lane = road.network.get_lane(current_lane_index)
                last_lane_front_v, last_lane_rear_v, _, _ = neighbour_vehicles(road,
                                                                               np.array(trajs['ego']['trajectory'])[
                                                                                   i],
                                                                               bv_trajs=bv_trajs_set,
                                                                               bv_ids=bv_ids_set,
                                                                               lane_index=last_lane_index)
                current_lane_front_v, current_lane_rear_v, _, _ = neighbour_vehicles(road,
                                                                                     np.array(
                                                                                         trajs['ego']['trajectory'])[
                                                                                         i],
                                                                                     bv_trajs=bv_trajs_set,
                                                                                     bv_ids=bv_ids_set)

                for j in range(0, i)[::-1]:

                    pos = ego_trajs[j][:2]
                    lon, lat = last_lane.local_coordinates(pos)

                    if abs(lat) <= 0.3 or lon < 0:
                        start_lane_change_index = j
                        break

                for j in range(i, ego_trajs.shape[0]):
                    pos = ego_trajs[j][:2]
                    lon, lat = current_lane.local_coordinates(pos)
                    if abs(lat) <= 0.3 or lon > current_lane.length:
                        end_lane_change_index = j
                        break
                # 通过起止点的时间那一帧来找到起点和终点的连线
                start_point = np.array(ego_trajs)[start_lane_change_index, :2]
                end_point = np.array(ego_trajs)[end_lane_change_index, :2]

                direction_vector = end_point - start_point
                direction_heading = math.atan2(direction_vector[1], direction_vector[0])

                # 计算当前车道的航向角单位向量
                start_lane_lon, _ = current_lane.local_coordinates(start_point)
                start_lane_heading = current_lane.heading_at(start_lane_lon)
                diff_heading = direction_heading - start_lane_heading
                diff_heading = (diff_heading + np.pi) % (2 * np.pi) - np.pi

                if diff_heading < 0:
                    towards = 'left'
                else:
                    towards = 'right'

                break

        last_lane_index = current_lane_index

    return {'lane_change': lc,
            'towards': towards,
            'last_lane_front_v': last_lane_front_v,
            'last_lane_rear_v': last_lane_rear_v,
            'current_lane_front_v': current_lane_front_v,
            'current_lane_rear_v': current_lane_rear_v,
            'lane_change_index': lane_change_index,
            'start_lane_change_index': start_lane_change_index,
            'end_lane_change_index': end_lane_change_index,
            'lane_change_timestamps': lane_change_timestamps}


def check_direction(traj, low_threshold=60, up_threshold=120):
    traj = np.array(traj)
    masked = np.where(traj.sum(axis=1) == 0.0, 0, 1)
    traj = traj[masked == 1]
    is_go_straight = False
    is_turn_left = False
    is_turn_right = False
    is_turn_round = False
    cumulative_angles = calc_cumsum_angle_change(traj)

    # 设置容差
    tolerance = 0.1  # 弧度
    # 检测掉头事件
    u_turns = np.abs(cumulative_angles) >= (np.pi - tolerance)
    u_turn_indices = np.where(u_turns)[0] + 1  # +1调整索引

    # 设置阈值范围 (60° 到 120°) 转弯角度
    low_threshold = np.radians(low_threshold)  # 60度
    up_threshold = np.radians(up_threshold)  # 120度

    # 判断是否发生左转或右转
    left_turns = (cumulative_angles >= low_threshold) & (
            cumulative_angles <= up_threshold)
    right_turns = (cumulative_angles <= -
    low_threshold) & (cumulative_angles >= -up_threshold)

    if u_turn_indices.shape[0] > 0:
        is_turn_round = True
        turn_round_end_index = int(u_turn_indices[0])
    elif np.sum(left_turns) > 0:
        is_turn_left = True
    elif np.sum(right_turns) > 0:
        is_turn_right = True
    else:
        is_go_straight = True

    return {
        'go_straight': is_go_straight,
        'turn_left': is_turn_left,
        'turn_right': is_turn_right,
        'turn_round': is_turn_round
    }


def process_raw_trajectory(trajectory):
    """
    处理原始轨迹，将坐标、速度进行转换。
    :param trajectory: 原始轨迹数据
    :return: 转换后的轨迹数据
    """
    trajectory = np.array(trajectory).copy()
    shape = trajectory.shape
    if len(trajectory.shape) == 1:
        trajectory = trajectory.reshape(1, -1)
    x, y = trajectory[:, 0].copy(), trajectory[:, 1].copy()
    trajectory[:, 0] = y
    trajectory[:, 1] = x
    headings = trajectory[:, 3].copy()
    headings = np.pi / 2 - headings
    headings = (headings + np.pi) % (2 * np.pi) - np.pi
    trajectory[:, 3] = headings
    return trajectory.reshape(shape)


def get_intersection_point(traj_a, traj_b):
    # 构建线段
    lines_a = [LineString([traj_a[i], traj_a[i + 1]])
               for i in range(len(traj_a) - 1)]
    lines_b = [LineString([traj_b[i], traj_b[i + 1]])
               for i in range(len(traj_b) - 1)]

    # 使用STRtree构建空间索引
    tree = STRtree(lines_a)

    # 找到所有交点
    intersections = []
    index = []
    for idx_b, line_b in enumerate(lines_b):
        possible_matches = tree.query(line_b)
        for line_a in possible_matches:
            if lines_a[line_a].intersects(line_b):
                intersection = lines_a[line_a].intersection(line_b)
                if intersection.is_empty:
                    continue
                # intersection可能是点、线段或多点，这里只考虑点的情况
                if 'Point' == intersection.geom_type:
                    intersections.append((intersection.x, intersection.y))
                    index.append((int(line_a), idx_b))

    return intersections, index


def neighbour_vehicles(road, ego_traj, bv_trajs, bv_ids, lane_index=None, lon_distance_limit=15.0,
                       lat_offfset_limit=2.0, lon_offset_limit=5.0):
    ego_traj = process_raw_trajectory(ego_traj)
    ego_speed = ego_traj[2]
    ego_heading = ego_traj[3]
    ego_lane_index = road.network.get_closest_ground_lane_index(
        ego_traj[:2], ego_heading)
    if lane_index is None:
        lane_index = ego_lane_index

    if lane_index != ego_lane_index:
        lon_offset_limit = 0.0
        lat_offfset_limit = 5.0

    lane = road.network.get_lane(lane_index)
    s_front = s_rear = None
    v_front = v_rear = None

    min_ttc = float('inf')
    ttc = float('inf')
    s, lat = lane.local_coordinates(ego_traj[:2])
    for i, bv_traj in enumerate(bv_trajs):
        bv_traj = process_raw_trajectory(bv_traj)
        bv_heading = bv_traj[3]
        bv_lane_index = road.network.get_closest_ground_lane_index(
            bv_traj[:2], bv_heading)
        if lane_index == bv_lane_index or lane_index[1] == bv_lane_index[0] or lane_index[0] == \
                bv_lane_index[1]:
            s_v, lat_v = lane.local_coordinates(bv_traj[:2])
            if abs(lat_v - lat) > lat_offfset_limit or abs(s_v - s) > lon_distance_limit:
                continue

            if s < (s_v - lon_offset_limit) and (s_front is None or s_v <= s_front):
                s_front = s_v
                v_front = bv_ids[i]
                bv_speed = bv_traj[2]
                ttc = abs(s - s_v) / (ego_speed - bv_speed +
                                      1e-5) if bv_speed < ego_speed else float('inf')
                min_ttc = min(min_ttc, ttc)
                if ttc <= 0 or min_ttc <= 0:
                    continue

            if s_v < (s - lon_offset_limit) and (s_rear is None or s_v > s_rear):
                s_rear = s_v
                v_rear = bv_ids[i]

    return v_front, v_rear, min_ttc, ttc


def str_2_tuple(s):
    return ast.literal_eval(s)


def create_road(map_file):
    projector = lanelet2.projection.UtmProjector(lanelet2.io.Origin(0.0, 0.0))
    laneletmap = lanelet2.io.load(map_file, projector)
    roads_dict, graph, laneletmap, indegree, outdegree = utils.load_lanelet_map(
        laneletmap)
    net = RoadNetwork()
    none = LineType.NONE
    lanes_id = {}
    for i, (k, road) in enumerate(roads_dict.items()):
        center = road['center']
        left = road['left']
        right = road['right']
        start_point = (int(road['center'][0][0]), int(road['center'][0][1]))
        end_point = (int(road['center'][-1][0]), int(road['center'][-1][1]))
        net.add_lane(f"{start_point}", f"{end_point}", PolyLane(
            center, left, right, line_types=(none, none)))
        lane_id = f"{start_point}_{end_point}"
        if lanes_id.get(lane_id) is None:
            lanes_id[lane_id] = set()

    # 加载人行道
    pedestrian_marking_id = 0
    for k, v in laneletmap.items():
        ls = v['type']
        if ls["type"] == "pedestrian_marking":
            pedestrian_marking_id += 1
            ls_points = v['points']
            net.add_lane(f"P_{pedestrian_marking_id}_start", f"P_{pedestrian_marking_id}_end",
                         PolyLaneFixedWidth(ls_points, line_types=(none, none), width=5))

    lanes_id_key = list(lanes_id.keys())

    for i, k1 in enumerate(lanes_id_key):

        lane_1_start = str_2_tuple(k1.split('_')[0])
        lane_1_end = str_2_tuple(k1.split('_')[1])
        for j, k2 in enumerate(lanes_id_key):
            if i == j:
                continue
            lane_2_start = str_2_tuple(k2.split('_')[0])
            lane_2_end = str_2_tuple(k2.split('_')[1])

            start_dist = np.linalg.norm(
                np.array(lane_1_start) - np.array(lane_2_start))
            end_dist = np.linalg.norm(
                np.array(lane_1_end) - np.array(lane_2_end))

            if 0 < start_dist < 5 and 0 < end_dist < 5:
                lanes_id[k1].add(k2)
                lanes_id[k2].add(k1)

    for k, v in lanes_id.copy().items():
        if len(v) == 0:
            lanes_id.pop(k)

    return Road(network=net, lanelet=laneletmap), lanes_id


def feature_extract(road, lands_id, trajs: dict, file_name):
    ego_key = trajs['ego']['agent_type']
    if ego_key == 'pedestrian':
        trajectory = feature_pedestrian_extract(road, trajs, file_name)
    elif ego_key == 'bicycle':
        trajectory = feature_bicycle_extract(road, lands_id, trajs, file_name)
    return trajectory


def euclidean_distance(x1, y1, x2, y2):
    return np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)


def find_closest_sidewalk(ego_pos_x, ego_pos_y, sidewalk_list, threshold):
    min_distance = float('inf')  # 初始化最小距离为无穷大
    closest_sl_in = None  # 用于存储距离最近的点
    ped_way = None  # 用于存储最近的人行道
    for sl in sidewalk_list:
        for sl_in in sl:
            x1, y1 = sl_in
            dist = euclidean_distance(ego_pos_x, ego_pos_y, x1, y1)  # 计算欧几里得距离
            if dist < min_distance:  # 更新最小距离
                min_distance = dist
                closest_sl_in = sl_in
                ped_way = sl  # 更新最近的点
    if closest_sl_in is not None and min_distance < threshold:  # 如果找到了最近的点并且距离小于threshold
        return closest_sl_in, ped_way
    return None, None


def check_sidewalk(sidewalk_list, ego_trajs):
    ego_traj = process_raw_trajectory(ego_trajs)
    f_crossing = False
    f_u_turn = False
    flag_find_first = False
    threshold = 5.0

    # 遍历找到横穿
    for i, traj in enumerate(ego_traj):
        ego_pos_x, ego_pos_y = traj[:2]
        if not flag_find_first:
            entry_point, ped_way = find_closest_sidewalk(ego_pos_x, ego_pos_y, sidewalk_list, threshold)
            if entry_point is not None:
                flag_find_first = True
        if flag_find_first:
            x2, y2 = ped_way[1] if entry_point == ped_way[0] else ped_way[0]
            if euclidean_distance(ego_pos_x, ego_pos_y, x2, y2) < threshold:
                f_crossing = True
                break

    # 遍历找到掉头场景
    u_traj = np.array(ego_trajs)
    masked = np.where(u_traj.sum(axis=1) == 0.0, 0, 1)
    u_traj = u_traj[masked == 1]
    u_entry_pos_x, u_entry_pos_y = u_traj[0][:2]
    u_out_pos_x, u_out_pos_y = u_traj[-1][:2]
    cumulative_angles = calc_cumsum_angle_change(u_traj)
    # 设置容差
    tolerance = 0.1  # 弧度
    # 检测掉头事件
    u_turns = np.abs(cumulative_angles) >= (np.pi - tolerance)
    u_turn_indices = np.where(u_turns)[0] + 1  # +1调整索引
    if u_turn_indices.shape[0] > 0 and euclidean_distance(u_entry_pos_x, u_entry_pos_y,u_out_pos_x, u_out_pos_y) > 10:
        for u_turn_moment_index in u_turn_indices:
            if u_turn_moment_index > len(u_traj):
                continue
            u_turn_pos_x, u_turn_pos_y = u_traj[u_turn_moment_index][:2]
            if euclidean_distance(u_turn_pos_x, u_entry_pos_x, u_turn_pos_y, u_entry_pos_y) < 2:
                f_u_turn = True

    return f_crossing, f_u_turn


def check_pass_sidewalk(ego_trajectory):
    ego_t = process_raw_trajectory(ego_trajectory)
    for i, traj in enumerate(ego_t):
        ego_heading = traj[3]
        moment_trajectory_lane_index = road.network.get_closest_lane_index(traj[:2], ego_heading)
        for element in moment_trajectory_lane_index:
            if isinstance(element, str) and element.startswith('P'):
                return True  # 当人算
    return False  # 当车算


def find_ped_stay(ego_trajectory):
    ego_all_timestamps = ego_trajectory[:, -1]
    stay_threshold = 0.003

    # 找驻留特征
    for i, t in enumerate(ego_all_timestamps):
        if i + 14 >= len(ego_all_timestamps):  # 如果剩余的帧数不足 15 帧，跳出
            break

        flag = True
        # 获取第i帧的位置
        initial_pos = ego_trajectory[i][:2]  # 获取该帧的x, y位置
        for j in range(i + 1, i + 15):  # 从 i 到 i+15 进行比较
            per_ped_pos = ego_trajectory[j][:2]  # 获取每一帧的x, y位置
            # 计算当前帧与前一帧的欧几里得距离
            dist = euclidean_distance(initial_pos[0], initial_pos[1], per_ped_pos[0], per_ped_pos[1])
            if dist > stay_threshold:  # 如果某一帧的距离大于0.1m
                flag = False
                break
            initial_pos = per_ped_pos  # 更新初始位置，继续检查下一帧

        if flag:  # 如果15帧内都没有超过0.1m的移动距离，则认为是驻留
            return True
    return False


def find_start_moment(road, intersection_index, v_locus):
    v_heading = v_locus[intersection_index, 3]
    entry_index = road.network.get_closest_ground_lane_index(
        v_locus[intersection_index, :2], v_heading)
    start_entry_index = 0
    for i in range(intersection_index)[::-1]:
        per_entry_index = road.network.get_closest_ground_lane_index(
            v_locus[i, :2], v_locus[i, 3])
        if per_entry_index[0] != entry_index[0] or per_entry_index[1] != entry_index[1]:
            start_entry_index = i
            break
    return start_entry_index


def find_moment_pet(ego_locus, bv_locus, ego_speed, bv_speed):
    d_ego_locus = np.cumsum(np.linalg.norm(
        ego_locus[1:] - ego_locus[:-1], axis=1))[::-1]
    d_bv_locus = np.cumsum(np.linalg.norm(
        bv_locus[1:] - bv_locus[:-1], axis=1))[::-1]
    pet_list = abs(abs(d_ego_locus / (ego_speed + 1e-8)) -
                   abs((d_bv_locus / (bv_speed + 1e-8))))
    pet_list = pet_list[pet_list <= 50]

    return pet_list


def find_appear_sudden(t_dict, moment, x1, x2, y1, y2):
    # 计算平均坐标
    avg_point_x = round((x1 + x2) / 2, 4)
    avg_point_y = round((y1 + y2) / 2, 4)

    # 遍历字典中的每个背景车辆
    for i, (k, v) in enumerate(t_dict.copy().items()):
        if k != 'ego':
            bv_trajs = np.array(v['trajectory'])
            for traj in bv_trajs:
                x, y, speed, heading, timestamps = traj
                if timestamps == moment:
                    distance = euclidean_distance(x, y, avg_point_x, avg_point_y)
                    if distance < 0.5:
                        return True
    return False


def feature_pedestrian_extract(road, trajs: dict, f_name):
    ego_trajs = trajs['ego']['trajectory']
    ego_trajs = np.array(ego_trajs)
    scenario_info = {}
    scenario_info['risk_metrics'] = {}
    scenario_info['risk_metrics']['PET'] = None
    scenario_info['risk_metrics']['PET_list'] = []
    scenario_info['intersection_info'] = {}
    scenario_info['travel_characteristics'] = {}
    scenario_info['travel_characteristics']['crossing'] = False
    scenario_info['travel_characteristics']['U_turn'] = False
    scenario_info['travel_characteristics']['stay'] = False
    scenario_info['travel_characteristics']['appear_sudden'] = False
    # 收集人行横道数据，注意这是个转换之后的坐标
    sidewalk_list = road.network.get_pedestrian_index()
    # 穿过人行横道 or 行人掉头
    flag_crossing, flag_U_turn = check_sidewalk(sidewalk_list, ego_trajs)
    # 找相关的交互车
    pet_list = []
    min_pet = float('inf')
    for i, (k, v) in enumerate(trajs.copy().items()):

        if k != 'ego':
            background_vehicle = k
            bv_trajs = v['trajectory']
            bv_trajs = np.array(bv_trajs)
            masked = np.where(bv_trajs.sum(axis=1) == 0.0, 0, 1)
            bv_trajs = bv_trajs[masked == 1]

            # intersections：轨迹的所有交点位置; index_cp：交点在轨迹中对应的线段索引
            intersections, index_cp = get_intersection_point(
                ego_trajs[:, :2].tolist(), bv_trajs[:, :2].tolist())

            if len(intersections) == 1:
                # 找驻留
                flag_ped_stay = find_ped_stay(ego_trajs)
                scenario_info['travel_characteristics']['stay'] = True if flag_ped_stay else False
                # 记录pet_list
                ego_pet_start_time = find_start_moment(road, index_cp[0][0], ego_trajs)
                ego_pet_end_time = max(index_cp[0][0], index_cp[0][1])
                bv_pet_end_time = min(index_cp[0][0], index_cp[0][1])
                bv_pet_start_time = bv_pet_end_time - ego_pet_end_time + ego_pet_start_time
                if bv_pet_start_time < 0:
                    continue
                av_heading = ego_trajs[index_cp[0][0]][-2]
                bv_heading = bv_trajs[index_cp[0][1]][-2]
                bv_start_idx = np.where(masked == 1)[0]
                av_index = index_cp[0][0]
                bv_index = index_cp[0][1] + int(bv_start_idx[0])
                # 找后车
                later_vehicle = background_vehicle
                later_vehicle_trajectory = bv_trajs
                later_index = bv_index - int(bv_start_idx[0])
                if bv_index < av_index:
                    later_vehicle = 'ego'
                    later_vehicle_trajectory = ego_trajs
                    later_index = av_index
                intersection_time = later_vehicle_trajectory[later_index][4]
                trajs_mutual_moment = int((intersection_time - later_vehicle_trajectory[:1, -1]) / 100)
                if 0 > trajs_mutual_moment or trajs_mutual_moment >= len(bv_trajs):
                    continue
                ego_intersection_pos_x, ego_intersection_pos_y = ego_trajs[trajs_mutual_moment][:2]
                bv_intersection_pos_x, bv_intersection_pos_y = bv_trajs[trajs_mutual_moment][:2]
                # 找“鬼探头”
                flag_appear_sudden = find_appear_sudden(trajs, intersection_time,
                                                        ego_intersection_pos_x,
                                                        ego_intersection_pos_y,
                                                        bv_intersection_pos_x,
                                                        bv_intersection_pos_y)
                v_intersection_distance = math.sqrt((ego_intersection_pos_x - bv_intersection_pos_x) ** 2 + (
                        ego_intersection_pos_y - bv_intersection_pos_y) ** 2)
                later_v_speed = 1.1
                if later_vehicle != 'ego':
                    later_v_speed = later_vehicle_trajectory[max(trajs_mutual_moment - 2, 0)][2]
                # 后到达的那个与当时刻的前面那个距离必须在15m以内
                if v_intersection_distance < 15.0 and later_v_speed > 1.0:
                    pet_list = find_moment_pet(ego_trajs[ego_pet_start_time:ego_pet_end_time, :2],
                                               bv_trajs[bv_pet_start_time:bv_pet_end_time, :2],
                                               ego_trajs[ego_pet_start_time:ego_pet_end_time - 1, 2],
                                               bv_trajs[bv_pet_start_time:bv_pet_end_time - 1, 2])
                    min_pet = np.min(pet_list) if pet_list.size > 0 else None
                    ego_all_timestamps = np.array(trajs['ego']['trajectory'])[:, -1]
                    later_v_time = int(
                        (intersection_time - ego_all_timestamps[1]) / 100 + 1)
                    for i, t in enumerate(ego_all_timestamps):
                        if t == intersection_time:
                            surr_trajs_set = []
                            surr_ids_set = []
                            for key, value in trajs.items():
                                if key == later_vehicle:
                                    continue
                                try:
                                    surr_tj = value['trajectory'][i]
                                except IndexError:
                                    continue
                                if np.sum(np.array(surr_tj)[:-1]) == 0:
                                    continue
                                surr_ids_set.append(key)
                                surr_trajs_set.append(surr_tj)
                            if len(surr_ids_set) == 0:
                                continue
                    if later_v_time < len(later_vehicle_trajectory):
                        current_front_v, _, _, _ = neighbour_vehicles(road, later_vehicle_trajectory[later_v_time],
                                                                      surr_trajs_set, surr_ids_set)
                    else:
                        current_front_v = None
                    if (current_front_v is None or current_front_v == background_vehicle) and (
                            flag_crossing or flag_crossing or flag_appear_sudden):
                        intersection_info = {
                            'intersection': intersections,
                            'index_cp': index_cp,
                            'av_heading': av_heading,
                            'bv_heading': bv_heading,
                            'bv_direction_info': check_direction(bv_trajs),
                            'PET': min_pet,
                        }
                        scenario_info['travel_characteristics'][
                            'crossing'] = True if flag_crossing else False
                        scenario_info['travel_characteristics'][
                            'U_turn'] = True if flag_U_turn else False
                        if flag_U_turn:
                            print(f"pedestrian u_turn path {f_name}")
                        scenario_info['travel_characteristics'][
                            'appear_sudden'] = True if flag_appear_sudden else False
                        scenario_info['intersection_info'][background_vehicle] = intersection_info
                        if isinstance(pet_list, np.ndarray) and min_pet is not None:
                            scenario_info['risk_metrics']['PET_list'] = pet_list.tolist()
                        elif isinstance(pet_list, list) and min_pet is not None:
                            scenario_info['risk_metrics']['PET_list'] = pet_list

    scenario_info['risk_metrics']['PET'] = min_pet if min_pet != float(
        'inf') else None
    trajs['ego']['scenario_info'] = scenario_info
    return trajs


def feature_bicycle_extract(road, lands_id, trajs: dict, f_name):
    ego_trajs = trajs['ego']['trajectory']
    ego_trajs = np.array(ego_trajs)
    flag_pass_pedestrian = check_pass_sidewalk(ego_trajs)
    # 当成人
    if flag_pass_pedestrian:
        scenario_info = {}
        scenario_info['risk_metrics'] = {}
        scenario_info['risk_metrics']['PET'] = None
        scenario_info['risk_metrics']['PET_list'] = []
        scenario_info['intersection_info'] = {}
        scenario_info['travel_characteristics'] = {}
        scenario_info['travel_characteristics']['crossing'] = False
        scenario_info['travel_characteristics']['U_turn'] = False
        scenario_info['travel_characteristics']['stay'] = False
        scenario_info['travel_characteristics']['appear_sudden'] = False
        # 收集人行横道数据，注意这是个转换之后的坐标
        sidewalk_list = road.network.get_pedestrian_index()
        # 穿过人行横道 or 掉头
        flag_crossing, flag_U_turn = check_sidewalk(sidewalk_list, ego_trajs)
        # 找相关的交互车
        pet_list = []
        min_pet = float('inf')
        for i, (k, v) in enumerate(trajs.copy().items()):

            if k != 'ego':
                background_vehicle = k
                bv_trajs = v['trajectory']
                bv_trajs = np.array(bv_trajs)
                masked = np.where(bv_trajs.sum(axis=1) == 0.0, 0, 1)
                bv_trajs = bv_trajs[masked == 1]

                # intersections：轨迹的所有交点位置; index_cp：交点在轨迹中对应的线段索引
                intersections, index_cp = get_intersection_point(
                    ego_trajs[:, :2].tolist(), bv_trajs[:, :2].tolist())

                if len(intersections) == 1:
                    # 找驻留
                    flag_ped_stay = find_ped_stay(ego_trajs)
                    scenario_info['travel_characteristics']['stay'] = True if flag_ped_stay else False
                    # 记录pet_list
                    ego_pet_start_time = find_start_moment(road, index_cp[0][0], ego_trajs)
                    ego_pet_end_time = max(index_cp[0][0], index_cp[0][1])
                    bv_pet_end_time = min(index_cp[0][0], index_cp[0][1])
                    bv_pet_start_time = bv_pet_end_time - ego_pet_end_time + ego_pet_start_time
                    if bv_pet_start_time < 0:
                        continue
                    av_heading = ego_trajs[index_cp[0][0]][-2]
                    bv_heading = bv_trajs[index_cp[0][1]][-2]
                    bv_start_idx = np.where(masked == 1)[0]
                    av_index = index_cp[0][0]
                    bv_index = index_cp[0][1] + int(bv_start_idx[0])
                    # 找后车
                    later_vehicle = background_vehicle
                    later_vehicle_trajectory = bv_trajs
                    later_index = bv_index - int(bv_start_idx[0])
                    if bv_index < av_index:
                        later_vehicle = 'ego'
                        later_vehicle_trajectory = ego_trajs
                        later_index = av_index
                    intersection_time = later_vehicle_trajectory[later_index][4]
                    trajs_mutual_moment = int((intersection_time - later_vehicle_trajectory[:1, -1]) / 100)
                    if 0 > trajs_mutual_moment or trajs_mutual_moment >= len(bv_trajs):
                        continue
                    ego_intersection_pos_x, ego_intersection_pos_y = ego_trajs[trajs_mutual_moment][:2]
                    bv_intersection_pos_x, bv_intersection_pos_y = bv_trajs[trajs_mutual_moment][:2]
                    # 找“鬼探头”
                    flag_appear_sudden = find_appear_sudden(trajs, intersection_time,
                                                            ego_intersection_pos_x,
                                                            ego_intersection_pos_y,
                                                            bv_intersection_pos_x,
                                                            bv_intersection_pos_y)
                    v_intersection_distance = math.sqrt((ego_intersection_pos_x - bv_intersection_pos_x) ** 2 + (
                            ego_intersection_pos_y - bv_intersection_pos_y) ** 2)
                    # 后到达的那个与当时刻的前面那个距离必须在15m以内
                    if v_intersection_distance < 15.0:
                        pet_list = find_moment_pet(ego_trajs[ego_pet_start_time:ego_pet_end_time, :2],
                                                   bv_trajs[bv_pet_start_time:bv_pet_end_time, :2],
                                                   ego_trajs[ego_pet_start_time:ego_pet_end_time - 1, 2],
                                                   bv_trajs[bv_pet_start_time:bv_pet_end_time - 1, 2])
                        min_pet = np.min(pet_list) if pet_list.size > 0 else None
                        ego_all_timestamps = np.array(trajs['ego']['trajectory'])[:, -1]
                        later_v_time = int(
                            (intersection_time - ego_all_timestamps[1]) / 100 + 1)
                        for i, t in enumerate(ego_all_timestamps):
                            if t == intersection_time:
                                surr_trajs_set = []
                                surr_ids_set = []
                                for key, value in trajs.items():
                                    if key == later_vehicle:
                                        continue
                                    try:
                                        surr_tj = value['trajectory'][i]
                                    except IndexError:
                                        continue
                                    if np.sum(np.array(surr_tj)[:-1]) == 0:
                                        continue
                                    surr_ids_set.append(key)
                                    surr_trajs_set.append(surr_tj)
                                if len(surr_ids_set) == 0:
                                    continue
                        if later_v_time < len(later_vehicle_trajectory):
                            current_front_v, _, _, _ = neighbour_vehicles(road, later_vehicle_trajectory[later_v_time],
                                                                          surr_trajs_set, surr_ids_set)
                        else:
                            current_front_v = None
                        if (current_front_v is None or current_front_v == background_vehicle) and (
                                flag_crossing or flag_crossing or flag_appear_sudden):
                            intersection_info = {
                                'intersection': intersections,
                                'index_cp': index_cp,
                                'av_heading': av_heading,
                                'bv_heading': bv_heading,
                                'bv_direction_info': check_direction(bv_trajs),
                                'PET': min_pet,
                            }
                            scenario_info['travel_characteristics'][
                                'crossing'] = True if flag_crossing else False
                            scenario_info['travel_characteristics'][
                                'U_turn'] = True if flag_U_turn else False
                            if flag_U_turn:
                                print(f"bicycle u_turn path {f_name}")
                            scenario_info['travel_characteristics'][
                                'appear_sudden'] = True if flag_appear_sudden else False
                            scenario_info['intersection_info'][background_vehicle] = intersection_info
                            if isinstance(pet_list, np.ndarray) and min_pet is not None:
                                scenario_info['risk_metrics']['PET_list'] = pet_list.tolist(
                                )
                            elif isinstance(pet_list, list) and min_pet is not None:
                                scenario_info['risk_metrics']['PET_list'] = pet_list
        scenario_info['risk_metrics']['PET'] = min_pet if min_pet != float(
            'inf') else None
    # 当成车
    elif not flag_pass_pedestrian:
        ego_start_pos_x,ego_start_pos_y = ego_trajs[0][:2]
        ego_end_pos_x,ego_end_pos_y = ego_trajs[-1][:2]
        ego_speed_list = ego_trajs[:, 2]
        avg_speed = sum(np.abs(ego_speed_list)) / ego_speed_list.size
        # 作为一个车起点和终点还小于15m基本可以断定为边界外
        if euclidean_distance(ego_start_pos_x,ego_start_pos_y,ego_end_pos_x,ego_end_pos_y) < 20:
            return
        print(f"ke neng zhen de you zai xian nei de path {f_name}")
        if avg_speed > 7:
            trajs['ego']['agent_type'] = "motorcycle"
        scenario_info = {}
        scenario_info['intersection_info'] = {}
        scenario_info['risk_metrics'] = {}
        scenario_info['risk_metrics']['PET'] = None
        scenario_info['risk_metrics']['TTC'] = None
        scenario_info['travel_characteristics'] = {}
        scenario_info['travel_characteristics']['go_straight'] = False
        scenario_info['travel_characteristics']['turn_left'] = False
        scenario_info['travel_characteristics']['turn_right'] = False
        scenario_info['travel_characteristics']['turn_round'] = False
        scenario_info['travel_characteristics']['brake'] = False
        scenario_info['travel_characteristics']['following'] = False
        scenario_info['travel_characteristics']['lane_change'] = False
        scenario_info['following_info'] = []
        scenario_info['brake_info'] = []
        scenario_info['risk_metrics']['TTC_list'] = []
        scenario_info['risk_metrics']['PET_list'] = []
        # 判断变道
        scenario_info['laneChanging_info'] = search_lc_data(road, lands_id, trajs)
        # 判断跟车和制动
        # 定义阈值
        min_speed = 2.0  # 速度
        SPEED_DIFFERENCE_THRESHOLD = 1.5  # 速度差阈值
        BRAKING_ACCELERATION_THRESHOLD = -0.5  # 制动加速度阈值（负值）
        timestamps_ego_start = ego_trajs[0][-1]
        ttc_list = []
        # 初始化指针
        follow_start_index = None  # 跟车开始的时间步
        follow_end_index = None  # 跟车结束的时间步
        ego_all_timestamps = np.array(trajs['ego']['trajectory'])[:, -1]
        last_front_v = None
        follow = False
        follow_flag_first = False
        fall_count = 0
        min_ttc = float('inf')
        for i, timestamp in enumerate(ego_all_timestamps):
            bv_trajs_set = []
            bv_ids_set = []
            for key, value in trajs.items():
                if key == "ego":
                    continue
                bv_tj = value['trajectory'][i]
                if np.sum(np.array(bv_tj)[:-1]) == 0:
                    continue
                bv_ids_set.append(key)  # 添加车辆ID
                bv_trajs_set.append(bv_tj)  # 添加轨迹
            if len(bv_ids_set) == 0:
                continue
            current_front_v, _, ttc, per_ttc = neighbour_vehicles(
                road, ego_trajs[i], bv_trajs=bv_trajs_set, bv_ids=bv_ids_set)
            if per_ttc == float('inf') or per_ttc == 0.0:
                pass
            else:
                ttc_list.append(per_ttc)
            if current_front_v is not None:
                timestamps_ego = timestamp
                # 获取该时刻主车的位姿信息
                ego_traj = trajs['ego']['trajectory'][i]
                speed_ego = ego_traj[2]
                speed_bv = trajs[current_front_v]['trajectory'][i][2]
                speed_diff = speed_ego - speed_bv
                # front_vehicle_changed = False
                # 保证主车和背景车起码一辆车是运动的+判断速度差是否在阈值内
                if abs(speed_diff) < SPEED_DIFFERENCE_THRESHOLD and (speed_ego > min_speed or speed_bv > min_speed):
                    # 判断是否是第一次跟车
                    follow = True
                    if not follow_flag_first:
                        follow_start_index = timestamps_ego  # 记录跟车开始的时间步
                        follow_flag_first = True  # 标记为第一次跟车
                    follow_end_index = timestamps_ego
                min_ttc = min(min_ttc, ttc) if ttc < 15 else min_ttc
            if current_front_v is None and follow:
                fall_count += 1
            # 记录前车的问题
            front_vehicle_changed = True if last_front_v != current_front_v and current_front_v is not None and last_front_v is not None else False
            # 判断跟车结束的条件
            if follow and ((timestamps_ego - follow_end_index) > 1000 or fall_count > 5 or front_vehicle_changed):
                # 如果之前有跟车行为并且此时不再符合跟车条件，记录结束时间
                start = int((follow_start_index - timestamps_ego_start) / 100)
                end = int((follow_end_index - timestamps_ego_start) / 100)
                if end - start >= 5:
                    bv_direction_info = check_direction(
                        trajs[last_front_v]['trajectory'])
                    ego_cumulative_angles = calc_cumsum_angle_change(
                        ego_trajs[start:end])[-1]
                    bv_cumulative_angles = calc_cumsum_angle_change(
                        trajs[last_front_v]['trajectory'][start:end])[-1]
                    scenario_info['following_info'].append({
                        'following_vehicle_id': last_front_v,
                        'following_start': follow_start_index,
                        'following_end': follow_end_index,
                        'following_duration': follow_end_index - follow_start_index,
                        'following_ego_cumulative_angles': ego_cumulative_angles,
                        'following_bv_cumulative_angles': bv_cumulative_angles,
                        'following_front_vehicle': bv_direction_info,
                    })
                follow = False  # 重置跟车标志
                follow_flag_first = False  # 重置第一次跟车标志
                follow_start_index = None
                follow_end_index = None
                fall_count = 0

            if current_front_v is not None:
                last_front_v = current_front_v

        # 制动判断
        if len(scenario_info['following_info']) > 0:
            # 如果检测到跟车行为，进一步判断制动行为
            for fm in scenario_info['following_info']:
                speeds_ego = ego_trajs[:, 2]
                accelerations_ego = np.diff(speeds_ego) / 0.1  # 时间步长为0.1s
                # 处理制动行为
                start = int((fm['following_start'] - timestamps_ego_start) / 100)
                end = int((fm['following_end'] - timestamps_ego_start) / 100)
                masked = np.where(accelerations_ego <
                                  BRAKING_ACCELERATION_THRESHOLD, True, False)
                if np.sum(masked[start:end]) > 0:
                    # 找到第一个True的索引
                    brake_ego_start = np.argmax(
                        masked[start:end]) * 100 + fm['following_start']
                    # 找到最后一个True的索引
                    last_true_idx = len(masked[start:end]) - \
                                    np.argmax(np.flip(masked[start:end]))
                    brake_ego_end = last_true_idx * 100 + fm['following_start']
                    brake_vehicle_id = fm['following_vehicle_id']
                    bv_direction_info = check_direction(
                        trajs[brake_vehicle_id]['trajectory'])
                    b_start = np.argmax(masked[start:end])
                    b_end = last_true_idx
                    if b_end - b_start >= 5:
                        ego_cumulative_angles = calc_cumsum_angle_change(
                            ego_trajs[b_start:b_end])[-1]
                        bv_cumulative_angles = \
                            calc_cumsum_angle_change(trajs[last_front_v]['trajectory'][b_start:b_end])[
                                -1]
                        scenario_info['brake_info'].append({
                            'brake_vehicle_id': brake_vehicle_id,
                            'brake_start': brake_ego_start,
                            'brake_end': brake_ego_end,
                            'brake_duration': brake_ego_end - brake_ego_start,
                            'brake_ego_cumulative_angles': ego_cumulative_angles,
                            'brake_bv_cumulative_angles': bv_cumulative_angles,
                            'brake_front_vehicle': bv_direction_info,
                        })
        # 路口交互判断
        pet_list = []
        min_pet = float('inf')
        for i, (k, v) in enumerate(trajs.copy().items()):

            if k != 'ego':
                background_vehicle = k
                bv_trajs = v['trajectory']
                bv_trajs = np.array(bv_trajs)
                masked = np.where(bv_trajs.sum(axis=1) == 0.0, 0, 1)
                bv_trajs = bv_trajs[masked == 1]

                if bv_trajs.shape[0] < 50:
                    continue

                # intersections：轨迹的所有交点位置; index_cp：交点在轨迹中对应的线段索引
                intersections, index_cp = get_intersection_point(
                    ego_trajs[:, :2].tolist(), bv_trajs[:, :2].tolist())

                if len(intersections) == 1:
                    curve_distance, _ = fastdtw(
                        ego_trajs[:10, :2], bv_trajs[:10, :2], dist=euclidean)
                    if curve_distance <= 250:
                        continue
                    ego_pet_start_time = find_start_moment(
                        road, index_cp[0][0], ego_trajs)
                    ego_pet_end_time = max(index_cp[0][0], index_cp[0][1])
                    bv_pet_end_time = min(index_cp[0][0], index_cp[0][1])
                    bv_pet_start_time = bv_pet_end_time - ego_pet_end_time + ego_pet_start_time
                    if bv_pet_start_time < 0:
                        continue
                    av_heading = ego_trajs[index_cp[0][0]][-2]
                    bv_heading = bv_trajs[index_cp[0][1]][-2]
                    bv_start_idx = np.where(masked == 1)[0]
                    av_index = index_cp[0][0]
                    bv_index = index_cp[0][1] + int(bv_start_idx[0])
                    pet = abs(av_index - bv_index) * 0.1
                    if pet < min_pet:
                        min_pet = pet
                    # 找后车
                    later_vehicle = background_vehicle
                    later_vehicle_trajectory = bv_trajs
                    later_index = bv_index - int(bv_start_idx[0])
                    if bv_index < av_index:
                        later_vehicle = 'ego'
                        later_vehicle_trajectory = ego_trajs
                        later_index = av_index
                    intersection_time = later_vehicle_trajectory[later_index][4]
                    trajs_mutual_moment = int((intersection_time - later_vehicle_trajectory[:1, -1]) / 100)
                    if 0 > trajs_mutual_moment or trajs_mutual_moment >= len(bv_trajs):
                        continue
                    ego_intersection_pos_x, ego_intersection_pos_y = ego_trajs[trajs_mutual_moment][:2]
                    bv_intersection_pos_x, bv_intersection_pos_y = bv_trajs[trajs_mutual_moment][:2]
                    v_intersection_distance = math.sqrt((ego_intersection_pos_x - bv_intersection_pos_x) ** 2 + (
                            ego_intersection_pos_y - bv_intersection_pos_y) ** 2)
                    if later_vehicle_trajectory[max(trajs_mutual_moment - 2, 0)][2] > 2.5 and v_intersection_distance < 25.0:
                        pet_list = find_moment_pet(ego_trajs[ego_pet_start_time:ego_pet_end_time, :2],
                                                   bv_trajs[bv_pet_start_time:bv_pet_end_time, :2],
                                                   ego_trajs[ego_pet_start_time:ego_pet_end_time - 1, 2],
                                                   bv_trajs[bv_pet_start_time:bv_pet_end_time - 1, 2])
                        if ego_pet_end_time - ego_pet_start_time == 1:
                            pet_list = [1.0]
                        ego_all_timestamps = np.array(
                            trajs['ego']['trajectory'])[:, -1]
                        later_v_time = int(
                            (intersection_time - ego_all_timestamps[1]) / 100 + 1)
                        for i, t in enumerate(ego_all_timestamps):
                            if t == intersection_time:
                                surr_trajs_set = []
                                surr_ids_set = []
                                for key, value in trajs.items():
                                    if key == later_vehicle:
                                        continue
                                    try:
                                        surr_tj = value['trajectory'][i]
                                    except IndexError:
                                        continue
                                    if np.sum(np.array(surr_tj)[:-1]) == 0:
                                        continue
                                    surr_ids_set.append(key)
                                    surr_trajs_set.append(surr_tj)
                                if len(surr_ids_set) == 0:
                                    continue
                        if later_v_time < len(later_vehicle_trajectory):
                            current_front_v, _, _, _ = neighbour_vehicles(road, later_vehicle_trajectory[later_v_time],
                                                                          surr_trajs_set, surr_ids_set)
                        else:
                            current_front_v = None
                        if current_front_v is None or current_front_v == background_vehicle:
                            intersection_info = {
                                'intersection': intersections,
                                'index_cp': index_cp,
                                'av_heading': av_heading,
                                'bv_heading': bv_heading,
                                'bv_direction_info': check_direction(bv_trajs),
                                'PET': pet,
                            }
                            scenario_info['intersection_info'][background_vehicle] = intersection_info
                            if isinstance(pet_list, np.ndarray) and min_pet is not None:
                                scenario_info['risk_metrics']['PET_list'] = pet_list.tolist(
                                )
                            elif isinstance(pet_list, list) and min_pet is not None:
                                scenario_info['risk_metrics']['PET_list'] = pet_list

        scenario_info['risk_metrics']['PET'] = min_pet if min_pet != float(
            'inf') else None
        scenario_info['risk_metrics']['TTC'] = min_ttc if min_ttc != float(
            'inf') else None
        scenario_info['risk_metrics']['TTC_list'] = ttc_list if any(
            ttc_list) and min_ttc is not None else None

        if len(scenario_info['brake_info']) > 0:
            scenario_info['travel_characteristics']['brake'] = True
        if len(scenario_info['following_info']) > 0:
            scenario_info['travel_characteristics']['following'] = True
        if scenario_info['laneChanging_info']['lane_change']:
            scenario_info['travel_characteristics']['lane_change'] = True
        direction_info = check_direction(ego_trajs)
        scenario_info['travel_characteristics'].update(direction_info)
    trajs['ego']['scenario_info'] = scenario_info
    return trajs


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset-path', type=str, default="data/pedestrian_data/trajectory_set",
                        help="interaction dataset path")
    parser.add_argument('--save-path', type=str, default="data/pedestrian_data/Processed_trajectory_set")
    parser.add_argument('--map-path', type=str, default=MapPath)
    parser.add_argument('--interaction-data-path', type=str,
                        default="/home/chuan/work/TypicalScenarioExtraction/data/datasets/Interaction"
                                "/recorded_trackfiles/DR_USA_Intersection_MA")
    args = parser.parse_args()

    vehicle_dir = f'{project_dir}/{args.dataset_path}'  # 拼接数据集目录的绝对路径
    vehicle_names = os.listdir(vehicle_dir)[:]  # 列出数据集目录下的所有文件名
    vehicle_names = vehicle_names[:]

    if not os.path.exists(f'{project_dir}/{args.save_path}'):
        os.makedirs(f'{project_dir}/{args.save_path}')

    road, lanes_id = create_road(args.map_path)
    for i, vn in enumerate(vehicle_names[:]):
        print("\r", end="")
        print("progress: {} %. ".format((i + 1) / len(vehicle_names[:]) * 100), end="")
        path = os.path.join(vehicle_dir, vn)  # 拼接此车辆数据文件的绝对路径
        with open(path, "r") as f:
            trajectory_set = json.load(f)
        processed_trajectory_set = feature_extract(road, lanes_id, trajectory_set, vn)
        if processed_trajectory_set is not None:
            with open(f'{project_dir}/{args.save_path}/{vn}', 'w', encoding='utf-8') as f:
                json.dump(processed_trajectory_set, f, ensure_ascii=False)

    print("\n数据处理完成！")
