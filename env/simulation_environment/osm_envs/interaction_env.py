import json
import os
from typing import Optional
from simulation_environment.common.action import Action
from simulation_environment import utils
from simulation_environment.osm_envs.osm_env import AbstractEnv
from simulation_environment.road.road import Road, RoadNetwork
from simulation_environment.vehicle.humandriving import HumanLikeVehicle
from simulation_environment.road.lane import LineType, StraightLane, PolyLane, PolyLaneFixedWidth
import pickle
import math
import numpy as np


class InterActionEnv(AbstractEnv):
    """
    一个带有交互数据的十字路口驾驶环境，用于收集gail训练数据。
    """
    def __init__(self, data_path=None, osm_path=None, config: dict = None, render_mode: Optional[str] = None, bv_ids = None):

        self.data_path = data_path
        with open(data_path, "r") as f:
            self.trajectory_set = json.load(f)
        # 获取自车的长度和宽度
        self.ego_length = self.trajectory_set['ego']['length']
        self.ego_width = self.trajectory_set['ego']['width']
        # 获取自车的轨迹数据和持续时间
        self.ego_trajectory = self.process_raw_trajectory(self.trajectory_set['ego']['trajectory'])
        self.duration = len(self.ego_trajectory) - 1
        # 获取周围车辆的ID
        self.surrounding_vehicles = list(self.trajectory_set.keys())
        self.surrounding_vehicles.pop(0)
        self.bv_ids = bv_ids
        super(InterActionEnv, self).__init__(osm_path=osm_path, config=config, render_mode=render_mode)
    @classmethod
    def default_config(self):
        # 设置默认的配置
        config = super().default_config()
        config.update({
            "screen_width": 800,  # [px]
            "screen_height": 600,  # [px]
        })
        return config

    def _reset(self):
        if not self.config['make_road']:
            self._create_road()
            self._create_vehicles()
        else:
            self.config['make_road'] = True

    def _create_road(self):
        # 创建道路
        net = RoadNetwork()
        none = LineType.NONE
        for i, (k, road) in enumerate(self.roads_dict.items()):
            center = road['center']
            left = road['left']
            right = road['right']
            start_point = (int(road['center'][0][0]), int(road['center'][0][1]))
            end_point = (int(road['center'][-1][0]), int(road['center'][-1][1]))
            net.add_lane(f"{start_point}", f"{end_point}",
                         PolyLane(center, left, right, line_types=(none, none)))

        # 加载人行道
        pedestrian_marking_id = 0
        for k, v in self.laneletmap.items():
            ls = v['type']
            if ls["type"] == "pedestrian_marking":
                pedestrian_marking_id += 1
                ls_points = v['points']
                net.add_lane(f"P_{pedestrian_marking_id}_start", f"P_{pedestrian_marking_id}_end",
                             PolyLaneFixedWidth(ls_points, line_types=(none, none), width=5))

        self.road = Road(network=net, np_random=self.np_random, record_history=self.config["show_trajectories"], lanelet=self.laneletmap, collision_checker=False)

    def process_raw_trajectory(self, trajectory):
        """
        处理原始轨迹，将坐标、速度进行转换。
        :param trajectory: 原始轨迹数据
        :return: 转换后的轨迹数据
        """
        trajectory = np.array(trajectory).copy()
        x, y = trajectory[:, 0].copy(), trajectory[:, 1].copy()
        trajectory[:, 0] = y
        trajectory[:, 1] = x
        headings = trajectory[:, 3].copy()
        headings = np.pi / 2 - headings
        headings = (headings + np.pi) % (2 * np.pi) - np.pi
        trajectory[:, 3] = headings
        return trajectory

    def _create_vehicles(self, reset_time=0):
        """
        创建自车和NGSIM车辆，并将它们添加到道路上。
        """

        T = 100  # 设置T的值
        self.controlled_vehicles = []  # 初始化受控车辆列表
        whole_trajectory = self.ego_trajectory  # 获取整个轨迹
        ego_trajectory = np.array(whole_trajectory[reset_time:])  # 获取自车轨迹
        self.vehicle = HumanLikeVehicle(self.road, 'ego', ego_trajectory[0][:2], ego_trajectory[0][3], ego_trajectory[0][2],
                                        ngsim_traj=ego_trajectory, target_velocity=ego_trajectory[1][2],
                                        v_length=self.trajectory_set['ego']['length'], v_width=self.trajectory_set['ego']['width'])  # 创建自车实例
        # target = self.road.network.get_closest_lane_index(position=ego_trajectory[-1][:2])  # 获取目标车道
        # self.vehicle.plan_route_to(target[1])  # 规划车辆行驶路线
        self.vehicle.color = (50, 200, 0)  # 设置车辆颜色
        self.road.vehicles.append(self.vehicle)  # 将车辆添加到道路上

    def _create_bv_vehicles(self, current_time):

        reset_time = 0
        T = 200

        vehicles = []  # 初始化其他车辆列表
        for veh_id in self.surrounding_vehicles:  # 遍历周围车辆
            try:
                other_trajectory = np.array(self.trajectory_set[veh_id]['trajectory'][reset_time:])  # 获取其他车辆轨迹
                flag = ~(np.array(other_trajectory[current_time])).reshape(1, -1).any(axis=1)[0]  # 判断是否存在轨迹点
                if current_time == 0:  # 如果当前时间为0
                    pass
                else:
                    trajectory = np.array(self.trajectory_set[veh_id]['trajectory'][reset_time:])  # 获取当前时间轨迹点
                    if not flag and ~(np.array(trajectory[current_time-1])).reshape(1, -1).any(axis=1)[0]:  # 如果当前时间和上一时间存在轨迹点
                        flag = False
                    else:
                        flag = True

                if not flag:  # 如果不存在轨迹点

                    mask = np.where(np.sum(other_trajectory[:(T * 10), :2], axis=1) == 0, False, True)  # 过滤无效轨迹点
                    other_trajectory = self.process_raw_trajectory(other_trajectory[mask])
                    other_vehicle = HumanLikeVehicle(self.road, f"{veh_id}", other_trajectory[0][:2], other_trajectory[0][3], other_trajectory[0][2],
                                     ngsim_traj=other_trajectory, target_velocity=other_trajectory[1][2], start_step=self.steps,
                                     v_length=self.trajectory_set[veh_id]['length'],
                                     v_width=self.trajectory_set[veh_id]['width'])
                    if other_vehicle.planned_trajectory.shape[0] <= 5:  # 如果计划轨迹点不足5个，则跳过
                        continue
                    # other_target = self.road.network.get_closest_lane_index(position=other_vehicle.planned_trajectory[-1])  # 获取其他车辆的目标车道
                    # other_vehicle.plan_route_to(other_target[1])  # 规划其他车辆行驶路线

                    if (isinstance(self.bv_ids, str) and veh_id == self.bv_ids) or (isinstance(self.bv_ids, list) and veh_id in self.bv_ids):
                        other_vehicle.color = (0, 0, 255)

                    vehicles.append(other_vehicle)  # 将其他车辆添加到列表中

            except Exception as e:  # 捕获异常
                raise ValueError(f'Error in creating other vehicles, error {e} in {self.data_path}')
        else:
            if len(vehicles) > 0:
                for vh in self.road.vehicles:
                    vehicles.append(vh)  # 将其他车辆添加到列表中
                self.road.vehicles = vehicles  # 将道路上的车辆替换为新的车辆列表

    def _simulate(self, action: Optional[Action] = None) -> None:
        """Perform several steps of simulation with constant action."""
        self._create_bv_vehicles(self.steps)  # 创建虚拟车辆
        self.road.act()
        self.road.step(1 / self.config["simulation_frequency"])
        self.steps += 1
        self._clear_vehicles()  # 清除车辆
        self.enable_auto_render = False  # 关闭自动渲染

    def _reward(self, action):
        return 0.0

    def _is_terminated(self) -> bool:
        """
        Check whether the current state is a terminal state

        :return:is the state terminal
        """
        return self.steps >= self.duration

    def _is_truncated(self) -> bool:
        """
        Check we truncate the episode at the current step

        :return: is the episode truncated
        """
        return False

    def _clear_vehicles(self) -> None:
        """
        清除车辆
        """
        # 判断车辆是否要离开仿真环境
        is_leaving = lambda vehicle: (self.steps >= (
                    vehicle.planned_trajectory.shape[0] + vehicle.start_step - 1) and not vehicle.IDM)

        vehicles = []
        for vh in self.road.vehicles:
            try:
                if vh in self.controlled_vehicles or not is_leaving(vh):  # 如果车辆是受控车辆或不需要离开，则保留在环境中
                    vehicles.append(vh)
            except Exception as e:
                print(e)

        self.road.vehicles = vehicles  # 清除需要离开的车辆
