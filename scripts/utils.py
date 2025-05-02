from __future__ import division, print_function, absolute_import

import lanelet2
from threading import Thread
from env.NGSIM_env.road.road import Road, RoadNetwork
from env.NGSIM_env.road.lane import LineType, PolyLane, PolyLaneFixedWidth
from env.NGSIM_env import utils
import numpy as np
from env.NGSIM_env.envs.common.abstract import AbstractEnv
from env.NGSIM_env.vehicle.dynamics import Vehicle
from env.NGSIM_env.road.graphics import RoadGraphics, WorldSurface
from env.NGSIM_env.vehicle.humandriving import HumanLikeVehicle
from env.NGSIM_env.vehicle.graphics import VehicleGraphics

# 读入osm地图转为路网
class OSMRoadNetworkLoader:
    def __init__(self, osm_path, show_trajectories=False):
        self.osm_path = osm_path
        self.show_trajectories = show_trajectories
        self.roads_dict = None
        self.laneletmap = None
        self.road = None

    def load_map(self):
        """读取 OSM 地图并解析为 Lanelet2 格式"""
        projector = lanelet2.projection.UtmProjector(lanelet2.io.Origin(0.0, 0.0))
        laneletmap = lanelet2.io.load(self.osm_path, projector)
        # 你可以设置大致原点为：
        self.roads_dict, self.graph, self.laneletmap, self.indegree, self.outdegree = utils.load_lanelet_map(laneletmap)

    def create_road_network(self):
        """基于 roads_dict 和 laneletmap 构建 RoadNetwork"""
        # print("🚦 生成车道数量：", len(self.roads_dict)) # 调试
        net = RoadNetwork()
        none = LineType.NONE

        # 添加普通车道
        for i, (k, road) in enumerate(self.roads_dict.items()):
            center = road['center']
            left = road['left']
            right = road['right']
            start_point = (int(center[0][0]), int(center[0][1]))
            end_point = (int(center[-1][0]), int(center[-1][1]))
            net.add_lane(f"{start_point}", f"{end_point}",
                         PolyLane(center, left, right, line_types=(none, none)))

        # 添加人行道或其他行人标记车道
        pedestrian_marking_id = 0
        for k, v in self.laneletmap.items():
            ls = v['type']
            if ls.get("type") == "pedestrian_marking":
                pedestrian_marking_id += 1
                ls_points = v['points']
                net.add_lane(f"P_{pedestrian_marking_id}_start", f"P_{pedestrian_marking_id}_end",
                             PolyLaneFixedWidth(ls_points, line_types=(none, none), width=5))

        # 构建最终 Road 实例
        self.road = Road(network=net, np_random=None, record_history=self.show_trajectories,
                         lanelet=self.laneletmap)

    def get_road(self):
        return self.road

def process_raw_trajectory(trajectory):
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