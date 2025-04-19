import math
from abc import ABC, abstractmethod, abstractproperty
import numpy as np
import pandas as pd
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
import pickle
import json


class Record(ABC):
    """某时刻特征"""

    def __init__(self, **kwargs):
        # 使用 **kwargs 接收不定数量的参数，并将其以键值对的形式保存在对象的实例变量中
        # 根据传入的参数更新对象的属性
        for k, v in kwargs.items():
            self.__setattr__(k, v)

        # 初始化对象的一些属性
        self.id = None  # 记录的唯一 ID
        self.veh_id = None  # 车辆 ID
        self.ped_id = None  # 行人 ID
        self.frame_ID = None  # 帧 ID
        self.unix_time = None  # Unix 时间戳

    def get(self, k):
        # 定义一个 get 方法，用来获取对象的属性
        return self.__getattribute__(k)

    def set(self, k, v):
        # 定义一个 set 方法，用来设置对象的属性
        self.__setattr__(k, v)

    def __deepcopy__(self):
        # 定义一个深拷贝方法，用来创建并返回对象的深度副本
        cls = self.__class__
        result = cls.__new__(cls)
        for k, v in self.__dict__.items():
            setattr(result, k, v)
        return result


class SnapShot(ABC):
    """某时刻的所有车辆"""

    def __init__(self, **kwargs):
        # 使用 **kwargs 接收不定数量的参数，并将其以键值对的形式保存在对象的实例变量中
        # 根据传入的参数更新对象的属性
        for k, v in kwargs.items():
            self.__setattr__(k, v)

        # 初始化对象的一些属性
        self.unix_time = None  # Unix 时间戳
        self.vr_list = list()  # 保存车辆记录的列表

    def get(self, k):
        # 定义一个 get 方法，用来获取对象的属性
        return self.__getattribute__(k)

    def set(self, k, v):
        # 定义一个 set 方法，用来设置对象的属性
        self.__setattr__(k, v)

    def add_vehicle_record(self, record):
        # 定义一个方法，用来向车辆记录列表中添加记录
        self.vr_list.append(record)

    # @abstractmethod
    # def sort_vehs(self, ascending=True):
    #     raise NotImplementedError


GLB_TIME_THRES = 20000


class Vehicle(ABC):
    """某辆车的所有特征"""

    def __init__(self, **kwargs):
        # 使用 **kwargs 接收不定数量的参数，并将其以键值对的形式保存在对象的实例变量中
        # 根据传入的参数更新对象的属性
        for k, v in kwargs.items():
            self.__setattr__(k, v)

        # 初始化对象的一些属性
        self.veh_id = None  # 车辆 ID
        self.vr_list = list()  # 保存车辆记录的列表
        self.trajectory = list()  # 保存车辆轨迹的列表
        self.trajectory_ = list()  # 保存划分完成的车辆轨迹的列表

    def get(self, k):
        # 定义一个 get 方法，用来获取对象的属性
        return self.__getattribute__(k)

    def set(self, k, v):
        # 定义一个 set 方法，用来设置对象的属性
        self.__setattr__(k, v)

    def build_trajectory(self, trajectory: list):
        # 定义一个方法，用来构建车辆轨迹
        self.trajectory.extend(trajectory)

    def add_vehicle_record(self, record: Record):
        # 定义一个方法，用来向车辆记录列表中添加记录
        self.vr_list.append(record)

    def build_trajectory_(self):
        # 定义一个方法，用来划分车辆轨迹

        vr_list = self.vr_list
        assert (len(vr_list) > 0)

        self.trajectory_ = list()  # 清空之前的划分结果
        cur_time = vr_list[0].unix_time
        tmp_trj = [vr_list[0]]

        for tmp_vr in vr_list[1:]:
            if tmp_vr.unix_time - cur_time > GLB_TIME_THRES:  # 如果时间戳大于阈值，则将当前轨迹片段添加到轨迹列表中，并开始构建新的轨迹片段
                if len(tmp_trj) > 1:
                    self.trajectory_.append(tmp_trj)
                tmp_trj = [tmp_vr]
            else:
                tmp_trj.append(tmp_vr)
            cur_time = tmp_vr.unix_time

        if len(tmp_trj) > 1:  # 处理最后一个轨迹片段
            self.trajectory_.append(tmp_trj)


class Pedestrian(ABC):
    """行人的所有特征"""

    def __init__(self, **kwargs):
        # 使用 **kwargs 接收不定数量的参数，并将其以键值对的形式保存在对象的实例变量中
        # 根据传入的参数更新对象的属性
        for k, v in kwargs.items():
            self.__setattr__(k, v)

        # 初始化对象的一些属性
        self.ped_id = None  # 行人 ID
        self.vr_list = list()  # 保存行人记录的列表
        self.trajectory = list()  # 保存行人轨迹的列表
        self.trajectory_ = list()  # 保存划分完成的行人轨迹的列表

    def get(self, k):
        # 定义一个 get 方法，用来获取对象的属性
        return self.__getattribute__(k)

    def set(self, k, v):
        # 定义一个 set 方法，用来设置对象的属性
        self.__setattr__(k, v)

    def build_trajectory(self, trajectory: list):
        # 定义一个方法，用来构建行人轨迹
        self.trajectory.extend(trajectory)

    def add_vehicle_record(self, record: Record):
        # 定义一个方法，用来向行人记录列表中添加记录
        self.vr_list.append(record)

    def build_trajectory_(self):
        # 定义一个方法，用来划分行人轨迹

        vr_list = self.vr_list
        assert (len(vr_list) > 0)

        self.trajectory_ = list()  # 清空之前的划分结果
        cur_time = vr_list[0].unix_time
        tmp_trj = [vr_list[0]]

        for tmp_vr in vr_list[1:]:
            if tmp_vr.unix_time - cur_time > GLB_TIME_THRES:  # 如果时间戳大于阈值，则将当前轨迹片段添加到轨迹列表中，并开始构建新的轨迹片段
                if len(tmp_trj) > 1:
                    self.trajectory_.append(tmp_trj)
                tmp_trj = [tmp_vr]
            else:
                tmp_trj.append(tmp_vr)
            cur_time = tmp_vr.unix_time

        if len(tmp_trj) > 1:  # 处理最后一个轨迹片段
            self.trajectory_.append(tmp_trj)


class DataProcess(ABC):
    """提取数据集的所有车辆、行人特征"""

    def __init__(self,
                 config: dict = None,
                 snapshot=SnapShot,
                 vehicle=Vehicle,
                 vehicle_record=Record,
                 pedestrian=Pedestrian,
                 # ped_record=PedestrianRecord,
                 ):
        '''
        DataProcess类的构造函数。

        Args:
        - config: dict, 配置参数字典。
        - snapshot: SnapShot类的实例，用于存储数据快照。
        - vehicle: Vehicle类的实例，表示车辆。
        - vehicle_record: Record类的实例，表示车辆信息记录。
        - pedestrian: Pedestrian类的实例，表示行人或自行车。
        '''

        self.config = self.default_config()
        if config:
            self.config.update(config)

        self.snapshot = snapshot
        self.vehicle = vehicle
        self.vehicle_record = vehicle_record

        self.vr_dict = dict()  # 车辆信息记录字典
        self.snap_dict = dict()  # 数据快照字典
        self.veh_dict = dict()  # 车辆字典

        self.pedestrian = pedestrian
        # self.ped_record = ped_record
        self.ped_dict = dict()
        self.ped_index = None

        self.snap_ordered_list = list()  # 排序后的数据快照列表
        self.veh_ordered_list = list()  # 排序后的车辆列表
        self.ped_ordered_list = list()  # 排序后的行人列表

        self.pd_data = None  # 存储读取的数据(pandas.DataFrame)用于存储车辆数据
        self.ped_data = None  # 存储读取的数据(pandas.DataFrame)用于存储行人或自行车数据

    @classmethod
    def default_config(cls):
        '''
        默认配置参数。

        Returns:
        - dict, 默认配置参数字典。
        '''
        return {
            "dataset_path": None,
            "coordinate_x_name": "",
            "coordinate_y_name": "",
            "speed_x_name": "",
            "speed_y_name": "",
            "time_name": "",
            "frame_id_name": "",
            "vehicle_id_name": "",
            'length_name': '',
            'width_name': '',
            'heading_name': '',
            'process_pedestrian': False
        }

    @abstractmethod
    def read_data(self):
        '''
        读取数据的抽象方法。
        '''
        raise NotImplementedError

    @classmethod
    def separate_data_by_key_value(cls, data, key, value):
        '''
        根据指定的键和值分离数据。

        Args:
        - data: pd.DataFrame, 数据集。
        - key: str, 键名。
        - value: Any, 值。

        Returns:
        - pd.DataFrame, 分离后的数据集。
        '''
        try:
            data_copy = data[data[key] == value].copy()
        except Exception as e:
            raise e
        else:
            return data_copy

    def make_dataframe_become_dict(self, df: pd.DataFrame) -> dict:
        '''
        将dataframe格式的数据转换成字典。

        Args:
        - df: pd.DataFrame, 数据集。

        Returns:
        - dict, 转换后的字典，其中key为(vehicle_id, total_frame)元组，value为该车辆的所有信息。
        '''

        index = self.get_vehicles_set(df)
        dict_data = dict()
        for idx in index:
            vehicle_data = self.get_the_only_vehicle(df, idx)
            dict_data[idx] = vehicle_data
        if self.config['process_pedestrian']:
            self.ped_index = self.get_ped_set(self.ped_data)
            for idx in self.ped_index:
                ped_data = self.get_the_only_vehicle(self.ped_data, idx)
                dict_data[idx] = ped_data
        return dict_data

    def get_the_only_vehicle(self, df: pd.DataFrame, vehicle_id: int) -> pd.DataFrame:
        '''
        获取具有指定vehicle_id的车辆的DataFrame。

        Args:
        - df: pd.DataFrame, 数据集。
        - vehicle_id: int, 车辆ID。

        Returns:
        - pd.DataFrame, DataFrame中符合条件的车辆数据。
        '''
        return df[df[self.config["vehicle_id_name"]] == vehicle_id].copy()

    def get_vehicles_set(self, df) -> set:
        '''
        获取数据集中车辆的唯一标识。

        Args:
        - df: pd.DataFrame, 数据集。

        Returns:
        - set, 车辆唯一标识集合。
        '''
        index = df[[self.config["vehicle_id_name"]]].values.tolist()
        index = set([idx[0] for idx in index])
        return index

    def get_ped_set(self, df) -> set:
        '''
        获取数据集中行人的唯一标识。

        Args:
        - df: pd.DataFrame, 数据集。

        Returns:
        - set, 行人唯一标识集合。
        '''
        index = df[[self.config["vehicle_id_name"]]].values.tolist()
        index = set([idx[0] for idx in index])
        sorted_index = sorted(index, key=lambda x: int(x[1:]))  # 从P后的部分提取数字并排序

        return sorted_index

    def process(self):
        '''
        数据处理方法。
        '''

        # 获取数据的列名
        list_name_vehicle = self.pd_data.columns.tolist()

        # 将DataFrame转换为字典
        pd_dict = self.make_dataframe_become_dict(self.pd_data)

        print('Processing raw data...')

        counter = 0
        self.vr_dict = dict()
        self.snap_dict = dict()
        self.veh_dict = dict()

        # 遍历每个键值对，其中k为字典的键，vehicle为字典的值
        for k, vehicle in pd_dict.items():
            # 对vehicle进行拷贝并排序
            vehicle_copy = vehicle.copy().sort_values(by=self.config['frame_id_name']).reset_index(drop=True)

            # 如果配置文件中设置了处理行人数据，并且k在ped_index中
            if self.config['process_pedestrian'] and k in self.ped_index:
                # 获取行人数据的列名
                list_name_ped = self.ped_data.columns.tolist()

                # 创建行人对象
                vh = self.pedestrian()

                # 设置行人ID
                vh.set('ped_id', vehicle_copy.loc[0, self.config["vehicle_id_name"]])

                # 构建行人轨迹
                vh.build_trajectory(vehicle_copy.loc[:, [self.config['coordinate_x_name'],
                                                         self.config['coordinate_y_name']]].to_numpy().tolist())

                # 遍历vehicle的每一行
                for i in range(vehicle.shape[0]):
                    vrecord = dict()
                    for n in list_name_ped:
                        vrecord[n] = vehicle_copy.loc[i, n]

                    # 计算速度向量的x和y分量，并计算方向角度
                    else:
                        vx = vrecord.get(self.config['speed_x_name'])
                        vy = vrecord.get(self.config['speed_y_name'])
                        vrecord[self.config['heading_name']] = math.atan2(vy, vx)

                        # 设置车辆宽度和长度
                        vrecord['width'] = 0.5
                        vrecord['length'] = 0.5

                        # 创建车辆记录对象
                        tmp_vr = self.vehicle_record(**vrecord)
                        tmp_vr.set('id', counter)
                        tmp_vr.set('ped_id', vehicle_copy.loc[i, self.config["vehicle_id_name"]])
                        tmp_vr.set('frame_ID', int(vehicle_copy.loc[i, self.config["frame_id_name"]]))
                        tmp_vr.set('unix_time', int(vehicle_copy.loc[i, self.config["time_name"]]))

                        # 将车辆记录对象添加到vr_dict中
                        self.vr_dict[tmp_vr.id] = tmp_vr
                        counter += 1

                        # 如果unix_time不在snap_dict的键中，则创建一个快照对象，并将其添加到snap_dict中
                        if tmp_vr.unix_time not in self.snap_dict.keys():
                            ss = self.snapshot()
                            ss.set('unix_time', tmp_vr.unix_time)
                            self.snap_dict[tmp_vr.unix_time] = ss

                        # 将车辆记录对象添加到对应的快照对象和行人对象中
                        self.snap_dict[tmp_vr.unix_time].add_vehicle_record(tmp_vr)
                        vh.add_vehicle_record(tmp_vr)
                else:
                    self.veh_dict[k] = vh
            else:
                # 创建车辆对象
                vh = self.vehicle()

                # 设置车辆ID
                vh.set('veh_id', vehicle_copy.loc[0, self.config["vehicle_id_name"]])

                # 构建车辆轨迹
                vh.build_trajectory(vehicle_copy.loc[:, [self.config['coordinate_x_name'],
                                                         self.config['coordinate_y_name']]].to_numpy().tolist())

                # 遍历vehicle的每一行
                for i in range(vehicle.shape[0]):
                    vrecord = dict()
                    for n in list_name_vehicle:
                        vrecord[n] = vehicle_copy.loc[i, n]

                    # 创建车辆记录对象
                    tmp_vr = self.vehicle_record(**vrecord)
                    tmp_vr.set('id', counter)
                    tmp_vr.set('veh_id', vehicle_copy.loc[i, self.config["vehicle_id_name"]])
                    tmp_vr.set('frame_ID', int(vehicle_copy.loc[i, self.config["frame_id_name"]]))
                    tmp_vr.set('unix_time', int(vehicle_copy.loc[i, self.config["time_name"]]))

                    # 将车辆记录对象添加到vr_dict中
                    self.vr_dict[tmp_vr.id] = tmp_vr
                    counter += 1

                    # 如果unix_time不在snap_dict的键中，则创建一个快照对象，并将其添加到snap_dict中
                    if tmp_vr.unix_time not in self.snap_dict.keys():
                        ss = self.snapshot()
                        ss.set('unix_time', tmp_vr.unix_time)
                        self.snap_dict[tmp_vr.unix_time] = ss

                    # 将车辆记录对象添加到对应的快照对象和车辆对象中
                    self.snap_dict[tmp_vr.unix_time].add_vehicle_record(tmp_vr)
                    vh.add_vehicle_record(tmp_vr)
                else:
                    self.veh_dict[k] = vh
        else:
            # 将snap_dict的键排序后，存储到snap_ordered_list中
            self.snap_ordered_list = list(self.snap_dict.keys())
            self.snap_ordered_list.sort()

            # 将veh_dict的键排序后，存储到veh_ordered_list中
            self.veh_ordered_list = list(self.veh_dict.keys())
            # self.veh_ordered_list.sort()

    # 可对该方法进行重载
    def __call__(self, *args, **kwargs):

        # 读数据
        self.read_data()
        # 数据处理
        self.process()

    def build_trajecotry(self, period, vehicle_id):
        assert vehicle_id in list(self.veh_dict.keys())  # 确保vehicle_id在veh_dict字典的键中

        surroundings = []  # 存储周围车辆信息的列表
        record_trajectory = {'ego': {'length': 0, 'width': 0, 'trajectory': []}}  # 初始化记录轨迹的字典

        for veh_ID, v in self.veh_dict.items():
            v.build_trajectory_()  # 为车辆字典中的车辆构建轨迹

        ego_trajectories = self.veh_dict[vehicle_id].trajectory_  # 获取自车的轨迹列表
        selected_trajectory = ego_trajectories[period]  # 获取指定时间段的自车轨迹

        D = 100  # 周围车辆的范围

        ego = []  # 存储自车信息的列表
        nearby_IDs = []  # 存储周围车辆id的列表

        for position in selected_trajectory:
            record_trajectory['ego']['length'] = position.get(self.config['length_name'])  # 记录自车的长度
            record_trajectory['ego']['width'] = position.get(self.config['width_name'])  # 记录自车的宽度
            speed = np.linalg.norm(
                [position.get(self.config['speed_x_name']), position.get(self.config['speed_y_name'])])  # 计算自车的速度
            ego.append(
                [position.get(self.config['coordinate_x_name']), position.get(self.config['coordinate_y_name']), speed,
                 position.get(self.config['heading_name']),
                 position.get(self.config['time_name'])])  # 将自车的位置、速度、朝向、时间戳添加到ego列表中

            records = self.snap_dict[position.unix_time].vr_list  # 获取指定时间点的车辆记录
            other = []  # 存储其他车辆信息的列表
            for record in records:
                if record.get(self.config['vehicle_id_name']) != vehicle_id:
                    other.append(
                        [record.get(self.config['vehicle_id_name']), record.get(self.config['length_name']),
                         record.get(self.config['width_name']), record.get(self.config['coordinate_x_name']),
                         record.get(self.config['coordinate_y_name']), record.get(self.config['heading_name']),
                         record.get(self.config['speed_x_name']),
                         record.get(self.config['speed_y_name']),
                         record.get(self.config['time_name'])])  # 将其他车辆的相关信息添加到other列表中

                    # 构建了包含 x 和 y 坐标差的列表，以计算两点之间的直线距离
                    d = np.linalg.norm([
                        position.get(self.config['coordinate_x_name']) - record.get(self.config['coordinate_x_name']),
                        position.get(self.config['coordinate_y_name']) - record.get(self.config['coordinate_y_name'])
                    ])

                    if d <= D:
                        nearby_IDs.append(record.get(self.config['vehicle_id_name']))  # 若距离小于等于D，则将车辆id添加到nearby_IDs列表中

            surroundings.append(other)  # 添加其他车辆信息列表到surroundings列表中

        record_trajectory['ego']['trajectory'] = ego  # 将自车轨迹添加到record_trajectory字典中

        for v_ID in set(nearby_IDs):
            record_trajectory[v_ID] = {'length': 0, 'width': 0,
                                       'trajectory': []}  # 若车辆id在nearby_IDs中的唯一值集合中，则在record_trajectory字典中为该车辆id初始化相应的key-value对

        # 填充数据
        for timestep_record in surroundings:
            scene_IDs = []  # 存储每个时间片中实际出现的车辆id的列表
            for vehicle_record in timestep_record:
                v_ID = vehicle_record[0]
                v_length = vehicle_record[1]
                v_width = vehicle_record[2]
                v_x = vehicle_record[3]
                v_y = vehicle_record[4]
                v_heading = vehicle_record[5]
                speed = np.linalg.norm([vehicle_record[6], vehicle_record[7]])  # 计算车辆的速度
                time_stamp = vehicle_record[8]
                if v_ID in set(nearby_IDs):
                    scene_IDs.append(v_ID)
                    record_trajectory[v_ID]['length'] = v_length
                    record_trajectory[v_ID]['width'] = v_width
                    record_trajectory[v_ID]['trajectory'].append([v_x, v_y, speed,
                                                                  v_heading,
                                                                  time_stamp])  # 添加车辆的位置、速度和朝向到record_trajectory字典中，如果车辆不在当前场景ID集合中则添加[0, 0, 0, 0]作为缺失值

            for v_ID in set(nearby_IDs):
                if v_ID not in scene_IDs:
                    record_trajectory[v_ID]['trajectory'].append([0, 0, 0, 0, 0])

        return record_trajectory

    def build_pedestrian_trajecotry(self, period, p_id):
        assert p_id in list(self.ped_dict.keys())  # 确保p_id在ped_dict字典的键中

        surroundings = []  # 存储周围车辆信息的列表
        record_trajectory = {'ego': {'length': 0, 'width': 0, 'trajectory': []}}  # 初始化记录轨迹的字典

        for veh_ID, v in self.ped_dict.items():
            v.build_trajectory_()  # 为行人字典中的行人构建轨迹

        ego_trajectories = self.ped_dict[p_id].trajectory_  # 获取行人的轨迹列表
        selected_trajectory = ego_trajectories[period]  # 获取指定时间段的行人轨迹

        D = 50  # 周围车辆的范围

        ego = []  # 存储行人信息的列表
        nearby_IDs = []  # 存储周围车辆id的列表

        for position in selected_trajectory:
            record_trajectory['ego']['length'] = position.get(self.config['length_name'])  # 记录行人的长度
            record_trajectory['ego']['width'] = position.get(self.config['width_name'])  # 记录行人的宽度
            speed = np.linalg.norm(
                [position.get(self.config['speed_x_name']), position.get(self.config['speed_y_name'])])  # 计算行人的速度
            ego.append(
                [position.get(self.config['coordinate_x_name']), position.get(self.config['coordinate_y_name']), speed,
                 position.get(self.config['heading_name'])])  # 将行人的位置、速度和朝向添加到ego列表中

            records = self.snap_dict[position.unix_time].vr_list  # 获取指定时间点的车辆记录
            other = []  # 存储其他车辆信息的列表
            for record in records:
                if record.get(self.config['vehicle_id_name']) != p_id:
                    other.append(
                        [record.get(self.config['vehicle_id_name']), record.get(self.config['length_name']),
                         record.get(self.config['width_name']), record.get(self.config['coordinate_x_name']),
                         record.get(self.config['coordinate_y_name']), record.get(self.config['heading_name']),
                         record.get(self.config['speed_x_name']),
                         record.get(self.config['speed_y_name'])])  # 将其他车辆的相关信息添加到other列表中

                    d = abs(
                        position.get(self.config['coordinate_y_name']) - record.get(self.config['coordinate_y_name']))
                    if d <= D:
                        nearby_IDs.append(record.get(self.config['vehicle_id_name']))  # 若距离小于等于D，则将车辆id添加到nearby_IDs列表中
            surroundings.append(other)  # 添加其他车辆信息列表到surroundings列表中

        record_trajectory['ego']['trajectory'] = ego  # 将行人轨迹添加到record_trajectory字典中

        for v_ID in set(nearby_IDs):
            record_trajectory[v_ID] = {'length': 0, 'width': 0,
                                       'trajectory': []}
            # 若车辆id在nearby_IDs中的唯一值集合中，则在record_trajectory字典中为该车辆id初始化相应的key-value对

        # 填充数据
        for timestep_record in surroundings:
            scene_IDs = []  # 存储每个时间片中实际出现的车辆id的列表
            for vehicle_record in timestep_record:
                v_ID = vehicle_record[0]
                v_length = vehicle_record[1]
                v_width = vehicle_record[2]
                v_x = vehicle_record[3]
                v_y = vehicle_record[4]
                v_heading = vehicle_record[5]
                speed = np.linalg.norm([vehicle_record[6], vehicle_record[7]])  # 计算车辆的速度
                if v_ID in set(nearby_IDs):
                    scene_IDs.append(v_ID)
                    record_trajectory[v_ID]['length'] = v_length
                    record_trajectory[v_ID]['width'] = v_width
                    record_trajectory[v_ID]['trajectory'].append([v_x, v_y, speed,
                                                                  v_heading])  # 添加车辆的位置、速度和朝向到record_trajectory字典中，如果车辆不在当前场景ID集合中则添加[0, 0, 0, 0]作为缺失值
            for v_ID in set(nearby_IDs):
                if v_ID not in scene_IDs:
                    record_trajectory[v_ID]['trajectory'].append([0, 0, 0, 0])

        # 轨迹平滑
        # for key in record_trajectory.keys():
        #     orginal_trajectory = record_trajectory[key]['trajectory']
        # smoothed_trajectory = trajectory_smoothing(orginal_trajectory)
        # record_trajectory[key]['trajectory'] = smoothed_trajectory

        return record_trajectory
