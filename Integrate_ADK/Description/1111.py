import highway_env
from highway_env.envs.common.abstract import AbstractEnv
from highway_env.road.road import Road, RoadNetwork
from highway_env.envs.common.action import Action
from highway_env.vehicle.controller import ControlledVehicle
from highway_env.vehicle.kinematics import Vehicle
from stable_baselines3 import PPO
import numpy as np

LANES = 2
ANGLE = 0
START = 0
LENGHT = 200
SPEED_LIMIT = 30
SPEED_REWARD_RANGE = [10, 30]
COL_REWARD = -1
HIGH_SPEED_REWARD = 0
RIGHT_LANE_REWARD = 0
DURATION = 500.0


class myEnv(AbstractEnv):

    @classmethod
    def default_config(cls) -> dict:
        config = super().default_config()
        config.update(
            {
                'observation': {
                    'type': 'Kinematics',
                    "absolute": True,
                    "normalize": False,
                },
                'action': {'type': 'DiscreteMetaAction',
                           'target_speeds': np.array([10, 20, 30])  # 目标速度设置
                           },
                "features": ["x", "y", "vx", "vy"],
                "features_range": {
                    "x": [-10000, 10000],
                    "y": [-10000, 10000],
                    "vx": [-10, 10],
                    "vy": [-10, 30]},

                "reward_speed_range": SPEED_REWARD_RANGE,
                "simulation_frequency": 20,
                "policy_frequency": 20,
                'screen_width': 1200,
                'screen_height': 150,
                # "other_vehicles": 5,
                # "centering_position": [0.3, 0.5],
            }
        )
        return config

    def _reset(self) -> None:
        self._create_road()
        self._create_vehicles()

    def _create_road(self) -> None:
        self.road = Road(
            network=RoadNetwork.straight_road_network(LANES, speed_limit=SPEED_LIMIT),
            np_random=self.np_random,
            record_history=False,
        )

    # 创建车辆
    def _create_vehicles(self) -> None:

        # 自车
        vehicle = Vehicle.create_random(self.road, speed=27, lane_id=1, spacing=0.3)
        vehicle = self.action_type.vehicle_class(
            self.road,
            vehicle.position,
            vehicle.heading,
            vehicle.speed,
        )
        self.vehicle = vehicle
        self.road.vehicles.append(vehicle)

        # 周围车辆1
        vehicle = Vehicle.create_random(self.road, speed=22, lane_id=1, spacing=0.35)
        vehicle = self.action_type.vehicle_class(
            self.road,
            vehicle.position + [75, 0],  # 初始化目标场景，不加这一项的话，默认距离为5
            vehicle.heading,
            vehicle.speed,
        )
        self.road.vehicles.append(vehicle)
        vehicle.color = (255, 255, 255)

    # 重写的奖励函数
    def _reward(self, action: Action) -> float:
        reward_all = 0

        # 速度奖励，通过惩罚使得最低车速为20
        r_speed = (1 / 10) * (self.vehicle.speed - 20)

        # 目标车道奖励
        ego_lane = self.vehicle.lane_index[2]
        if ego_lane == 1:
            r_target_lane = 0
        else:
            r_target_lane = 0.1

        # 碰撞后奖励
        if self.vehicle.crashed:
            print("撞上了！")
            r_collsion = -180
        else:
            r_collsion = 0

        # 车辆横向位移偏出车道奖励
        r_out = 0
        r_change_lane = 0
        ego_pos_x = self.vehicle.position[1]

        if ego_pos_x > 4 and ego_pos_x < 0:
            r_target_lane = 0
            r_out = -200
            print("车离开车道！")
            self.vehicle.crashed = True  # 与路沿碰撞
            return r_speed + r_out + r_collsion + r_target_lane

        reward_all = r_speed + r_out + r_collsion + r_target_lane

        return reward_all

    def _is_terminal(self) -> bool:

        if self.vehicle.crashed:
            return True

        if self.time >= DURATION:
            print("时间到了！")
            return True

        if False and not self.vehicle.on_road:
            print("车不在路上了！")
            return True

        return False


if __name__ == '__main__':
    env = myEnv()

    model = PPO('MlpPolicy', env,
                policy_kwargs=dict(net_arch=[256, 256]),
                learning_rate=5e-4,
                batch_size=32,
                gamma=0.8,
                verbose=1,
                tensorboard_log="highway_dqn/")
    model.learn(int(1e5))
    model.save("highway_dqn/PPOmodel")
