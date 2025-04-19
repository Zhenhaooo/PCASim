from NGSIM_env.envs.common.abstract import AbstractEnv
from NGSIM_env.road.road import Road, RoadNetwork
from NGSIM_env.vehicle.behavior import IDMVehicle
from NGSIM_env.vehicle.humandriving import HumanLikeVehicle, InterActionVehicle
from NGSIM_env.road.lane import StraightLane
import numpy as np

class IntersectionScenario(AbstractEnv):
    def __init__(self):
        super().__init__()
        self.config = {
            "observation": {"type": "TimeToCollision", "horizon": 5},
            "duration": 40,
            "policy_frequency": 5
        }
        self.make_road()
        self.make_vehicles()

    def make_road(self):
        self.road_network = RoadNetwork.from_real_dataset(
            dataset='vehicle_tracks_000_trajectory_set_28.json',
            lane_types=[StraightLane, StraightLane],
            lane_speed_limits={'fast': 45.0},
            temporary_control=True
        )
        self.road = Road(network=self.road_network,
                        np_random=self.np_random,
                        collision_checker=True)

    def make_vehicles(self):
        # Ego vehicle
        ego_lane = self.road_network.get_lane(lane_index=0)
        ego_vehicle = HumanLikeVehicle(
            road=self.road,
            position=ego_lane.center_at(40.0),
            heading=ego_lane.heading_at(40.0),
            speed=40.0,
            target_speed=40.0
        )
        ego_vehicle.behavior = IDMVehicle(
            self.road,
            position=ego_vehicle.position,
            speed=ego_vehicle.speed,
            target_speed=40.0,
            emergency_decel=-5.8,
            braking_threshold=1.8
        )
        self.road.vehicles.append(ego_vehicle)
        self.vehicle = ego_vehicle

        # Adversarial Vehicle 1 (left side unsafe lane change)
        adv1_lane = self.road_network.get_lane(lane_index=1)
        adv1_position = ego_vehicle.position + np.array([0.0, 3.61])
        adv1 = InterActionVehicle(
            road=self.road,
            position=adv1_position,
            heading=adv1_lane.heading_at(40.0),
            speed=38.0,
            target_lane_index=0
        )
        adv1.lateral_behavior = {
            "type": "SuddenLaneChangeBehavior",
            "direction": "right",
            "lateral_accel": 0.85,
            "trigger_distance": 15.0,
            "max_offset": 3.61
        }
        self.road.vehicles.append(adv1)

        # Adversarial Vehicle 2 (rear sudden brake)
        adv2_lane = self.road_network.get_lane(lane_index=0)
        adv2_position = ego_vehicle.position + np.array([-18.0, 0.0])
        adv2 = InterActionVehicle(
            road=self.road,
            position=adv2_position,
            heading=ego_lane.heading_at(40.0 - 18.0),
            speed=42.0,
            target_speed=42.0
        )
        adv2.longitudinal_behavior = {
            "type": "SuddenBrakeBehavior",
            "reaction_time": 2.75,
            "decel_profile": [-3.5, -4.2, -6.0],
            "activation_speed": 42.0
        }
        self.road.vehicles.append(adv2)

        # Add background traffic
        for _ in range(17):  # Total 20 vehicles including main actors
            lane_idx = np.random.choice([0, 1])
            vehicle = IDMVehicle.make_on_lane(
                self.road,
                lane_index=lane_idx,
                longitudinal=np.random.uniform(0, 100),
                speed=np.random.uniform(35, 45)
            )
            self.road.vehicles.append(vehicle)

if __name__ == '__main__':
    from NGSIM_env.utils.graphics import EnvViewer
    
    scenario = IntersectionScenario()
    viewer = EnvViewer(scenario)
    viewer.zoom = 10
    done = False
    
    while not done:
        scenario.act()
        scenario.step()
        viewer.display()
        done = scenario.config["duration"] * scenario.config["policy_frequency"] <= scenario.steps