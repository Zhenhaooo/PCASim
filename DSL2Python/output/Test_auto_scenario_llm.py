import numpy as np
from NGSIM_env.road.lane import StraightLane, LineType
from NGSIM_env.road.road import RoadNetwork, Road
from NGSIM_env.vehicle.behavior import IDMVehicle
from NGSIM_env.vehicle.humandriving import HumanLikeVehicle
from NGSIM_env.vehicle.controller import ControlledVehicle
from NGSIM_env.scenario import Scenario

class IntersectionScenario(Scenario):
    def __init__(self):
        super().__init__()
        self.intersection_start = -15
        self.ego_vehicle = None
        self.adversary1 = None
        self.adversary2 = None

    def _make_road(self):
        net = RoadNetwork()
        lane_width = 3.5
        
        # Create two approach lanes (north to south)
        for lane_idx in range(2):
            origin = [-100, lane_idx * lane_width]
            end = [100, lane_idx * lane_width]
            line_types = (LineType.TEMPORARY, LineType.TEMPORARY)
            lane = StraightLane(origin, end, line_types=line_types, width=lane_width, speed_limit=8)
            net.add_lane("north", "south", lane)
            
        self.road = Road(network=net)
        return self.road

    def _make_vehicles(self):
        # Ego vehicle (right lane)
        ego_lane = self.road.network.get_lane(("north", "south", 0))
        ego_pos = ego_lane.position(self.intersection_start - 25, 0)
        self.ego_vehicle = HumanLikeVehicle(
            road=self.road,
            position=ego_pos,
            speed=6,
            heading=0
        )
        self.road.vehicles.append(self.ego_vehicle)

        # Adversary 1 (left lane)
        left_lane = self.road.network.get_lane(("north", "south", 1))
        adv1_pos = left_lane.position(self.intersection_start - 25, 0)
        self.adversary1 = HumanLikeVehicle(
            road=self.road,
            position=adv1_pos,
            speed=7,
            heading=0
        )
        self.road.vehicles.append(self.adversary1)

        # Adversary 2 (same lane as ego)
        adv2_pos = ego_lane.position(self.intersection_start - 30, 0)
        self.adversary2 = HumanLikeVehicle(
            road=self.road,
            position=adv2_pos,
            speed=8,
            heading=0
        )
        self.road.vehicles.append(self.adversary2)

        # Background vehicles
        for _ in range(20):
            lane = self.road.network.random_lane()
            v = IDMVehicle(
                road=self.road,
                position=lane.position(0, 0),
                speed=np.random.uniform(0.8*lane.speed_limit, 1.2*lane.speed_limit),
                heading=lane.heading_at(0)
            )
            self.road.vehicles.append(v)
            
        return self.road.vehicles

    def _update_controls(self):
        # Ego braking behavior
        if self.ego_vehicle.position[0] < self.intersection_start:
            self.ego_vehicle.control = ControlledVehicle.keep_lane()
        else:
            self.ego_vehicle.control = ControlledVehicle.brake(intensity=0.65)

        # Adversary 1 behavior
        self.adversary1.target_speed = self.ego_vehicle.speed * 1.25
        self.adversary1.control = ControlledVehicle.lane_change(
            direction='right',
            aggressive=True,
            min_lateral_distance=1.05
        )

        # Adversary 2 behavior
        self.adversary2.target_speed = self.ego_vehicle.speed * 1.3
        self.adversary2.control = ControlledVehicle.lane_change(
            direction='left',
            aggressive=True,
            min_lateral_distance=1.00
        )

    def step(self, action=None):
        self._update_controls()
        super().step(action)
        return self.road.vehicles

if __name__ == '__main__':
    from NGSIM_env.environment import HighwayEnvironment
    
    scenario = IntersectionScenario()
    env = HighwayEnvironment(scenario=scenario)
    env.reset()
    
    for _ in range(500):  # Run simulation for 500 steps
        env.step()
        env.render()