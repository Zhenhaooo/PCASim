import numpy as np
from highway_env import Road, RoadNetwork
from highway_env.vehicle.behavior import IDMVehicle
from highway_env.vehicle.controller import ControlledVehicle
from highway_env.road.regulation import RegulatedRoad
from highway_env.road.lane import CircularLane, StraightLane, AbstractLane
from highway_env.envs.common import AbstractEnv
from highway_env.utils import Vector
from typing import List, Tuple

class ConstrainedNavigation(IDMVehicle):
    def __init__(self, road: Road, position: Vector, heading: float = 0, speed: float = 8):
        super().__init__(road, position, heading, speed)
        self.obstacle_buffer = 5.0
        self.construction_adjustment = 0.7
        self.perception_range = 40
        self.base_speed = speed

    def step(self, dt: float):
        obstacles = self.detect_obstacles()
        
        if self.emergency_stop_required(obstacles):
            self.execute_full_stop()
        elif self.in_construction_zone():
            self.adjust_for_construction()
        else:
            self.maintain_progress()
        super().step(dt)

    def detect_obstacles(self) -> List[ControlledVehicle]:
        return [v for v in self.road.vehicles 
                if v is not self and 
                self.lane_distance_to(v) < self.perception_range]

    def emergency_stop_required(self, obstacles: List[ControlledVehicle]) -> bool:
        return any(self.speed - v.speed > 5 and 
                   self.lane_distance_to(v) < 10 
                   for v in obstacles)

    def adjust_for_construction(self):
        self.target_speed = self.base_speed * self.construction_adjustment
        self.action['steering'] = self.steering_control() + 0.15

    def in_construction_zone(self) -> bool:
        return any(lane.width < AbstractLane.DEFAULT_WIDTH 
                   for lane in self.road.network.lanes)

class AggressiveUrbanBehavior(IDMVehicle):
    def __init__(self, road: Road, position: Vector, heading: float = 0, speed: float = 12):
        super().__init__(road, position, heading, speed)
        self.brake_prob = 0.3
        self.base_speed = speed

    def step(self, dt: float):
        if np.random.rand() < self.brake_prob:
            self.execute_sudden_brake()
        super().step(dt)

    def execute_sudden_brake(self):
        self.action['brake'] = np.clip(np.random.uniform(0.6, 1.0), 0, 1)
        self.action['acceleration'] = 0

class ConstructionScenario(AbstractEnv):
    def __init__(self):
        self.traffic_density = 50
        self.slow_lane_speed = 8
        super().__init__()

    def make_road(self) -> Road:
        net = RoadNetwork()
        lane_width = AbstractLane.DEFAULT_WIDTH - 1.5  # Narrow lanes

        # Intersection geometry
        c = [50, 50]  # Center point
        radii = 40
        for heading in [0, np.pi/2, np.pi, 3*np.pi/2]:
            start = [c[0] + radii * np.cos(heading), 
                     c[1] + radii * np.sin(heading)]
            end = [c[0] + radii * np.cos(heading + np.pi),
                   c[1] + radii * np.sin(heading + np.pi)]
            net.add_lane("entry", "int", 
                CircularLane(start, radii, heading, heading + np.pi/2, 
                width=lane_width, line_types=(LineType.CONTINUOUS, LineType.NONE)))
            net.add_lane("int", "exit",
                CircularLane(end, radii, heading + np.pi/2, heading + np.pi,
                width=lane_width, line_types=(LineType.NONE, LineType.CONTINUOUS)))

        # Roadblock obstacle
        roadblock = StaticObstacle(self.road, [25, 40], 8, 3, np.pi/2)
        self.road.objects.append(roadblock)

        return RegulatedRoad(network=net, npc_vehicles=self.generate_traffic())

    def generate_traffic(self) -> List[ControlledVehicle]:
        vehicles = []
        for _ in range(self.traffic_density):
            offset = np.random.uniform(-30, 30, size=2)
            position = [50 + offset[0], 50 + offset[1]]
            heading = np.random.choice([0, np.pi/2, np.pi, 3*np.pi/2])
            vehicles.append(AggressiveUrbanBehavior(
                self.road, position, heading,
                speed=np.random.uniform(10, 15)
            ))
        return vehicles

    def reset(self) -> None:
        super().reset()
        ego_vehicle = ConstrainedNavigation(
            self.road, [0, 0], 0, self.slow_lane_speed
        )
        self.road.vehicles.append(ego_vehicle)
        self.vehicle = ego_vehicle

if __name__ == '__main__':
    scenario = ConstructionScenario()
    scenario.reset()
    for _ in range(600):  # 60 seconds at 10 Hz
        scenario.step()
        if not scenario.vehicle.active:  # Exit condition check
            break