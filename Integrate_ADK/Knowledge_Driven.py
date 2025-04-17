import osmnx as ox
import random
import geopandas as gpd
import json


def read_road_network(osm_file):
    graph = ox.graph_from_xml(osm_file, simplify=True)
    nodes, edges = ox.graph_to_gdfs(graph)
    return nodes, edges


def infer_lane_type(oneway, length):
    if oneway:
        return 'solid' if length >= 15 else 'dashed'
    else:
        return 'dashed'


def extract_lane_types(edges):
    edges['lane_type'] = edges.apply(lambda row: infer_lane_type(row['oneway'], row['length']), axis=1)
    lane_types = edges['lane_type'].unique()
    return lane_types


def generate_traffic_facility():
    return random.choice(['fast', 'slow', 'special', 'normal'])


def simulate_temporary_changes():
    change_type = random.choice(['T', 'P', 'Else'])
    tp = 'no temporary changes'
    if change_type == 'T':
        tp = "Temporary road closure at intersection"
    elif change_type == 'P':
        tp = "Temporary traffic control in place"
    elif change_type == 'Else':
        tp = "no temporary changes"
    else:
        tp = "no temporary changes"
    return tp


class TrafficParticipant:
    def __init__(self, vehicle_type, position, behavior):
        self.vehicle_type = vehicle_type
        self.position = position
        self.behavior = behavior

    def describe(self):
        return f"{self.vehicle_type} at {self.position} is {self.behavior}"


class EgoVehicle:
    def __init__(self, vehicle_type, position, global_behavior):
        self.vehicle_type = vehicle_type
        self.position = position
        self.global_behavior = global_behavior

    def describe(self):
        return f"Ego vehicle {self.vehicle_type} at {self.position}, behavior: {self.global_behavior}"


def analyze_road_network(osm_file):
    nodes, edges = read_road_network(osm_file)
    lane_types = extract_lane_types(edges)
    return nodes, edges, lane_types


def simulate_traffic_participants():
    vehicle_types = ['car', 'truck', 'bus']
    behaviors = ['moving', 'stopping', 'turning', 'changing lanes']
    participants = []

    for _ in range(5):
        vehicle_type = random.choice(vehicle_types)
        position = (random.uniform(-10, 10), random.uniform(-10, 10))
        behavior = random.choice(behaviors)
        participants.append(TrafficParticipant(vehicle_type, position, behavior))

    return participants


def generate_driving_scenario(osm_file):
    nodes, edges, lane_types = analyze_road_network(osm_file)
    lane_type = generate_traffic_facility()
    temporary_change = simulate_temporary_changes()
    participants = simulate_traffic_participants()
    ego_vehicle = EgoVehicle("ego vehicle", (0, 0), "driving straight")

    scenario = {
        "road_network": {
            "nodes": nodes.to_json(),
            "edges": edges.to_json(),
            "lane_types": lane_types.tolist()
        },
        "traffic_facility": lane_type,
        "temporary_change": temporary_change,
        "traffic_participants": [p.describe() for p in participants],
        "ego_vehicle": ego_vehicle.describe()
    }

    return scenario


# === 新增：统一接口函数 ===
def generate_knowledge_translation(osm_file):
    scenario = generate_driving_scenario(osm_file)
    lane_type = scenario['traffic_facility']
    temporary_changes = scenario['temporary_change']
    ego = scenario['ego_vehicle']
    return f"{temporary_changes}; lane type: {lane_type}; {ego}"
