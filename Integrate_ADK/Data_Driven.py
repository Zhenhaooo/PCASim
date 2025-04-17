import json

def parse_scenario_info(data):
    """
    解析并提取 scenario_info 中的对抗性行为数据。
    支持行为类型包括：brake, following, go_straight, lane_change, turn_left, turn_right, turn_round
    """
    adversarial_behaviors = []
    scenario_info = data.get("scenario_info", {})
    travel_characteristics = scenario_info.get("travel_characteristics", {})

    # 1. Brake
    if travel_characteristics.get("brake", False):
        for b in scenario_info.get("brake_info", []):
            adversarial_behaviors.append({
                'behavior': 'brake',
                'ego_vehicle_id': "ego_vehicle",
                'background_vehicle_id': b.get('brake_vehicle_id'),
                'start_time': b.get('brake_start'),
                'end_time': b.get('brake_end'),
                'duration': b.get('brake_duration'),
                'brake_direction': b.get('brake_direction')
            })

    # 2. Following
    if travel_characteristics.get("following", False):
        for f in scenario_info.get("following_info", []):
            adversarial_behaviors.append({
                'behavior': 'following',
                'ego_vehicle_id': "ego_vehicle",
                'background_vehicle_id': f.get('following_vehicle_id'),
                'start_time': f.get('following_start'),
                'end_time': f.get('following_end'),
                'duration': f.get('following_duration'),
                'following_direction': f.get('following_direction')
            })

    # 3. Go straight / 5. Turn left / 6. Turn right
    for direction in ['go_straight', 'turn_left', 'turn_right']:
        if travel_characteristics.get(direction, False):
            intersection_info = scenario_info.get("intersection", {})
            for iid, idata in intersection_info.items():
                adversarial_behaviors.append({
                    'behavior': direction,
                    'ego_vehicle_id': "ego_vehicle",
                    'background_vehicle_id': iid,
                    'start_time': idata.get('index_cp', [None])[0],
                    'end_time': idata.get('index_cp', [None])[-1],
                    'intersection_info': idata.get('av_heading'),
                    'PET': idata.get('PET')
                })

    # 4. Lane change
    if travel_characteristics.get("lane_change", False):
        lane_change = scenario_info.get("laneChanging_info", {})
        adversarial_behaviors.append({
            'behavior': 'lane_change',
            'ego_vehicle_id': "ego",
            'start_time': lane_change.get('lane_change_timestamps'),
            'lane_change_towards': lane_change.get('towards'),
            'lane_change_start_index': lane_change.get('start_lane_change_index'),
            'lane_change_end_index': lane_change.get('end_lane_change_index'),
            'last_lane_front_v': lane_change.get('last_lane_front_v'),
            'last_lane_rear_v': lane_change.get('last_lane_rear_v'),
            'current_lane_front_v': lane_change.get('current_lane_front_v'),
            'current_lane_rear_v': lane_change.get('current_lane_rear_v')
        })

    # 7. Turn round
    if travel_characteristics.get("turn_round", False):
        adversarial_behaviors.append({
            'behavior': 'turn_round',
            'ego_vehicle_id': "ego_vehicle",
            'start_time': None,
            'end_time': None
        })

    return adversarial_behaviors


# === 新增：统一接口函数 ===
def generate_data_insight(file_path):
    """
    给定 ego 数据 JSON 路径，返回行为模式的自然语言摘要
    """
    with open(file_path, 'r') as f:
        data = json.load(f)

    ego_data = data.get('ego', {})
    scenarios = parse_scenario_info(ego_data)
    insights = []

    for s in scenarios:
        behavior = s['behavior']
        if behavior == 'brake':
            insights.append(f"ego vehicle braked in direction {s.get('brake_direction')}")
        elif behavior == 'following':
            insights.append("ego vehicle followed another vehicle")
        elif behavior == 'go_straight':
            insights.append("ego vehicle went straight through intersection")
        elif behavior == 'turn_left':
            insights.append("ego vehicle turned left at an intersection")
        elif behavior == 'turn_right':
            insights.append("ego vehicle turned right at an intersection")
        elif behavior == 'lane_change':
            insights.append(f"ego vehicle changed lane towards {s.get('lane_change_towards')}")
        elif behavior == 'turn_round':
            insights.append("ego vehicle performed a U-turn")

    return "; ".join(insights) if insights else "no significant behavior observed"
