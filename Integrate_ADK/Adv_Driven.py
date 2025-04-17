import random


def generate_traffic_density():
    density = random.choice(['low', 'medium', 'high'])
    count = {'low': 5, 'medium': 20, 'high': 50}[density]
    gap = {'low': 10, 'medium': 5, 'high': 2}[density]
    return density, count, gap


def generate_weather_conditions():
    weather = random.choice(['clear', 'rainy', 'foggy', 'snowy', 'stormy'])
    condition = {
        'clear': 'dry', 'rainy': 'wet', 'foggy': 'low visibility',
        'snowy': 'icy', 'stormy': 'hazardous conditions'
    }[weather]
    return weather, condition


def generate_complex_driving_behavior():
    b = random.choice(['sudden_brake', 'unsafe_lane_change', 'speeding', 'tailgating'])
    if b == 'sudden_brake':
        return b, {'reaction_time': random.uniform(1, 3)}
    if b == 'unsafe_lane_change':
        return b, {'direction': random.choice(['left', 'right']), 'distance': random.uniform(1, 5)}
    if b == 'speeding':
        return b, {'speed_increase': random.uniform(5, 15)}
    return b, {'distance_to_vehicle': random.uniform(0.5, 1.5)}


def generate_emergency_scenario():
    e = random.choice(['roadblock', 'sudden_stop', 'obstacle_on_road'])
    if e == 'roadblock':
        return e, {'position': random.choice(['intersection', 'exit', 'lane'])}
    if e == 'sudden_stop':
        return e, {'stop_time': random.uniform(0.5, 1.5)}
    return e, {'obstacle_type': random.choice(['vehicle', 'fallen_tree', 'debris']),
               'obstacle_position': random.uniform(5, 15)}


def generate_road_changes():
    r = random.choice(['construction_zone', 'road_closure', 'traffic_control'])
    if r == 'construction_zone':
        return r, {'lane_restriction': random.choice(['one_lane', 'two_lane']),
                   'warning_sign': random.choice(['construction_ahead', 'detour'])}
    if r == 'road_closure':
        return r, {'closure_location': random.choice(['junction', 'intersection', 'highway'])}
    return r, {'control_type': random.choice(['traffic_light', 'yield_sign', 'stop_sign'])}


def generate_perception_challenges():
    p = random.choice(['low_visibility', 'obstacle_detection'])
    if p == 'low_visibility':
        return p, {'visibility': random.choice(['fog', 'rain', 'night'])}
    return p, {'obstacle_type': random.choice(['vehicle', 'pedestrian', 'large_object'])}


# === 修改：添加参数 num_behaviors，允许生成多个对抗行为 ===
def generate_adv_scenario(num_behaviors=2):
    density, count, gap = generate_traffic_density()
    weather, condition = generate_weather_conditions()

    # 生成多个对抗行为
    behaviors = []
    behavior_infos = []
    for _ in range(num_behaviors):
        behavior, b_info = generate_complex_driving_behavior()
        behaviors.append(behavior)
        behavior_infos.append(b_info)

    # emergency, e_info = generate_emergency_scenario()
    # road_change, r_info = generate_road_changes()
    perception, p_info = generate_perception_challenges()

    return {
        'traffic_density': density, 'vehicle_count': count, 'min_gap': gap,
        # 'weather': weather, 'road_condition': condition,
        'driving_behaviors': behaviors,  # 修改字段名
        # 'emergency_scenario': emergency,
        # 'road_change': road_change,
        # 'perception_challenge': perception,
        'detail_info': {
            'behavior_info_list': behavior_infos,  # 对应多个行为信息
            # 'emergency_info': e_info,
            # 'road_change_info': r_info,
            # 'perception_info': p_info
        }
    }


# === 修改：添加 num_behaviors 参数，并改为展示多个行为 ===
def generate_adversarial_extension(num_behaviors=2): # 可以改成1，2，3，4对应这个场景中目前生成多少的对抗性行为
    adv = generate_adv_scenario(num_behaviors)
    behavior_descriptions = "; ".join(
        f"{b} ({info})" for b, info in zip(adv['driving_behaviors'], adv['detail_info']['behavior_info_list'])
    )
    return (
        f"Traffic density: {adv['traffic_density']} ({adv['vehicle_count']} vehicles); "
        # f"Weather: {adv['weather']} ({adv['road_condition']}); "
        f"Driving behaviors: {behavior_descriptions}; "
        # f"Emergency: {adv['emergency_scenario']} ({adv['detail_info']['emergency_info']}); "
        # f"Road change: {adv['road_change']} ({adv['detail_info']['road_change_info']}); "
        # f"Perception challenge: {adv['perception_challenge']} ({adv['detail_info']['perception_info']})"
    )

# def generate_adv_scenario():
#     density, count, gap = generate_traffic_density()
#     weather, condition = generate_weather_conditions()
#     behavior, b_info = generate_complex_driving_behavior()
#     emergency, e_info = generate_emergency_scenario()
#     road_change, r_info = generate_road_changes()
#     perception, p_info = generate_perception_challenges()
#
#     return {
#         'traffic_density': density, 'vehicle_count': count, 'min_gap': gap,
#         'weather': weather, 'road_condition': condition,
#         'driving_behavior': behavior, 'emergency_scenario': emergency,
#         'road_change': road_change, 'perception_challenge': perception,
#         'detail_info': {'behavior_info': b_info, 'emergency_info': e_info,
#                         'road_change_info': r_info, 'perception_info': p_info}
#     }
#
#
# # === 新增：统一接口函数 ===
# def generate_adversarial_extension():
#     adv = generate_adv_scenario()
#     return (
#         f"Traffic density: {adv['traffic_density']} ({adv['vehicle_count']} vehicles); "
#         f"Weather: {adv['weather']} ({adv['road_condition']}); "
#         f"Driving behavior: {adv['driving_behavior']}; "
#         f"Emergency: {adv['emergency_scenario']}; "
#         f"Road change: {adv['road_change']}; "
#         f"Perception challenge: {adv['perception_challenge']}"
#     )
