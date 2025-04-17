import random

def generate_traffic_density():
    density = random.choice(['low', 'medium', 'high'])
    count = {'low': 5, 'medium': 20, 'high': 50}[density]
    gap = {'low': 10, 'medium': 5, 'high': 2}[density]
    return density, count, gap

# 增加 relative_pos 字段生成
def generate_complex_driving_behavior():
    behavior = random.choice(['sudden_brake', 'unsafe_lane_change', 'speeding', 'tailgating'])
    relative_pos = random.choice(['front', 'rear', 'left', 'right'])

    if behavior == 'sudden_brake':
        details = {'reaction_time': random.uniform(1, 3)}
    elif behavior == 'unsafe_lane_change':
        details = {
            'direction': random.choice(['left', 'right']),
            'distance': random.uniform(1, 5)
        }
    elif behavior == 'speeding':
        details = {'speed_increase': random.uniform(5, 15)}
    else:  # tailgating
        details = {'distance_to_vehicle': random.uniform(0.5, 1.5)}

    return {
        'behavior': behavior,
        'behavior_details': details,
        'relative_pos': relative_pos
    }

# def generate_road_changes():
#     r = random.choice(['construction_zone', 'road_closure'])
#     if r == 'construction_zone':
#         return r, {
#             'lane_restriction': random.choice(['one_lane', 'two_lane']),
#             'warning_sign': random.choice(['construction_ahead', 'detour'])
#         }
#     if r == 'road_closure':
#         return r, {
#             'closure_location': random.choice(['junction', 'intersection', 'highway'])
#         }

# 改：明确对抗车辆，每个包含 behavior+details+relative_pos
def generate_adv_scenario(num_behaviors=2):
    density, count, gap = generate_traffic_density()

    adversarial_vehicles = []
    for _ in range(num_behaviors):
        adv_info = generate_complex_driving_behavior()
        adversarial_vehicles.append(adv_info)

    # road_change, r_info = generate_road_changes()

    return {
        'traffic_density': density,
        'vehicle_count': count,
        'min_gap': gap,
        'adversarial_vehicles': adversarial_vehicles,
        # 'road_change': road_change,
        'detail_info': {
            # 'road_change_info': r_info
        }
    }

# 输出格式展示增强：清晰列出每个对抗车行为+位置
def generate_adversarial_extension(num_behaviors=2, return_json=False):
    adv = generate_adv_scenario(num_behaviors)
    vehicles = adv['adversarial_vehicles']

    vehicle_descriptions = []
    for i, v in enumerate(vehicles):
        desc = (
            f"Relative to the ego vehicle, the adversarial vehicle [{i+1}] is at the **{v['relative_pos']}** "
            f"and is **{v['behavior']}** ({v['behavior_details']})."
        )
        vehicle_descriptions.append(desc)

    natural_description = (
        f"Traffic density: {adv['traffic_density']} ({adv['vehicle_count']} vehicles);\n"
        + "\n".join(vehicle_descriptions)
    )

    if return_json:
        return {
            "traffic_density": adv['traffic_density'],
            "vehicle_count": adv['vehicle_count'],
            "adversarial_vehicles": vehicles,
            "natural_description": natural_description
        }
    else:
        return natural_description