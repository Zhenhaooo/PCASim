import os
import json
import pandas as pd

from Data_Driven import generate_data_insight, parse_scenario_info
from Knowledge_Driven import generate_knowledge_translation
from Adv_Driven import generate_adversarial_extension

def extract_trajectory_snippets(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)

    ego_data = data.get("ego", {})
    scenarios = parse_scenario_info(ego_data)

    trajectory_snippets = []
    for s in scenarios:
        snippet = {
            "behavior": s["behavior"],
            "ego_vehicle_id": s.get("ego_vehicle_id"),
            "background_vehicle_id": s.get("background_vehicle_id"),
            "start_time": s.get("start_time"),
            "end_time": s.get("end_time")
        }
        trajectory_snippets.append(snippet)

    return trajectory_snippets

def assemble_description_with_snippet(scenario_id, json_path, osm_path):
    description = {}

    description["scenario_type"] = scenario_id
    description["data_driven_insight"] = generate_data_insight(json_path)
    description["knowledge_driven_translation"] = generate_knowledge_translation(osm_path)
    description["adversarial_extension"] = generate_adversarial_extension()

    description["final_natural_language"] = (
        f"In scenario '{description['scenario_type']}', the ego vehicle behavior shows: {description['data_driven_insight']}. "
        f"According to road and rule-based knowledge, we get: {description['knowledge_driven_translation']}. "
        f"In addition, adversarial conditions are introduced: {description['adversarial_extension']}."
    )

    description["trajectory_snippets"] = extract_trajectory_snippets(json_path)
    return description

def process_folder(folder_path, osm_path, scenario_id="intersection", output_excel="./will_Description_V2/assembled_descriptions.xlsx"):
    results = []

    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.endswith(".json"):
                full_path = os.path.join(root, file)
                try:
                    desc = assemble_description_with_snippet(scenario_id, full_path, osm_path)
                    desc["file_name"] = file
                    # 转为字符串保存 snippets
                    desc["trajectory_snippets"] = json.dumps(desc["trajectory_snippets"], ensure_ascii=False)
                    results.append(desc)
                    print(f"Processed: {file}")
                except Exception as e:
                    print(f"Error processing {file}: {e}")

    df = pd.DataFrame(results)
    df.to_excel(output_excel, index=False)
    print(f"\n All done! Output saved to: {output_excel}")

# 示例运行
if __name__ == "__main__":
    process_folder(
        folder_path="./brake",
        osm_path="./map/DR_USA_Intersection_MA.osm",
        scenario_id="intersection"
    )


"""
下面是之前尝试处理单个的代码
"""
# from Data_Driven import generate_data_insight
# from Knowledge_Driven import generate_knowledge_translation
# from Adv_Driven import generate_adversarial_extension
#
# import json
#
# def assemble_description(scenario_id, json_path, osm_path):
#     """
#     整合三个模块的输出为一个统一的语料结构
#     参数:
#         scenario_id: 场景标识名称（如 'intersection_stop_sign'）
#         json_path: ego 行为数据 JSON 文件路径
#         osm_path: 地图文件路径（.osm）
#
#     返回:
#         一个包含五个字段的字典
#     """
#     description = {}
#
#     # 场景类型
#     description["scenario_type"] = scenario_id
#
#     # 三个维度生成的语义片段
#     description["data_driven_insight"] = generate_data_insight(json_path)
#     description["knowledge_driven_translation"] = generate_knowledge_translation(osm_path)
#     description["adversarial_extension"] = generate_adversarial_extension()
#
#     # 统一自然语言摘要
#     description["final_natural_language"] = (
#         f"In scenario '{scenario_id}', the ego vehicle behavior shows: {description['data_driven_insight']}. "
#         f"According to road and rule-based knowledge, we get: {description['knowledge_driven_translation']}. "
#         f"In addition, adversarial conditions are introduced: {description['adversarial_extension']}."
#     )
#
#     return description
#
#
# # === 示例运行入口 ===
# if __name__ == "__main__":
#     # 替换为你自己的数据路径
#     scenario_id = "intersection"
#     json_path = "./brake/go_straight/vehicle_tracks_000_trajectory_set_23.json"
#     osm_path = "./map/DR_USA_Intersection_MA.osm"
#
#     desc = assemble_description(scenario_id, json_path, osm_path)
#
#     # 打印输出为 JSON 格式
#     print(json.dumps(desc, indent=4))
