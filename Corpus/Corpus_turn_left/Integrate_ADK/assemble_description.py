import os
import json
import pandas as pd

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
# print(f"工作根目录设置为: {ROOT_DIR}")

def get_full_path(relative_path):
    """基于工作根目录和相对路径返回文件的绝对路径"""
    return os.path.join(ROOT_DIR, relative_path)

def read_json_file(file_name):
    full_path = get_full_path(file_name)
    with open(full_path, 'r') as file:
        data = json.load(file)
    return data


from SourceDriven.Data_Driven import generate_data_insight, parse_scenario_info
from SourceDriven.Knowledge_Driven import generate_knowledge_translation
from SourceDriven.Adv_Driven import generate_adversarial_extension

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


def assemble_description_with_snippet(scenario_id, json_path, osm_path, folder_path, json_name):
    description = {}

    dataset_name = os.path.basename(folder_path)  # 提取文件夹名称作为数据集名称

    description["scenario_type"] = scenario_id
    description["data_driven_insight"] = generate_data_insight(json_path)
    description["knowledge_driven_translation"] = generate_knowledge_translation(osm_path)

    adv_info = generate_adversarial_extension(return_json=True)
    description["adversarial_extension"] = adv_info["natural_description"]
    description["adversarial_vehicles_info"] = json.dumps(adv_info["adversarial_vehicles"], ensure_ascii=False)

    dataset_info = f"The description is based on the real dataset '{dataset_name}', and the file being processed is '{json_name}'."

    # 原有的自然语言描述
    description["final_natural_language"] = (
        f"In scenario '{description['scenario_type']}', the ego vehicle behavior shows: {description['data_driven_insight']}. "
        f"According to road and rule-based knowledge, we get: {description['knowledge_driven_translation']}. "
        f"In addition, adversarial conditions are introduced: {description['adversarial_extension']}."
    )

    # 在 final_natural_language 中添加dataset_info
    description["final_natural_language"] += " " + dataset_info

    description["trajectory_snippets"] = extract_trajectory_snippets(json_path)
    return description

# def assemble_description_with_snippet(scenario_id, json_path, osm_path):
#     description = {}
#
#     description["scenario_type"] = scenario_id
#     description["data_driven_insight"] = generate_data_insight(json_path)
#     description["knowledge_driven_translation"] = generate_knowledge_translation(osm_path)
#     description["adversarial_extension"] = generate_adversarial_extension()
#
#     description["final_natural_language"] = (
#         f"In scenario '{description['scenario_type']}', the ego vehicle behavior shows: {description['data_driven_insight']}. "
#         f"According to road and rule-based knowledge, we get: {description['knowledge_driven_translation']}. "
#         f"In addition, adversarial conditions are introduced: {description['adversarial_extension']}."
#     )
#
#     description["trajectory_snippets"] = extract_trajectory_snippets(json_path)
#     return description

# def assemble_description_with_snippet(scenario_id, json_path, osm_path):
#     description = {}
#
#     description["scenario_type"] = scenario_id
#     description["data_driven_insight"] = generate_data_insight(json_path)
#     description["knowledge_driven_translation"] = generate_knowledge_translation(osm_path)
#
#     adv_info = generate_adversarial_extension(return_json=True)
#     description["adversarial_extension"] = adv_info["natural_description"]
#     description["adversarial_vehicles_info"] = json.dumps(adv_info["adversarial_vehicles"], ensure_ascii=False)
#
#     # combination
#     description["final_natural_language"] = (
#         f"In scenario '{description['scenario_type']}', the ego vehicle behavior shows: {description['data_driven_insight']}. "
#         f"According to road and rule-based knowledge, we get: {description['knowledge_driven_translation']}. "
#         f"In addition, adversarial conditions are introduced: {description['adversarial_extension']}."
#
#     )
#
#     description["trajectory_snippets"] = extract_trajectory_snippets(json_path)
#     return description


def process_folder(folder_path, osm_path, scenario_id="intersection", output_excel="./will_Description_V3/assembled_descriptions.xlsx"):
    results = []

    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.endswith(".json"):
                full_path = os.path.join(root, file)
                try:
                    desc = assemble_description_with_snippet(scenario_id, full_path, osm_path, folder_path, file)
                    desc["file_name"] = file
                    desc["trajectory_snippets"] = json.dumps(desc["trajectory_snippets"], ensure_ascii=False)
                    results.append(desc)
                    print(f"Processed: {file}")
                except Exception as e:
                    print(f"Error processing {file}: {e}")

    df = pd.DataFrame(results)
    df.to_excel(output_excel, index=False)
    print(f"\n All done! Output saved to: {output_excel}")

if __name__ == "__main__":
    folder_path = get_full_path("Data/turn_left")
    osm_path = get_full_path("map/DR_USA_Intersection_MA.osm")
    output_excel = get_full_path("Corpus/Description/assembled_descriptions.xlsx")

    process_folder(
        folder_path= folder_path,
        osm_path=osm_path,
        output_excel= output_excel
    )