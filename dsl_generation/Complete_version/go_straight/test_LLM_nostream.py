"""
pip install langchain langchain-community openai faiss-cpu
pip install langchain-huggingface
pip install -U sentence-transformers

pip install scikit-learn
"""
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'  # 使用镜像源
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
api_path = os.path.join(ROOT_DIR, 'API_Keys.txt')
with open(api_path, 'r') as file:
    api_key = file.read().strip()
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
import pandas as pd
from collections import Counter
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import time
import re
from collections import defaultdict
from collections import defaultdict, Counter
import time
from difflib import SequenceMatcher
from json_find import find_similar_descriptions

from dsl_to_env_generator_test import convert_dsl_to_highway_env_code


# === 读取数据 ===
xlsx_path = os.path.join(ROOT_DIR, 'Corpus/Description/final_snippet_export.xlsx')
sheets = pd.read_excel(xlsx_path, sheet_name=None)

rag_docs = []
for i in range(len(sheets["chatstyle.description"])):
    try:
        description = sheets["chatstyle.description"].iloc[i, 0]
        geometry = sheets["geometry.snippet"].iloc[i, 0]
        spawn = sheets["spawn.snippet"].iloc[i, 0]
        behavior = sheets["behavior.snippet"].iloc[i, 0]

        text_block = f"""[Scene {i}]
{description}

[Scenic Geometry]
{geometry}

[Scenic Spawn]
{spawn}

[Scenic Behavior]
{behavior}
"""
        rag_docs.append(text_block)
    except Exception as e:
        print(f"[{i}] 跳过异常数据: {e}")

# === 文本切分与向量索引 ===
splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
docs = splitter.create_documents(rag_docs)

embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"  # 或改用本地路径 "./models/all-MiniLM-L6-v2"
)

vectorstore = FAISS.from_documents(docs, embedding_model)
retriever = vectorstore.as_retriever()

# === 严格嵌入 CoT + Semantic Check Prompt ===
cot_prompt = PromptTemplate.from_template("""
You are an expert in autonomous driving testing and simulation.

Below is an example of an input testing scenario text and the corresponding scenario representation:

CONTEXT BLOCK:
{context}

LLM Input: {question}

LLM Output: {{Scenario Representation}}

Chain of Thought:

Let’s think step by step:

{{“left turn” means the vehicle approaches the intersection, so we choose the common “intersection” as “Road Topology” and the “global behavior” is “turn left”; 
“Unprotected” means there is no specific traffic signal（such as a green arrow）to protect the turn, so we do not choose “traffic light” as “traffic sign”; 
“Unprotected left turn” requires yielding to oncoming traffic, so we choose “yield” as “longitudinal oracle”}} 

---

Syntax Alignment Checking:

(1) Think again if the elements in the generated output scenario representation are consistent with the input testing scenario text. If not, correct the inconsistencies and output the revised scenario representation.

(2) Check again the elements in the generated output scenario representation are from the Scenario Repository. {{Dictionary of Hierarchical Scenario Repository}}. If you cannot find a close element in the Scenario Repository consistent with the input testing scenario text, keep your output as the answer.

(3) Correct the output to dictionary data structure format.
""")

# === Step 4: 使用 prompt 构建 RAG 问答链 ===
# ChatGPT
# llm = ChatOpenAI(
#     openai_api_key="your_openai_key",  # 替换为你的 OpenAI API 密钥
#     model="gpt-4",  # 或 "gpt-3.5-turbo"
#     temperature=0.7
# )

# DeepSeek
llm = ChatOpenAI(
    openai_api_key=api_key,
    openai_api_base="https://api.deepseek.com/v1",
    model="deepseek-chat",  # "deepseek-chat" 或 "deepseek-reasoner"
    temperature=0.6
)


# rag_chain = RetrievalQA.from_chain_type(
#     llm=llm,
#     retriever=retriever,
#     chain_type_kwargs={"prompt": cot_prompt},
#     return_source_documents=True
# )

# === 修正后的 RAG 链配置 ===
rag_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    chain_type="stuff",  #意味着检索到的多个文档会“拼接在一起”（stuffed）作为输入传给语言模型（适用于上下文较短的情况）
    chain_type_kwargs={
        "prompt": cot_prompt,
        "document_variable_name": "context"  # 显式绑定文档变量
    },
    return_source_documents=True
)

# === 问题 ===
description_text = '''In this intersection scenario, the ego vehicle maintains a straight trajectory through the intersection while adhering to traffic rules. The vehicle operates in the fast lane without any temporary control modifications. The simulation introduces high traffic density conditions with approximately 50 vehicles in the vicinity. Two adversarial vehicles are specifically positioned: 
1) Adversary 1 performs dangerous tailgating behavior at an unsafe following distance of 0.56 meters to the ego's right rear
2) Adversary 2 executes an unsafe left lane change maneuver from the right side at a distance of 4.02 meters.  '''
# 调用函数获取最相似的描述
json_path = os.path.join(ROOT_DIR, "Corpus/Description/description_fileinfo_pairs.json")
similar_descriptions = find_similar_descriptions(description_text, json_path, top_k=3)

# 输出结果
for item in similar_descriptions:
    print(f"DataSet: {item['DataSet']}")
    print(f"Description: {item['Description']}")
    print(f"Similarity: {item['Similarity']:.4f}\n")

query = """
Assuming you are an expert in autonomous driving testing, your task is to generate scenario representation from the following given testing scenario description text based on the Domain-Specific Language.

The Hierarchical Scenario Repository provides a dictionary of scenario components corresponding to each element that you can choose from. When creating scenario representation, please first consider the following elements for each subcomponent. If there is no element that can describe a similar meaning, then create a new element yourself.

{Dictionary of Hierarchical Scenario Repository}

Few-Shot

Below are two examples of the input testing scenario texts and the corresponding scenario representations:

LLM Input1:{{In the intersection scenario, the ego vehicle maintains a straight trajectory while traversing the intersection under temporary traffic control measures, with the lane designated as a slow-speed zone. The adversarial conditions feature high traffic density (50 vehicles) with two notable adversarial vehicles:  
1) Adversarial Vehicle [1] positioned to the left of the ego vehicle (lateral distance: 4.44m) executes an unsafe lane change maneuver to the left.  
2) Adversarial Vehicle [2] ahead of the ego vehicle performs a sudden braking action with a reaction time of 1.68s.  
This scenario is reconstructed from the real-world dataset [vehicle_tracks_003_trajectory_set_25.json] of the NGSIM US-101 highway dataset.}}
LLM output1:{{
#------geometry.snippet------#
from NGSIM_env.road.road import Road, RoadNetwork  
from NGSIM_env.road.lanelet import OSMReader  

# Primary scene setup  
road = Road(network=RoadNetwork.from_lanelet2(  
    OSMReader.load("[vehicle_tracks_003_trajectory_set_25.json]")  
))  
ego_initial_pose = (x=152.3, y=45.1, heading=1.57)  # Facing north (π/2 rad)

#------spawn.snippet------#
from NGSIM_env.vehicle.humandriving import HumanLikeVehicle, InterActionVehicle  

# Ego vehicle  
ego = HumanLikeVehicle(  
    road=road,  
    position=ego_initial_pose[:2],  
    heading=ego_initial_pose[2],  
    target_speed=8.0  # m/s (slow lane)  
)  

# Adversarial vehicles  
adv1 = InterActionVehicle(  
    road=road,  
    position=(ego_initial_pose[0]-4.44, ego_initial_pose[1]+10.0),  
    heading=ego_initial_pose[2],  
    behavior_type="aggressive_lane_change"  
)  

adv2 = InterActionVehicle(  
    road=road,  
    position=(ego_initial_pose[0], ego_initial_pose[1]+20.0),  
    heading=ego_initial_pose[2],  
    behavior_type="sudden_brake"  
)

#------behavior.snippet------#
from NGSIM_env.vehicle.behavior import IDMVehicle  

# Ego behavior (rule-based)  
ego.set_behavior(  
    trajectory_type="straight",  
    lane_change_approved=False,  # Comply with temporary control  
    speed_control=IDMVehicle(target_speed=8.0)  
)  

# Adversarial behaviors  
adv1.set_behavior(  
    lane_change_params={"direction": "left", "duration": 2.5},  
    urgency=0.9  # Aggressive maneuver  
)  

adv2.set_behavior(  
    braking_params={"deceleration": 4.0, "reaction_time": 1.68},  # ~0.4g brake  
    trigger_distance=15.0  # Activate at 15m ahead  
)
}}



LLM Input2:{{In this intersection crossing scenario under temporary traffic control, the ego vehicle maintains a straight trajectory through the intersection while complying with traffic rules. The road configuration designates the current lane as a fast lane with low traffic density (approximately 5 vehicles in the vicinity). Two adversarial vehicles create challenging conditions: 
1) Adversary 1 positioned on the right performs a sudden braking maneuver with a reaction time of 1.35 seconds
2) Adversary 2 on the left engages in dangerous tailgating behavior, maintaining an unsafe following distance of 0.75 meters. This scenario is reconstructed from real-world driving data [vehicle_tracks_003_trajectory_set_95.json].}}
LLM output2:{{
#------geometry.snippet------#
from NGSIM_env.road.road import Road, RoadNetwork  
from NGSIM_env.road.lanelet import OSMReader  

# Primary scene setup  
road = Road(network=RoadNetwork.from_lanelet2(  
    OSMReader.load("[vehicle_tracks_003_trajectory_set_95.json]")  
))  
ego_initial_pose = (x=0.0, y=0.0, heading=0.0)  # Facing east at intersection entry

#------spawn.snippet------#
from NGSIM_env.vehicle.humandriving import HumanLikeVehicle, InterActionVehicle  

# Ego vehicle (fast lane configuration)
ego = HumanLikeVehicle(  
    road=road,  
    position=ego_initial_pose[:2],  
    heading=ego_initial_pose[2],  
    target_speed=12.0,  # m/s (fast lane speed)
    lane_type="fast"  
)  

# Adversarial vehicles
adv1 = InterActionVehicle(  
    road=road,  
    position=(ego_initial_pose[0], ego_initial_pose[1]-1.5),  # Right offset
    heading=ego_initial_pose[2],  
    behavior_type="sudden_brake"  
)  

adv2 = InterActionVehicle(  
    road=road,  
    position=(ego_initial_pose[0]-0.75, ego_initial_pose[1]+1.5),  # Left-rear position
    heading=ego_initial_pose[2],  
    behavior_type="tailgating"  
)

#------behavior.snippet------#
from NGSIM_env.vehicle.behavior import IDMVehicle  

# Ego behavior (rule-compliant under traffic control)
ego.set_behavior(  
    trajectory_type="straight",  
    intersection_handling="strict_no_change",  
    speed_control=IDMVehicle(target_speed=12.0),  
    lane_change_approved=False  
)  

# Adversarial behaviors
adv1.set_behavior(  
    braking_params={
        "deceleration": 6.0,  # ~0.6g emergency brake
        "reaction_time": 1.35  
    },  
    trigger_distance=5.0  
)  

adv2.set_behavior(  
    following_params={
        "safety_distance": 0.75, 
        "min_gap": 0.3  # Dangerously close
    },  
    urgency=1.2  # Aggressive tailgating
)
}}

The Hierarchical Scenario Repository provides a dictionary of scenario components corresponding to each element that you can choose from. When creating scenario representation, please first consider the following elements for each subcomponent. If there is no element that can describe a similar meaning, then create a new element yourself.

Based on the above description and examples, convert the following testing scenario text into the corresponding scenario representation:
{In this intersection scenario, the ego vehicle maintains a straight trajectory through the intersection while adhering to traffic rules. The vehicle operates in the fast lane without any temporary control modifications. The simulation introduces high traffic density conditions with approximately 50 vehicles in the vicinity. Two adversarial vehicles are specifically positioned: 
1) Adversary 1 performs dangerous tailgating behavior at an unsafe following distance of 0.56 meters to the ego's right rear
2) Adversary 2 executes an unsafe left lane change maneuver from the right side at a distance of 4.02 meters. }

give me the code please!
"""

# def self_consistent_invoke(question, rag_chain, num_votes=5, delay=0.5):
#     """
#     非流式版本的 Exact Match Voting：多次 invoke 并对结果投票
#     """
#     all_outputs = []
#
#     for i in range(num_votes):
#         try:
#             print(f"\n---- 第 {i+1} 次调用 ----")
#             response = rag_chain.invoke(question)
#             output = response["result"].strip()
#             print(output)
#             all_outputs.append(output)
#             time.sleep(delay)
#         except Exception as e:
#             print(f"[第 {i+1} 次] 出错：{e}")
#
#     # 投票统计
#     print("\n==================== Voting 统计结果 ====================\n")
#     vote_result, count = Counter(all_outputs).most_common(1)[0]
#     print(f"出现频率最高的输出（共出现 {count} 次）:\n")
#     print(vote_result)
#
#     return vote_result, all_outputs
def self_consistent_rag_chain_embedding(question, rag_chain, num_votes=5, delay=0.5):
    """
    非流式版本 + 嵌入相似度 Voting 的 Self-Consistency 实现。
    输出每个采样答案的平均相似度，并选择最“中心”的答案作为最终结果。
    """
    all_answers = []

    for i in range(num_votes):
        try:
            response_text = ""
            print(f"\n---- 第 {i+1} 次采样（Embedding 模式） ----")
            response = rag_chain.invoke({"query": question})
            response_text = response["result"]
            print(response_text)

            all_answers.append(response_text.strip())
            time.sleep(delay)
        except Exception as e:
            print(f"[采样 {i+1}] 调用失败: {e}")

    if not all_answers:
        print("无有效输出，结束。")
        return None, []

    print("\n\n==================== 开始计算嵌入相似度矩阵 ====================\n")
    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(all_answers)
    sim_matrix = cosine_similarity(embeddings)

    avg_sim = sim_matrix.mean(axis=1)
    best_idx = int(np.argmax(avg_sim))
    final_answer = all_answers[best_idx]

    print("\n==================== 所有采样结果及相似度统计 ====================\n")
    for i, (ans, sim) in enumerate(zip(all_answers, avg_sim)):
        print(f"\n--- Answer {i+1} ---")
        print(f"[平均相似度] -> {sim:.4f}")
        print(ans[:500] + ("..." if len(ans) > 500 else ""))

    print(f"\n==================== 最终 Voting 结果（第 {best_idx+1} 个输出） ====================\n")
    print(final_answer)

    return final_answer, all_answers, avg_sim.tolist()

def fuzzy_equal(a, b, threshold=0.85):
    """使用模糊比对判断两个字符串是否相近"""
    return SequenceMatcher(None, a or "", b or "").ratio() > threshold

def structured_voting_pipeline_clustered(question, rag_chain, num_votes=5, delay=0.5, sim_threshold=0.6):
    """
    DSL结构字段聚类（使用模糊匹配） + 最大簇字段级投票 + 返回最接近的原始 DSL。
    """
    all_answers = []
    for i in range(num_votes):
        print(f"\n---- 第 {i + 1} 次采样（Structured_clustered 模式） ----")
        try:
            response = rag_chain.invoke({"query": question})
            response_text = response["result"]
            print(response_text)
            all_answers.append(response_text.strip())
            time.sleep(delay)
        except Exception as e:
            print(f"[采样 {i + 1}] 调用失败: {e}")

    parsed_list = [parse_dsl_fields(ans) for ans in all_answers]

    # === 使用模糊比对的结构字段相似度函数 ===
    def field_similarity(dict1, dict2):
        keys = set(dict1.keys()).union(set(dict2.keys()))
        same = 0
        for k in keys:
            v1 = dict1.get(k)
            v2 = dict2.get(k)
            if fuzzy_equal(v1, v2):
                same += 1
        return same / len(keys) if keys else 0

    # === 聚类 ===
    clusters = []
    for cur in parsed_list:
        added = False
        for cluster in clusters:
            if field_similarity(cur, cluster[0]) >= sim_threshold:
                cluster.append(cur)
                added = True
                break
        if not added:
            clusters.append([cur])

    print(f"\n=== 找到 {len(clusters)} 个结构聚类 ===")
    for i, cl in enumerate(clusters):
        print(f" - 聚类 {i+1} 包含 {len(cl)} 个样本")

    largest_cluster = max(clusters, key=len)
    print(f"\n 选中聚类（大小={len(largest_cluster)}）作为最终投票候选")

    field_values = defaultdict(list)
    for d in largest_cluster:
        for k, v in d.items():
            field_values[k].append(v)

    final_fields = {}
    for key, vals in field_values.items():
        most_common_val, freq = Counter(vals).most_common(1)[0]
        final_fields[key] = most_common_val

    # === 从原始 DSL 中找字段最接近的原始版本 ===
    best_idx = -1
    best_sim = -1
    for i, ans in enumerate(all_answers):
        parsed = parse_dsl_fields(ans)
        sim = field_similarity(parsed, final_fields)
        if sim > best_sim:
            best_sim = sim
            best_idx = i

    final_dsl = all_answers[best_idx]
    print(f"\n--------- 选择原始第 {best_idx + 1} 个 DSL 作为最终结果（字段最接近投票结果）---------")
    print("\n==================== 最终 DSL 输出 ====================\n")
    print(final_dsl)

    return final_dsl

# def structured_voting_pipeline_clustered(question, rag_chain, num_votes=5, delay=0.5, sim_threshold=0.75):
#     """
#     DSL结构字段聚类 + 最大簇字段级投票 + 重建可执行DSL。
#     """
#     from collections import defaultdict
#     import time
#
#     # === Step 1: 采样多次输出 ===
#     all_answers = []
#     for i in range(num_votes):
#         print(f"\n---- 第 {i + 1} 次采样（Structured_clustered 模式） ----")
#         try:
#             response = rag_chain.invoke({"query": question})
#             response_text = response["result"]
#             print(response_text)
#             all_answers.append(response_text.strip())
#             time.sleep(delay)
#         except Exception as e:
#             print(f"[采样 {i + 1}] 调用失败: {e}")
#
#     # === Step 2: 将 DSL 输出解析为字段结构 ===
#     parsed_list = [parse_dsl_fields(ans) for ans in all_answers]
#
#     # === Step 3: 自定义结构字段相似度函数 ===
#     def field_similarity(dict1, dict2):
#         keys = set(dict1.keys()).union(set(dict2.keys()))
#         same = sum(1 for k in keys if dict1.get(k) == dict2.get(k))
#         return same / len(keys) if keys else 0
#
#     # === Step 4: 对结构字段进行聚类 ===
#     clusters = []
#     for i, cur in enumerate(parsed_list):
#         added = False
#         for cluster in clusters:
#             if field_similarity(cur, cluster[0]) >= sim_threshold:
#                 cluster.append(cur)
#                 added = True
#                 break
#         if not added:
#             clusters.append([cur])
#
#     # === Step 5: 输出聚类统计 ===
#     print(f"\n=== 找到 {len(clusters)} 个结构聚类 ===")
#     for i, cl in enumerate(clusters):
#         print(f" - 聚类 {i+1} 包含 {len(cl)} 个样本")
#
#     # === Step 6: 从最大聚类中做字段 Voting ===
#     largest_cluster = max(clusters, key=len)
#     print(f"\n 选中聚类（大小={len(largest_cluster)}）作为最终投票候选")
#
#     field_values = defaultdict(list)
#     for d in largest_cluster:
#         for k, v in d.items():
#             field_values[k].append(v)
#
#     final_fields = {}
#     for key, vals in field_values.items():
#         most_common_val, freq = Counter(vals).most_common(1)[0]
#         final_fields[key] = most_common_val
#
#     # === Step 7: 从原始答案中选“结构最相近”的那个 DSL ===   : 字段 Voting 后，直接选择“原始 DSL”中最接近字段投票结果的那一个
#     def field_similarity(dict1, dict2):
#         keys = set(dict1.keys()).union(set(dict2.keys()))
#         same = sum(1 for k in keys if dict1.get(k) == dict2.get(k))
#         return same / len(keys) if keys else 0
#
#     best_idx = -1
#     best_sim = -1
#     for i, ans in enumerate(all_answers):
#         parsed = parse_dsl_fields(ans)
#         sim = field_similarity(parsed, final_fields)
#         if sim > best_sim:
#             best_sim = sim
#             best_idx = i
#
#     final_dsl = all_answers[best_idx]
#     print(f"\n--------- 选择原始第 {best_idx + 1} 个 DSL 作为最终结果（字段最接近投票结果）---------")
#     print("\n==================== 最终 DSL 输出 ====================\n")
#     print(final_dsl)
#
#     return final_dsl

def parse_dsl_fields(dsl_text):
    fields = {}
    match_behavior = re.search(r'ego\s*=\s*new\s+Car\s+with\s+behavior\s+(\w+)', dsl_text)
    if match_behavior:
        fields['ego_behavior'] = match_behavior.group(1)
    match_at = re.search(r'ego\s*=.*?at\s+(road\[[^\n]+)', dsl_text)
    if match_at:
        fields['ego_position'] = match_at.group(1)
    match_intersection = re.search(r'IntersectionRegion\s*=\s*.+?\n', dsl_text)
    if match_intersection:
        fields['intersection'] = match_intersection.group(0).strip()
    match_param = re.findall(r'param\s+(\w+)\s*=\s*([^\n]+)', dsl_text)
    for name, value in match_param:
        fields[f'param_{name}'] = value.strip()
    match_behavior_classes = re.findall(r'class\s+(\w+)\(Behavior\):', dsl_text)
    for i, cls in enumerate(match_behavior_classes):
        fields[f'class_behavior_{i}'] = cls
    return fields

def rebuild_dsl_from_fields(fields):
    lines = []

    if "intersection" in fields:
        lines.append(fields["intersection"])
        lines.append("")

    behavior = fields.get("ego_behavior", "DefaultBehavior")
    position = fields.get("ego_position", "road[0].lanes[0].centerline[0]")
    lines.append(f"ego = new Car with behavior {behavior},")
    lines.append(f"    at {position},")
    lines.append(f"    facing road[0].direction")
    lines.append("")

    for key, val in fields.items():
        if key.startswith("param_"):
            name = key[len("param_"):]
            lines.append(f"param {name} = {val}")
    lines.append("")

    for key, cls in fields.items():
        if key.startswith("class_behavior_"):
            lines.append(f"class {cls}(Behavior):")
            lines.append("    def step(self):")
            lines.append("        pass\n")

    return "\n".join(lines)


# === 使用 Self-Consistency Voting ===
print("\n==================== 使用 Self-Consistency Voting ====================\n")

voting_mode = "embedding"  # 可选 "embedding" 或 "structured"

if voting_mode == "embedding":
    final_answer, all_candidates, all_similarities = self_consistent_rag_chain_embedding(
        query, rag_chain, num_votes=5, delay=1
    )
elif voting_mode == "structured_cluster":
    final_answer = structured_voting_pipeline_clustered(query, rag_chain, num_votes=10, delay=1)
else:
    raise ValueError("未知 voting 模式")

print("\n==================== 最终输出 ====================\n")
print(final_answer)


# print("\n\n==================== 输入的问题 ====================\n", query)
# print("\n==================== 最终答案（Embedding 相似度 Voting） ====================\n")
# print(final_answer)

# === 可选：打印所有候选答案 ===
# for idx, ans in enumerate(all_candidates):
#     print(f"\n--- Candidate {idx+1} ---\n{ans}")


# 根据 final_answer 转换为 highway_env code
converted_code = convert_dsl_to_highway_env_code(final_answer)
# convert_dsl_to_highway_env_code(final_answer, api_key="sk-xxxxx")

# 打印或保存结果
print("\n==== 转换后的 Highway-Env Python 代码 ====\n")
print(converted_code)
