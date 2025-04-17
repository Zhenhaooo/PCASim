"""
pip install langchain langchain-community openai faiss-cpu
pip install langchain-huggingface
pip install -U sentence-transformers

pip install scikit-learn
"""
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'  # 使用镜像源

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

from dsl_to_env_generator import convert_dsl_to_highway_env_code


# === Step 1: 读取数据 ===
xlsx_path = "final_snippet_export.xlsx"
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

# === Step 2: 文本切分与向量索引 ===
splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
docs = splitter.create_documents(rag_docs)

# 使用镜像源或本地路径加载模型
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"  # 或改用本地路径 "./models/all-MiniLM-L6-v2"
)

vectorstore = FAISS.from_documents(docs, embedding_model)
retriever = vectorstore.as_retriever()

# === Step 3: 严格嵌入 CoT + Semantic Check Prompt ===
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
    openai_api_key="sk-4d8332cd221a45d9b505e5f93d7122b2",
    openai_api_base="https://api.deepseek.com/v1",
    model="deepseek-reasoner",  # "deepseek-chat" 或 "deepseek-reasoner"
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

# === Step 5: 示例问题 ===
# query = "Assuming you are an expert in autonomous driving testing, your task is to generate scenario representation from the following given testing scenario description text based on the Domain-Specific Language."
query = """
Assuming you are an expert in autonomous driving testing, your task is to generate scenario representation from the following given testing scenario description text based on the Domain-Specific Language.

The Hierarchical Scenario Repository provides a dictionary of scenario components corresponding to each element that you can choose from. When creating scenario representation, please first consider the following elements for each subcomponent. If there is no element that can describe a similar meaning, then create a new element yourself.

{Dictionary of Hierarchical Scenario Repository}

Few-Shot

Below are two examples of the input testing scenario texts and the corresponding scenario representations:



LLM Input1:{{In this intersection scenario, the ego vehicle maintains a straight trajectory while executing braking maneuvers. The road environment features a slow-speed lane without temporary traffic control modifications. The scenario incorporates high traffic density with approximately 50 vehicles. Two adversarial vehicles are positioned to the left of the ego vehicle, both exhibiting dangerous tailgating behavior at critically close distances (1.38m and 0.55m respectively). This configuration creates a challenging situation where the ego vehicle must navigate through dense traffic while responding to immediate rear threats from aggressive followers.}}
LLM output1:{{
#------geometry.snippet------#
from NGSIM_env.road.lane import StraightLane, LineType
from NGSIM_env.road.road import RoadNetwork

network = RoadNetwork()
intersection_center = (0, 0)
lane_width = 3.5

# Intersection approach lane (slow speed)
approach_lane = StraightLane([-100, 0], [100, 0],
                           line_types=(LineType.CONTINUOUS, LineType.CONTINUOUS),
                           width=lane_width,
                           speed_limit=8)  # slow speed limit
network.add_lane("north", "south", approach_lane)

# Define intersection boundaries
intersection_start = -15
intersection_end = 15

#------spawn.snippet------#
from NGSIM_env.vehicle.behavior import IDMVehicle
from NGSIM_env.vehicle.humandriving import HumanLikeVehicle

# Ego vehicle approaching intersection
ego = HumanLikeVehicle on approach_lane at (intersection_start - 25, 0),
    with speed 6 m/s,
    heading 0 deg,
    behavior=intersection_approach

# Aggressive tailgating vehicles
adversary1 = HumanLikeVehicle on approach_lane.left_lane at (intersection_start - 23.623, 1.05),
    with speed 6.5 m/s,
    heading 0 deg,
    behavior=close_tailgating_left

adversary2 = HumanLikeVehicle on approach_lane.left_lane at (intersection_start - 24.453, 1.05),
    with speed 7 m/s,
    heading 0 deg,
    behavior=extremely_close_tailgating

# High density background traffic
background_vehicles = []
for i in range(50):  # high traffic density
    background_vehicles.append(IDMVehicle on network.random_lane())
    
#------behavior.snippet------#
from NGSIM_env.vehicle.controller import ControlledVehicle

behavior intersection_approach:
    when ego.position.x < intersection_start:
        ego.control = ControlledVehicle.keep_lane()
    else:
        ego.control = ControlledVehicle.brake(intensity=0.5)  # cautious braking

behavior close_tailgating_left:
    adversary1.target_speed = ego.speed * 1.08
    adversary1.control = ControlledVehicle.lane_following(distance=1.38)  # dangerous following

behavior extremely_close_tailgating:
    adversary2.target_speed = ego.speed * 1.15
    adversary2.control = ControlledVehicle.lane_following(distance=0.55)  # extremely dangerous
    
}}



LLM Input2:{{In the intersection scenario, the ego vehicle maintains a straight trajectory while executing a braking maneuver. The road section features special lane markings but no temporary traffic control measures. The traffic density is low with only 5 background vehicles present. Two adversarial vehicles create challenging conditions: 
1) Adversary 1 is positioned on the right side of the ego vehicle at an unsafe following distance of approximately 1.5 meters (tailgating behavior)
2) Adversary 2 approaches from the rear attempting an aggressive left lane change at a distance of about 3 meters. The scenario tests the ego vehicle's ability to handle simultaneous rear and side threats while maintaining proper intersection approach protocol.}}
LLM output2:{{
#------geometry.snippet------#
from NGSIM_env.road.lane import StraightLane, LineType
from NGSIM_env.road.road import RoadNetwork

network = RoadNetwork()
intersection_center = (0, 0)
lane_width = 3.5

# Intersection approach lane with special markings
approach_lane = StraightLane([-100, 0], [100, 0],
                           line_types=(LineType.SPECIAL, LineType.SPECIAL),
                           width=lane_width,
                           speed_limit=10)
network.add_lane("north", "south", approach_lane)

# Define intersection boundaries
intersection_start = -15
intersection_end = 15
#------spawn.snippet------#
from NGSIM_env.vehicle.behavior import IDMVehicle
from NGSIM_env.vehicle.humandriving import HumanLikeVehicle

# Ego vehicle approaching intersection
ego = HumanLikeVehicle on approach_lane at (intersection_start - 25, 0),
    with speed 8 m/s,
    heading 0 deg,
    behavior=intersection_approach

# Right-side tailgating adversary
adversary1 = HumanLikeVehicle on approach_lane.right_lane at (intersection_start - 23.513, -1.05),
    with speed 8.5 m/s,
    heading 0 deg,
    behavior=close_tailgating_right

# Rear adversary attempting left lane change
adversary2 = HumanLikeVehicle on approach_lane at (intersection_start - 28, 0),
    with speed 9 m/s,
    heading 0 deg,
    behavior=aggressive_left_change

# Low density background traffic
background_vehicles = []
for i in range(5):
    background_vehicles.append(IDMVehicle on network.random_lane())
    
#------behavior.snippet------#
from NGSIM_env.vehicle.controller import ControlledVehicle

behavior intersection_approach:
    when ego.position.x < intersection_start:
        ego.control = ControlledVehicle.keep_lane()
    else:
        ego.control = ControlledVehicle.brake(intensity=0.6)  # controlled braking

behavior close_tailgating_right:
    adversary1.target_speed = ego.speed * 1.1
    adversary1.control = ControlledVehicle.lane_following(distance=1.5)  # dangerous tailgating

behavior aggressive_left_change:
    when distanceTo(ego) < 30:
        adversary2.control = ControlledVehicle.lane_change(direction='left', aggressive=True)
        adversary2.target_speed = ego.speed * 1.2
        adversary2.min_lateral_distance = 3.0  # close proximity change
}}

The Hierarchical Scenario Repository provides a dictionary of scenario components corresponding to each element that you can choose from. When creating scenario representation, please first consider the following elements for each subcomponent. If there is no element that can describe a similar meaning, then create a new element yourself.

Based on the above description and examples, convert the following testing scenario text into the corresponding scenario representation:

{In the intersection scenario, the ego vehicle exhibits braking behavior while attempting to proceed straight through the junction. The simulation environment features a temporary road closure at the intersection, with the ego vehicle occupying a slow-speed lane. The scenario incorporates medium traffic density with approximately 20 vehicles in the vicinity. Two adversarial vehicles are present: Vehicle [1] is positioned to the left of the ego vehicle and performs an unsafe lane change to the right (lateral offset: 1.05 meters). Vehicle [2] follows behind the ego vehicle and executes an unsafe lane change to the left (lateral offset: 1.00 meters), creating a potentially hazardous situation.}

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

    # === Step 1: 嵌入建模 ===
    print("\n\n==================== 开始计算嵌入相似度矩阵 ====================\n")
    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(all_answers)
    sim_matrix = cosine_similarity(embeddings)

    # === Step 2: 平均相似度（每个答案与其他答案的相似度均值） ===
    avg_sim = sim_matrix.mean(axis=1)
    best_idx = int(np.argmax(avg_sim))
    final_answer = all_answers[best_idx]

    # === Step 3: 打印所有答案及其相似度 ===
    print("\n==================== 所有采样结果及相似度统计 ====================\n")
    for i, (ans, sim) in enumerate(zip(all_answers, avg_sim)):
        print(f"\n--- Answer {i+1} ---")
        print(f"[平均相似度] -> {sim:.4f}")
        print(ans[:500] + ("..." if len(ans) > 500 else ""))  # 可调显示长度

    # === Step 4: 最终选择结果 ===
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

    # 1. Intersection 区域
    if "intersection" in fields:
        lines.append(fields["intersection"])
        lines.append("")

    # 2. Ego 实体定义
    behavior = fields.get("ego_behavior", "DefaultBehavior")
    position = fields.get("ego_position", "road[0].lanes[0].centerline[0]")
    lines.append(f"ego = new Car with behavior {behavior},")
    lines.append(f"    at {position},")
    lines.append(f"    facing road[0].direction")
    lines.append("")

    # 3. 参数定义
    for key, val in fields.items():
        if key.startswith("param_"):
            name = key[len("param_"):]
            lines.append(f"param {name} = {val}")
    lines.append("")

    # 4. 行为类定义
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
        query, rag_chain, num_votes=2, delay=1
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
# converted_code = convert_dsl_to_highway_env_code(final_answer)
# convert_dsl_to_highway_env_code(final_answer, api_key="sk-xxxxx")

# 打印或保存结果
# print("\n==== 转换后的 Highway-Env Python 代码 ====\n")
# print(converted_code)
