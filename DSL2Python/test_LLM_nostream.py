"""
pip install langchain langchain-community openai faiss-cpu
pip install langchain-huggingface
pip install -U sentence-transformers

pip install scikit-learn
"""
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'  # 使用镜像源
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))

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

# === 问题 ===
query = """
Assuming you are an expert in autonomous driving testing, your task is to generate scenario representation from the following given testing scenario description text based on the Domain-Specific Language.

The Hierarchical Scenario Repository provides a dictionary of scenario components corresponding to each element that you can choose from. When creating scenario representation, please first consider the following elements for each subcomponent. If there is no element that can describe a similar meaning, then create a new element yourself.

{Dictionary of Hierarchical Scenario Repository}

Few-Shot

Below are two examples of the input testing scenario texts and the corresponding scenario representations:

LLM Input1:{{In the intersection scenario analysis from the [vehicle_tracks_000_trajectory_set_27.json] dataset, the ego vehicle demonstrates a straight-line braking maneuver under normal lane conditions without temporary traffic pattern changes. The adversarial environment features high traffic density (50 vehicles) with two critical interactions: 
- Adversarial vehicle 1 positioned rearward executes a sudden brake maneuver with 1.22s reaction latency
- Adversarial vehicle 2 approaching laterally from the left initiates sudden braking with 1.93s reaction delay}}
LLM output1:{{
#------geometry.snippet------#
IntersectionScenario: 
    road_network = RoadNetwork.from_real_dataset(
        dataset='vehicle_tracks_000_trajectory_set_27.json',
        lane_types=[StraightLane, StraightLane]
    )
    
ego = Car:
    lane = road_network.get_lane(lane_index=0)
    initial_position = lane.center_at(30.0)  # Mid-intersection approach
    heading = lane.heading_at(30.0)

#------spawn.snippet------#
ego_vehicle = HumanLikeVehicle:
    lane = ego.lane
    position = ego.initial_position
    heading = ego.heading
    behavior_type = IDMBehavior
    
adv_vehicle1 = InterActionVehicle: 
    lane = ego.lane
    position = ego.initial_position + (-25.0, 0.0)  # 25m rear offset
    speed = 12.0  # Approaching from behind
    target_speed = 8.0
    
adv_vehicle2 = InterActionVehicle:
    lane = road_network.get_lane(lane_index=1)
    position = ego.initial_position + (0.0, 3.5)  # Left adjacent lane
    speed = 10.5  # Lateral approach speed
    lane_offset = 2.0
    
#------behavior.snippet------#
ego_behavior = IDMBehavior:
    target_speed = 30.0
    normal_acceleration = 2.0
    emergency_deceleration = -4.5  # Braking magnitude
    
adv1_behavior = SuddenBrakeBehavior:
    trigger_distance = 20.0  # Activation threshold
    reaction_time = 1.224
    deceleration_profile = [-3.0, -4.0, -5.0]  # Progressive braking
    
adv2_behavior = LateralBrakeBehavior:
    lateral_offset = 3.5
    reaction_delay = 1.928 
    max_deceleration = -6.0
    steering_effect = 0.15  # Lateral movement during brake
    
}}



LLM Input2:{{In the intersection scenario analysis, the ego vehicle exhibits straight-path braking behavior within a slow-type lane configuration without temporary traffic modifications. The environment features medium traffic density with 20 surrounding vehicles. Critical adversarial interactions include:  
- Adversarial Vehicle 1 positioned ahead performing an unsafe right lane change (lateral offset: 2.97m)  
- Adversarial Vehicle 2 preceding the ego vehicle while speeding with velocity increment of 11.33m/s  
This scenario reconstruction is based on trajectory patterns from the [vehicle_tracks_000_trajectory_set_67.json] dataset.}}
LLM output2:{{
#------geometry.snippet------#
IntersectionScenario:
    road_network = RoadNetwork.from_real_dataset(
        dataset='vehicle_tracks_000_trajectory_set_67.json',
        lane_types=[StraightLane, StraightLane],
        lane_speed_limits={'slow': 30.0}
    )
    
ego = Car:
    lane = road_network.get_lane(lane_index=0)
    initial_position = lane.center_at(55.0)  # Conflict zone alignment
    heading = lane.heading_at(55.0)
    lane_spec = 'slow'  # Original lane classification

#------spawn.snippet------#
ego_vehicle = HumanLikeVehicle:
    lane = ego.lane
    position = ego.initial_position
    heading = ego.heading
    target_speed = 25.0  # Initial speed from braking pattern
    behavior_type = IDMBehavior
    
adv_vehicle1 = InterActionVehicle:
    lane = road_network.get_lane(lane_index=0)
    position = ego_vehicle.position + (8.0, 0.0)  # Front position
    speed = 22.0  # Approaching speed
    lateral_offset = 3.0  # Dataset-calculated 2.97m
    
adv_vehicle2 = InterActionVehicle:
    lane = road_network.get_lane(lane_index=0)
    position = ego_vehicle.position + (15.0, 0.0)  # Leading position
    speed = ego_vehicle.target_speed + 11.33  # Speed differential
    acceleration_profile = [2.5, 3.0]
    
#------behavior.snippet------#
ego_behavior = IDMBehavior:
    min_gap = 3.2  # Conservative spacing
    emergency_deceleration = -5.2  # Observed braking magnitude
    reaction_time = 1.35  # Response delay
    
adv1_behavior = SuddenLaneChangeBehavior:
    trigger_condition = DistanceThreshold(8.0)
    direction = 'right'
    lateral_acceleration = 0.78  # m/s²
    max_offset = 3.0  # Final lateral displacement
    
adv2_behavior = SpeedingBehavior:
    speed_increment = 11.33
    acceleration_phases = [
        (0-2s: 2.5 m/s²),
        (2s+: 3.0 m/s²)
    ]
    speed_maintain_duration = 5.0
}}

The Hierarchical Scenario Repository provides a dictionary of scenario components corresponding to each element that you can choose from. When creating scenario representation, please first consider the following elements for each subcomponent. If there is no element that can describe a similar meaning, then create a new element yourself.

Based on the above description and examples, convert the following testing scenario text into the corresponding scenario representation:

{In an urban intersection scenario reconstructed from the [vehicle_tracks_000_trajectory_set_28.json] dataset, the ego vehicle exhibits consecutive braking maneuvers while maintaining straight-line motion through a fast-lane section with temporary traffic control measures. The medium-density traffic environment (20 vehicles) introduces two adversarial agents: Adversarial Vehicle 1 executes an unsafe leftward lane change from the right adjacent lane at 3.61m lateral distance, while Adversarial Vehicle 2 performs sudden braking maneuvers with 2.75s delayed response time on the same lateral plane.}

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
