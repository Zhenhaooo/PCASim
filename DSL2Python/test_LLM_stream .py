import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.runnables import RunnableMap
from langchain.prompts import PromptTemplate
import pandas as pd
from collections import Counter
import time
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import re
from collections import defaultdict

from collections import defaultdict, Counter
from difflib import SequenceMatcher
import time


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

embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vectorstore = FAISS.from_documents(docs, embedding_model)
retriever = vectorstore.as_retriever()

# === Step 3: Prompt 模板 ===
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

# === Step 4: 配置流式模型 ===

llm = ChatOpenAI(
    openai_api_key="sk-e8fe1470f8e3465b8cbbacab6decbdb5",
    openai_api_base="https://dashscope.aliyuncs.com/compatible-mode/v1",
    model="qwq-plus-2025-03-05",
    temperature=0.6,
    model_kwargs={"stream": True}
)

# === Step 5: 构建流式 RAG 链 ===
combine_docs_chain = create_stuff_documents_chain(
    llm=llm,
    prompt=cot_prompt
)

rag_chain = RunnableMap({
    "context": lambda x: retriever.invoke(x["question"]),
    "question": lambda x: x["question"]
}) | combine_docs_chain

# === Step 6: 示例问题 ===
query = """
Assuming you are an expert in autonomous driving testing, your task is to generate scenario representation from the following given testing scenario description text based on the Domain-Specific Language.

The Hierarchical Scenario Repository provides a dictionary of scenario components corresponding to each element that you can choose from. When creating scenario representation, please first consider the following elements for each subcomponent. If there is no element that can describe a similar meaning, then create a new element yourself.

{Dictionary of Hierarchical Scenario Repository}

Few-Shot

Below are two examples of the input testing scenario texts and the corresponding scenario representations:



LLM Input1:{{Ego vehicle approaches a four-way intersection intending to make an unprotected left turn. Multiple oncoming vehicles maintain straight trajectories through the junction at moderate speeds. Ego must yield right-of-way to all crossing traffic before initiating turning maneuver.}}

LLM output1:{{
IntersectionRegion = PolylineRegion([
    Point(road[0].lanes[0].centerline[10]),
    Point(road[1].lanes[0].centerline[10]),
    Point(road[2].lanes[0].centerline[10]), 
    Point(road[3].lanes[0].centerline[10])
]).buffer(15)

ego = new Car with behavior YieldLeftTurnBehavior,
    at road[1].lanes[0].centerline[5],
    facing road[1].direction

param oncoming_count = Range(2,4)
for i in range(oncoming_count):
    new Car as OncomingVehicle with:
        behavior FollowLaneBehavior(target_speed=Uniform(15,25)),
        at road[0].lanes[-1].centerline[-50] offset Uniform(-0.5,0.5),
        facing road[0].direction

class YieldLeftTurnBehavior(Behavior):
    def __init__(self, yield_distance=20, safe_gap=3):
        self.yield_zone = IntersectionRegion
        self.safety_margin = safe_gap

    def step(self):
        conflict_cars = [v for v in visibleObjects() 
            if isinstance(v, OncomingVehicle) 
            and v.position.distanceTo(self.agent.position) < 30]

        if len(conflict_cars) > 0:
            self.agent.setThrottle(0)
            self.agent.setBraking(0.8)
        else:
            steering = lane_centering_steer(self.agent)
            throttle = speed_maintenance_control(self.agent, 10)
            self.agent.setSteering(steering)
            self.agent.setThrottle(throttle)
}}



LLM Input2:{{At a signalized intersection, the ego vehicle executes a right-turn maneuver from the dedicated turning lane. A bicycle emerges from the adjacent bike lane on the right side, exhibiting lateral oscillation that may lead to unexpected incursion into the vehicular traffic lane. The ego must implement defensive yielding protocols including speed modulation and right-of-way assessment to ensure collision-free negotiation.}}

LLM output2:{{
# Define bicycle operation zone with lateral buffer
bikeCorridor = RectangularRegion(
    road.rightEdge.offsetBy(1.5),
    width=1.2,
    height=10,
    heading=roadDirection
)

# Right-turn lane geometry specification 
turnLane = road.lanes[-1]  # Rightmost lane as turning lane

# Ego positioning in dedicated turn lane
ego = new Car with behavior YieldToBikeBehavior(),
    at turnLane.start.offsetAlong(5),
    facing turnLaneDirection

# Bicycle with probabilistic lane incursion
bike = new Bicycle at bikeCorridor,
    with speed (1.5, 2.5),
    with behavior LateralDriftBehavior(
        baseSpeed=1.8,
        lateralShift=(0.3, 0.7),
        shiftProbability=0.4
    )

class YieldToBikeBehavior(Behavior):
    def __init__(self, reactionDistance=8):
        super().__init__()
        self.safeMargin = 2.5
        self.reactionDist = reactionDistance

    def step(self):
        bike = findClosest(type=Bicycle, maxDistance=15)
        if bike and self.checkConflict(bike):
            self.applyYieldProtocol(bike)
        else:
            self.maintainCourse()

    def checkConflict(self, bike):
        return (bike.distanceTo(ego) < self.reactionDist and 
                abs(bike.position.x - ego.position.x) < self.safeMargin)

    def applyYieldProtocol(self, bike):
        speedDiff = ego.speed - bike.speed
        decel = (speedDiff**2)/(2*self.reactionDist) if speedDiff > 0 else 0
        ego.setThrottle(max(0, 0.5 - decel/2))
        ego.setSteering(0.1)  # Gentle steering away from bike lane

# Bicycle-specific motion pattern
class LateralDriftBehavior(Behavior):
    def __init__(self, baseSpeed, lateralShift, shiftProbability):
        self.baseSpeed = baseSpeed
        self.shiftRange = lateralShift
        self.shiftProb = shiftProbability

    def step(self):
        if self.probability(self.shiftProb):
            lateral = self.uniform(*self.shiftRange)
            newPos = self.owner.position.offsetBy(lateral, 90)
            if newPos in bikeCorridor:
                self.owner.position = newPos
        self.owner.setSpeed(self.baseSpeed)
}}

The Hierarchical Scenario Repository provides a dictionary of scenario components corresponding to each element that you can choose from. When creating scenario representation, please first consider the following elements for each subcomponent. If there is no element that can describe a similar meaning, then create a new element yourself.

Based on the above description and examples, convert the following testing scenario text into the corresponding scenario representation:

{The scenario takes place at an intersection in a complex urban environment. The ego vehicle, a sedan, is initially located at coordinates (0, 0) and intends to drive straight through the intersection. The intersection is temporarily closed and includes a slow lane designation, indicating restricted maneuverability.

The surrounding traffic density is high, with approximately 50 other vehicles present. Weather conditions are clear, but despite dry road surfaces, visibility is low, posing a perception challenge for the ego vehicle.

Adversarial elements in the environment include sudden braking requirements, an emergency roadblock that disrupts normal traffic flow, and a construction zone that alters the road layout. These combined factors test the ego vehicle’s ability to safely navigate under constrained and dynamic urban conditions.
}

give me the code please!
"""

# # === Step 7: 流式调用并打印 ===
# response_text = ""
# print("\n==================== 思考过程 ====================\n")
# for chunk in rag_chain.stream({"question": query}):
#     print(chunk, end="", flush=True)
#     response_text += chunk
#
# print("\n\n==================== 完整回答 ====================\n")
# print(response_text)
#
# # === 获取相关文档片段 ===
# print("\n==================== 相关文档片段 ====================\n")
# for doc in retriever.invoke(query):
#     print(f"--- Page {doc.metadata.get('page', '?')} ---")
#     print(doc.page_content[:200] + "...")

# === Step 7: Self-Consistency Voting 方法 ===

def self_consistent_rag_chain_embedding(question, rag_chain, num_votes=5, delay=0.5):
    """
    流式版本 + 嵌入相似度 Voting 的 Self-Consistency 实现。
    输出每个采样答案的平均相似度，并选择最“中心”的答案作为最终结果。
    """
    all_answers = []

    for i in range(num_votes):
        try:
            response_text = ""
            print(f"\n---- 第 {i+1} 次采样（Embedding 模式） ----")
            for chunk in rag_chain.stream({"question": question}):
                print(chunk, end="", flush=True)
                response_text += chunk
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
#         response_text = ""
#         print(f"\n---- 第 {i+1} 次采样（Structured_clustered 模式） ----")
#         for chunk in rag_chain.stream({"question": question}):
#             print(chunk, end="", flush=True)
#             response_text += chunk
#         all_answers.append(response_text.strip())
#         time.sleep(delay)
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
#     # === Step 7: 从原始答案中选“结构最相近”的那个 DSL ===
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
#     print(f"\n 选择原始第 {best_idx + 1} 个 DSL 作为最终结果（字段最接近投票结果）")
#     print("\n==================== 最终 DSL 输出 ====================\n")
#     print(final_dsl)
#
#     return final_dsl
def fuzzy_equal(a, b, threshold=0.85):
    """模糊匹配两个字符串，返回是否视为相同字段值"""
    return SequenceMatcher(None, a or "", b or "").ratio() > threshold

def structured_voting_pipeline_clustered(question, rag_chain, num_votes=5, delay=0.5, sim_threshold=0.75):
    """
    流式LLM版本：DSL结构字段聚类（使用模糊匹配） + 最大簇字段投票 + 返回原始最接近 DSL。
    """
    all_answers = []
    for i in range(num_votes):
        response_text = ""
        print(f"\n---- 第 {i + 1} 次采样（Structured_clustered 流式） ----")
        for chunk in rag_chain.stream({"question": question}):
            print(chunk, end="", flush=True)
            response_text += chunk
        all_answers.append(response_text.strip())
        time.sleep(delay)

    parsed_list = [parse_dsl_fields(ans) for ans in all_answers]

    # === 使用模糊匹配的结构字段相似度函数 ===
    def field_similarity(dict1, dict2):
        keys = set(dict1.keys()).union(set(dict2.keys()))
        same = 0
        for k in keys:
            v1 = dict1.get(k)
            v2 = dict2.get(k)
            if fuzzy_equal(v1, v2):
                same += 1
        return same / len(keys) if keys else 0

    # === 聚类过程 ===
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
    print(f"\n✅ 选中聚类（大小={len(largest_cluster)}）作为最终投票候选")

    # 字段级投票
    field_values = defaultdict(list)
    for d in largest_cluster:
        for k, v in d.items():
            field_values[k].append(v)

    final_fields = {}
    for key, vals in field_values.items():
        most_common_val, freq = Counter(vals).most_common(1)[0]
        final_fields[key] = most_common_val

    # === 在原始DSL中找最相近的那个 ===
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

voting_mode = "structured_cluster"  # 可选 "embedding" 或 "structured"

if voting_mode == "embedding":
    final_answer, all_candidates, all_similarities = self_consistent_rag_chain_embedding(
        query, rag_chain, num_votes=5, delay=1
    )
elif voting_mode == "structured_cluster":
    final_answer = structured_voting_pipeline_clustered(query, rag_chain, num_votes=5, delay=1)
else:
    raise ValueError("未知 voting 模式")

print("\n==================== 最终输出 ====================\n")
print(final_answer)


# 可选：打印全部采样输出
# for idx, ans in enumerate(all_candidates):
#     print(f"\n--- Answer {idx+1} ---\n{ans}")

# === Step 9: 获取相关文档片段 ===
# print("\n==================== 相关文档片段 ====================\n")
# for doc in retriever.invoke(query):
#     print(f"--- Page {doc.metadata.get('page', '?')} ---")
#     print(doc.page_content[:200] + "...")
