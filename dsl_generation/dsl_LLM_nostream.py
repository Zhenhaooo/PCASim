"""
pip install langchain langchain-community openai faiss-cpu
pip install langchain-huggingface
pip install -U sentence-transformers

pip install scikit-learn
"""
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'  # 使用镜像源
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
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
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../code_compiler'))
# print(sys.path)
from json_find import find_similar_descriptions
from dsl_to_env_generator_test import convert_dsl_to_highway_env_code


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

rag_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    chain_type="stuff",
    chain_type_kwargs={
        "prompt": cot_prompt,
        "document_variable_name": "context"  # 显式绑定文档变量
    },
    return_source_documents=True
)

# === 问题 ===
description_text = ''' '''

json_path = os.path.join(ROOT_DIR, "Corpus/Description/description_fileinfo_pairs.json")
similar_descriptions = find_similar_descriptions(description_text, json_path, top_k=3)

# 输出结果
for item in similar_descriptions:
    print(f"DataSet: {item['DataSet']}")
    print(f"Description: {item['Description']}")
    print(f"Similarity: {item['Similarity']:.4f}\n")

# === 提取示例并生成对应输出 ===
# 先从 similar_descriptions 获取输入
input1 = similar_descriptions[0]['Description']
input2 = similar_descriptions[1]['Description']

try:
    row_idx1 = sheets["chatstyle.description"].iloc[:, 0].tolist().index(input1)
    row_idx2 = sheets["chatstyle.description"].iloc[:, 0].tolist().index(input2)
except ValueError as e:
    print(f"查找描述对应行时出错: {e}")
    row_idx1, row_idx2 = None, None

def get_scenario_representation(row_idx):
    if row_idx is None:
        return '{}'
    geometry = sheets["geometry.snippet"].iloc[row_idx, 0]
    spawn = sheets["spawn.snippet"].iloc[row_idx, 0]
    behavior = sheets["behavior.snippet"].iloc[row_idx, 0]
    return {
        "geometry": geometry,
        "spawn": spawn,
        "behavior": behavior
    }

output1 = get_scenario_representation(row_idx1)
output2 = get_scenario_representation(row_idx2)

# 场景库占位内容
scenario_repository = "你的Hierarchical Scenario Repository字典内容"


query = f"""
Assuming you are an expert in autonomous driving testing, your task is to generate scenario representation from the following given testing scenario description text based on the Domain-Specific Language.

The Hierarchical Scenario Repository provides a dictionary of scenario components corresponding to each element that you can choose from. When creating scenario representation, please first consider the following elements for each subcomponent. If there is no element that can describe a similar meaning, then create a new element yourself.

{scenario_repository}

Few-Shot

Below are two examples of the input testing scenario texts and the corresponding scenario representations:

LLM Input1:{{{input1}}}
LLM output1:{{{output1}}}

LLM Input2:{{{input2}}}
LLM output2:{{{output2}}}

The Hierarchical Scenario Repository provides a dictionary of scenario components corresponding to each element that you can choose from. When creating scenario representation, please first consider the following elements for each subcomponent. If there is no element that can describe a similar meaning, then create a new element yourself.

Based on the above description and examples, convert the following testing scenario text into the corresponding scenario representation:
{description_text}

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

print("\n==== 转换后的 Highway-Env Python 代码 ====\n")
print(converted_code)
