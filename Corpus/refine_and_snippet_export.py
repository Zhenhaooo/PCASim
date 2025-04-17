import openai
import pandas as pd
import time
import re
import json
import os

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter



# === 初始化新版 openai 客户端 ===
client = openai.OpenAI(
    api_key="sk-4d8332cd221a45d9b505e5f93d7122b2",  # ← 替换为你的 DeepSeek API Key
    base_url="https://api.deepseek.com/v1"
)

model_name = "deepseek-chat"

# === Scenic DSL 模板库：从已有文件加载历史 snippet（记忆机制）===
template_path = "./will_template_v2/snippet_template_library.json"
if os.path.exists(template_path):
    with open(template_path, "r", encoding="utf-8") as f:
        scenic_templates = json.load(f)
else:
    scenic_templates = {"geometry": [], "spawn": [], "behavior": []}

# === 正则提取函数 ===
def extract_snippet(text, keyword):
    pattern = rf"【{re.escape(keyword)}】\s*```scenic(.*?)```"
    match = re.search(pattern, text, re.DOTALL)
    return match.group(1).strip() if match else ""

def extract_section(text, keyword):
    pattern = rf"【{re.escape(keyword)}】(.*?)(【|$)"
    match = re.search(pattern, text, re.DOTALL)
    return match.group(1).strip() if match else ""

def load_highway_code_index(index_path="highway_code_index"):
    if os.path.exists(index_path):
        return FAISS.load_local(
            index_path,
            HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-MiniLM-L6-v2"),
            allow_dangerous_deserialization=True
        )

    repo_path = "highway-env"  # ← 你的项目 highway-env 根目录
    py_files = []
    for root, _, files in os.walk(repo_path):
        for f in files:
            if f.endswith(".py"):
                py_files.append(os.path.join(root, f))

    all_docs = []
    for path in py_files:
        try:
            loader = TextLoader(path, encoding="utf-8")
            all_docs.extend(loader.load())
        except Exception as e:
            continue

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    chunks = splitter.split_documents(all_docs)

    store = FAISS.from_documents(chunks, HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-MiniLM-L6-v2"))
    store.save_local(index_path)
    return store

# === 读取原始语料 Excel 文件 ===
df = pd.read_excel("./will_Description_V3/assembled_descriptions.xlsx")

# === 输出字段容器 ===
descriptions = []
geometry_snippets = []
spawn_snippets = []
behavior_snippets = []

# === few-shot 示例拼接函数 ===
def format_snippet_examples(templates, max_examples=6):
    examples = ""
    count = min(max_examples, len(templates["geometry"]))
    for i in range(count):
        examples += f"""【geometry.snippet】
```scenic
{templates["geometry"][i]}
【spawn.snippet】

```scenic

{templates["spawn"][i]}
【behavior.snippet】

```scenic

{templates["behavior"][i]}
"""
    return examples

# === 加载或构建 highway-env 中的代码向量索引 ===
code_store = load_highway_code_index()


# === 主处理循环 ===
for i, row in df.iterrows():
    original = row["final_natural_language"]
    # 拼接 few-shot 记忆库示例
    few_shot_examples = format_snippet_examples(scenic_templates)

    # 检索相关代码片段作为上下文
    rag_context = code_store.similarity_search(original, k=5)
    rag_snippets = "\n\n".join([doc.page_content for doc in rag_context])

    prompt = f"""
你是一个自动驾驶仿真专家，请完成以下任务：

1. **润色自然语言场景描述**：将下面的原始自然语言驾驶场景描述转换为清晰、自然且具有仿真视角的英文描述（段落形式）。必须保持语义一致性，原意不能更改。
2. 你应使用我们项目 highway-env 已有的基础类，避免引入过多自定义类，以保证结构统一与后续可维护性。(例如:
from NGSIM_env import utils
from NGSIM_env.envs.common.abstract import AbstractEnv
from NGSIM_env.road.road import Road, RoadNetwork
from NGSIM_env.vehicle.behavior import IDMVehicle
from NGSIM_env.vehicle.humandriving import HumanLikeVehicle, InterActionVehicle, Pedestrian, IntersectionHumanLikeVehicle
from NGSIM_env.road.lane import LineType, StraightLane, PolyLane, PolyLaneFixedWidth
from NGSIM_env.utils import *
import pickle
import lanelet2
from threading import Thread
from shapely.geometry import LineString, Point)
你可以从以下代码中了解已定义的类与用法：{rag_snippets}
3. **结构要求**：
   - geometry.snippet 定义一级场景：主车的位置，和运动方向
   - spawn.snippet 定义初始车辆（主车和对抗车的类型、相对位置、朝向等信息。
   - behavior.snippet 描述车辆的行为（主车和对抗车）
5. 确保三个 DSL 模块的语义与润色后的自然语言描述一致，DSL 中每个元素都应有清晰语义映射。

以下是你之前生成的 DSL 模块示例（记忆库）： {few_shot_examples}

请以如下格式输出：
【润色描述】
...

【geometry.snippet】
```scenic
...
```

【spawn.snippet】
```scenic
...
```

【behavior.snippet】
```scenic
...
```

原始描述：
\"\"\"{original}\"\"\"
"""

    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": "You are an expert in generating Scenic DSL from driving scene descriptions."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2
        )

        content = response.choices[0].message.content

        desc = extract_section(content, "润色描述")
        geo = extract_snippet(content, "geometry.snippet")
        spawn = extract_snippet(content, "spawn.snippet")
        behavior = extract_snippet(content, "behavior.snippet")

        descriptions.append(desc)
        geometry_snippets.append(geo)
        spawn_snippets.append(spawn)
        behavior_snippets.append(behavior)

        # === 更新 Scenic 记忆库 ===
        if geo not in scenic_templates["geometry"]:
            scenic_templates["geometry"].append(geo)
        if spawn not in scenic_templates["spawn"]:
            scenic_templates["spawn"].append(spawn)
        if behavior not in scenic_templates["behavior"]:
            scenic_templates["behavior"].append(behavior)

        print(f"[{i}] Success")
        time.sleep(1)

    except Exception as e:
        print(f"[{i}] Error: {e}")
        descriptions.append("")
        geometry_snippets.append("")
        spawn_snippets.append("")
        behavior_snippets.append("")

# === 保存 Scenic snippet 模板库 ===
dir_path = os.path.dirname(template_path)
if not os.path.exists(dir_path):
    os.makedirs(dir_path)

with open(template_path, "w", encoding="utf-8") as f:
    json.dump(scenic_templates, f, indent=2, ensure_ascii=False)

# === 多 Sheet 输出 Excel ===
sheets = {
    "chatstyle.description": pd.DataFrame({"chatstyle_description": descriptions}),
    "geometry.snippet": pd.DataFrame({"geometry.snippet": geometry_snippets}),
    "spawn.snippet": pd.DataFrame({"spawn.snippet": spawn_snippets}),
    "behavior.snippet": pd.DataFrame({"behavior.snippet": behavior_snippets}),
}

with pd.ExcelWriter("final_snippet_export.xlsx", engine="openpyxl") as writer:
    for sheet_name, df_sheet in sheets.items():
        df_sheet.to_excel(writer, sheet_name=sheet_name, index=False)

print("\n 所有内容已生成完成，保存为 final_snippet_export.xlsx 和 snippet_template_library.json")
