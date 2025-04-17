import openai
import pandas as pd
import time
import re
import json
import os

# === 初始化新版 openai 客户端 ===
client = openai.OpenAI(
    api_key="sk-4d8332cd221a45d9b505e5f93d7122b2",  # ← 替换为你的 DeepSeek API Key
    base_url="https://api.deepseek.com/v1"
)

model_name = "deepseek-reasoner"

# === Scenic DSL 模板库：从已有文件加载历史 snippet（记忆机制）===
template_path = "./will_template/snippet_template_library.json"
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

# === 读取原始语料 Excel 文件 ===
df = pd.read_excel("./will_Description_V2/assembled_descriptions.xlsx")

# === 输出字段容器 ===
descriptions = []
geometry_snippets = []
spawn_snippets = []
behavior_snippets = []

# === 主处理循环 ===
for i, row in df.iterrows():
    original = row["final_natural_language"]

    prompt = f"""
你是一个自动驾驶仿真专家，请完成以下任务：

1. **润色自然语言场景描述**：将下面的原始自然语言驾驶场景描述转换为清晰、自然且具有仿真视角的英文描述（段落形式），风格参考 ChatScene。必须保持语义一致性，原意不能缺失。
2. **生成模块化 Scenic DSL 代码片段**：将润色后的描述，转换为三个 Scenic DSL 的模块（geometry、spawn、behavior）。你应使用 Highway-Env 官方已有的基础类（从下面的结构可以去提取到基础类，如 
from highway_env import utils
from highway_env.envs.common.abstract import AbstractEnv
from highway_env.road.lane import AbstractLane, CircularLane, LineType, StraightLane
from highway_env.road.regulation import RegulatedRoad
from highway_env.road.road import RoadNetwork
from highway_env.vehicle.kinematics import Vehicle
from highway_env.road.road import Road
from highway_env.utils import Vector
from highway_env.vehicle.objects import RoadObject
......等等），避免引入过多自定义类，以保证结构统一与后续可维护性。

3. **结构要求**：
   - geometry.snippet 定义道路、车道线、障碍物、信号灯等静态场景构成。
   - spawn.snippet 定义初始车辆（主车和其他车）的类型、位置、朝向等信息。
   - behavior.snippet 描述车辆的行为模式（如加速、换道、避障等）。
5. 确保三个 DSL 模块的语义与润色后的自然语言描述一致，DSL 中每个元素都应有清晰语义映射。

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
            temperature=0.3
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
