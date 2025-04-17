import os
from pathlib import Path
from typing import List, Optional
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI

# ==== 0. 构建或加载 GitHub highway-env 示例代码索引 ====
def load_highway_code_index(index_path="highway_code_index"):
    if os.path.exists(index_path):
        return FAISS.load_local(index_path, HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-MiniLM-L6-v2"),allow_dangerous_deserialization=True)

    repo_path = "highway-env"
    # if not os.path.exists(repo_path):
        # os.system(f"git clone https://github.com/eleurent/highway-env {repo_path}")

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
        except:
            continue

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    chunks = splitter.split_documents(all_docs)

    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-MiniLM-L6-v2")
    store = FAISS.from_documents(chunks, embedding_model)

    store.save_local(index_path)
    return store

def build_conversion_prompt(dsl_text: str, rag_context: Optional[List[Document]] = None, fix_hint: str = None) -> str:
    rag_section = ""
    if rag_context:
        rag_section = "\n\n参考示例代码片段：\n" + "\n".join([d.page_content for d in rag_context[:3]])

    hint_section = f"\n\n[NOTE] Please correct the following syntax error from the last attempt:\n{fix_hint}\n" if fix_hint else ""

    return f"""
You are an expert in Python simulation and autonomous vehicle scenario setup.
You are provided with a custom Domain-Specific Language (DSL) for traffic scenarios.

Your job is to translate the following DSL into an executable Python script
compatible with a custom simulation environment that extends the structure of Highway-Env.
The DSL contains scene geometry, ego vehicle behaviors, dynamic objects, and logic.{hint_section}
{rag_section}

Instructions:
1. Preserve all logic in class behaviors (do not truncate).
2. Convert region and vehicle creation into Python objects using road network and vehicle modules.
3. Define all parameters, spawn positions, and behaviors using your simulation objects.
4. Return a complete and executable Python script, nothing else.
5. Format the output cleanly with imports and spacing.
6. Include a `if __name__ == '__main__'` block for launching the scenario if applicable.
7. Add all necessary imports such as Vehicle, Road, Lane, etc. from the simulator modules.
8. Ensure the output code compiles in Python 3.8+.

Here is the natural language description of the DSL:
In the intersection scenario, the ego vehicle exhibits braking behavior while attempting to proceed straight through the junction. The simulation environment features a temporary road closure at the intersection, with the ego vehicle occupying a slow-speed lane. The scenario incorporates medium traffic density with approximately 20 vehicles in the vicinity. Two adversarial vehicles are present: Vehicle [1] is positioned to the left of the ego vehicle and performs an unsafe lane change to the right (lateral offset: 1.05 meters). Vehicle [2] follows behind the ego vehicle and executes an unsafe lane change to the left (lateral offset: 1.00 meters), creating a potentially hazardous situation.

And here is the DSL:

```dsl
{dsl_text.strip()}
```
"""

def call_llm_for_code(prompt: str) -> str:
    llm = ChatOpenAI(
        openai_api_key="sk-4d8332cd221a45d9b505e5f93d7122b2",
        openai_api_base="https://api.deepseek.com/v1",
        model="deepseek-reasoner",
        temperature=0.6
    )
    return llm.invoke(prompt).content

# 清洗 LLM 输出
def strip_code_block_wrappers(text: str) -> str:
    if text.strip().startswith("```"):
        return "\n".join(line for line in text.strip().splitlines() if not line.strip().startswith("```"))
    return text

def clean_code_start(text: str) -> str:
    lines = text.strip().splitlines()
    for idx, line in enumerate(lines):
        if line.strip().startswith("from") or line.strip().startswith("import"):
            return "\n".join(lines[idx:])
    return text

# 验证 Python 语法
def validate_python_code(code: str) -> (bool, str):
    try:
        compile(code, filename="<dsl_generated>", mode="exec")
        return True, ""
    except SyntaxError as e:
        return False, f"{e.msg} on line {e.lineno}: {e.text.strip()}"

def save_code_to_file(code: str, path: str) -> None:
    Path(Path(path).parent).mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(code)
    print(f"\n  场景代码已保存至: {path}\n")

def try_run_generated_scenario(path: str) -> None:
    print("\n 尝试加载并运行生成的场景...")
    try:
        exec_globals = {}
        with open(path, "r", encoding="utf-8") as f:
            code = f.read()
        exec(code, exec_globals)
        print(" 模拟器代码运行成功。")
    except Exception as e:
        print(" 运行失败：", e)

# 主函数
def convert_dsl_to_highway_env_code(dsl_text: str,
                                    output_file: str = "auto_scenario_llm.py",
                                    max_debug_attempts: int = 3) -> str:
    code_index = load_highway_code_index()
    rag_context = code_index.similarity_search(dsl_text, k=3)
    debug_attempt = 1
    fix_hint = None

    while debug_attempt <= max_debug_attempts:
        print(f"\n 第 {debug_attempt} 次尝试调用 LLM 生成代码...")
        prompt = build_conversion_prompt(dsl_text, rag_context, fix_hint)
        raw_code = call_llm_for_code(prompt)
        cleaned_code = clean_code_start(strip_code_block_wrappers(raw_code))

        print("\n 正在验证生成代码语法...\n")
        is_valid, err_msg = validate_python_code(cleaned_code)
        if is_valid:
            save_code_to_file(cleaned_code, output_file)
            try_run_generated_scenario(output_file)
            return cleaned_code
        else:
            print(f" 第 {debug_attempt} 次 Debug 失败，错误信息：{err_msg}")
            fix_hint = f"SyntaxError: {err_msg}"
            debug_attempt += 1

    print(" 多轮尝试后仍然无法生成合法的 Python 代码。请检查 DSL 或 Prompt。")
    return ""

## 最后可以再运行这个进行一个纠正
# 请帮我把这段 highway-env 自定义道路与导航行为代码接入 highway_env 的渲染机制，实现完整的交互可视化测试（带画面渲染），不修改原有逻辑，仅封装为环境后用 env.render() 观察效果。注意切换用 gymnasium，并且主车和背景车的行为也请放入其中可以在场景中表现出来。
"""
pip install opencv-python
pip install pykalman

pip install gymnasium
"""


if __name__ == "__main__":
    final_answer = """
#------geometry.snippet------#
from NGSIM_env.road.lane import StraightLane, LineType
from NGSIM_env.road.road import RoadNetwork

network = RoadNetwork()
intersection_center = (0, 0)
lane_width = 3.5

# Slow-speed lane with temporary closure
approach_lane = StraightLane([-100, 0], [100, 0],
                           line_types=(LineType.CONTINUOUS, LineType.CONTINUOUS),
                           width=lane_width,
                           speed_limit=8)  # Reduced speed for closure
network.add_lane("north", "south", approach_lane)

# Define intersection boundaries
intersection_start = -15
intersection_end = 15

#------spawn.snippet------#
from NGSIM_env.vehicle.behavior import IDMVehicle
from NGSIM_env.vehicle.humandriving import HumanLikeVehicle

# Ego vehicle with braking behavior
ego = HumanLikeVehicle on approach_lane at (intersection_start - 25, 0),
    with speed 6 m/s,
    heading 0 deg,
    behavior=intersection_braking

# Left-side adversary making unsafe right change
adversary1 = HumanLikeVehicle on approach_lane.left_lane at (intersection_start - 23.5, 1.05),
    with speed 7 m/s,
    heading 0 deg,
    behavior=aggressive_right_change

# Rear adversary making unsafe left change
adversary2 = HumanLikeVehicle on approach_lane at (intersection_start - 28, 0),
    with speed 7.5 m/s,
    heading 0 deg,
    behavior=aggressive_left_change

# Medium density background traffic
background_vehicles = []
for i in range(20):
    background_vehicles.append(IDMVehicle on network.random_lane())

#------behavior.snippet------#
from NGSIM_env.vehicle.controller import ControlledVehicle

behavior intersection_braking:
    when ego.position.x < intersection_start:
        ego.control = ControlledVehicle.keep_lane()
    else:
        ego.control = ControlledVehicle.brake(intensity=0.55)  # Moderate braking

behavior aggressive_right_change:
    adversary1.target_speed = ego.speed * 1.15
    adversary1.control = ControlledVehicle.lane_change(
        direction='right', 
        aggressive=True,
        min_lateral_distance=1.05  # Unsafe lateral offset
    )

behavior aggressive_left_change:
    adversary2.target_speed = ego.speed * 1.25
    adversary2.control = ControlledVehicle.lane_change(
        direction='left',
        aggressive=True,
        min_lateral_distance=1.00  # Close proximity maneuver
    )
    """
    convert_dsl_to_highway_env_code(final_answer)