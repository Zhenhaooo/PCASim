# dsl_to_env_generator.py
"""
将 DSL 场景描述交给 LLM 进行结构化转换，生成可执行的 Highway-Env 场景配置代码，
并自动进行语法验证。
"""

import os
from pathlib import Path
from langchain_openai import ChatOpenAI

# ==== 1. 构造转换提示词 ====
def build_conversion_prompt(dsl_text: str) -> str:
    return f"""
You are an expert in Python simulation and autonomous vehicle scenario setup.
You are provided with a custom Domain-Specific Language (DSL) for traffic scenarios.

Your job is to translate the following DSL into an executable Python script
compatible with a custom simulation environment that extends the structure of Highway-Env.
The DSL contains scene geometry, ego vehicle behaviors, dynamic objects, and logic.

Instructions:
1. Preserve all logic in class behaviors (do not truncate).
2. Convert region and vehicle creation into Python objects using road network and vehicle modules.
3. Use realistic mappings: e.g., road[1].lanes[0].centerline[5] becomes a call to a SplineLane point or a Road class method.
4. Define all parameters, spawn positions, and behaviors using your simulation objects.
5. Return a complete and executable Python script, nothing else.
6. Format the output cleanly with imports and spacing.
7. Include a `if __name__ == '__main__'` block for launching the scenario if applicable.
8. Add all necessary imports such as Vehicle, Road, Lane, etc. from the simulator modules.
9. Ensure the output code compiles in Python 3.8+.

Here is the DSL:

```dsl
{dsl_text.strip()}
```
"""

# ==== 2. 调用 LLM 接口 ====
def call_llm_for_code(prompt: str) -> str:
    llm = ChatOpenAI(
        openai_api_key="sk-4d8332cd221a45d9b505e5f93d7122b2",
        openai_api_base="https://api.deepseek.com/v1",
        model="deepseek-reasoner",
        temperature=0.6
    )
    return llm.invoke(prompt).content

# ==== 3. 验证生成代码是否合法 Python ====
def validate_python_code(code: str) -> bool:
    try:
        compile(code, filename="<dsl_generated>", mode="exec")
        return True
    except SyntaxError as e:
        print(" LLM 生成的代码存在语法错误：", e)
        return False

# ==== 4. 保存为 .py 文件 ====
def save_code_to_file(code: str, path: str = "output/auto_scenario_llm.py") -> None:
    Path(Path(path).parent).mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(code)
    print(f"\n 场景代码已保存至: {path}\n")

# ==== 5. 主函数封装 ====
def convert_dsl_to_highway_env_code(dsl_text: str,
                                    output_file: str = "output/auto_scenario_llm.py") -> str:
    prompt = build_conversion_prompt(dsl_text)
    python_code = call_llm_for_code(prompt)

    print("\n 正在验证生成代码语法...\n")
    if validate_python_code(python_code):
        save_code_to_file(python_code, output_file)
        return python_code
    else:
        print(" 生成代码语法不通过，已终止保存。请检查 DSL 或 LLM 输出。")
        return ""

# ==== 6. 示例运行（调试用，可直接替换 DSL 内容） ====
if __name__ == "__main__":
    final_answer = """
```scenic
# Scenario parameters
param visibility_distance = 40  # Reduced visibility range
param traffic_density = 50
param slow_lane_speed = 8  # m/s (~28.8 km/h)

# Intersection geometry definition
IntersectionRegion = PolylineRegion([
    Point(0, 0),
    Point(50, 0),
    Point(50, 50),
    Point(0, 50)
]).buffer(20)

# Construction zone specification
ConstructionZone = PolylineRegion(
    IntersectionRegion.points
).narrowLanes(
    widthReduction=1.5,
    bufferDistance=15
)

# Emergency roadblock configuration
Roadblock = RectangularRegion(
    Point(25, 40),
    width=8,
    height=3,
    heading=90
)

# Ego vehicle definition
ego = new Car with behavior ConstrainedNavigation(
    at Point(0, 0),
    facing IntersectionRegion.orientation,
    with:
        - target_speed = slow_lane_speed
        - perception_range = visibility_distance
)

# Traffic generation
require len(vehicles) >= traffic_density
for _ in range(traffic_density):
    new Car with:
        behavior AggressiveUrbanBehavior(
            base_speed=Uniform(10, 15),
            braking_probability=0.3
        ),
        at IntersectionRegion.offsetBy(
            Uniform(-30, 30),
            Uniform(-30, 30)
        ),
        facing Choose([0, 90, 180, 270])

# Behavioral definitions
class ConstrainedNavigation(Behavior):
    def __init__(self, reaction_time=1.5):
        self.obstacle_buffer = 5.0
        self.construction_adjustment = 0.7

    def step(self):
        obstacles = self.detect_obstacles()
        
        if self.emergency_stop_required(obstacles):
            self.execute_full_stop()
        elif self.in_construction_zone():
            self.adjust_for_construction()
        else:
            self.maintain_progress()

    def detect_obstacles(self):
        return [obj for obj in visibleObjects()
                if obj.distanceTo(ego) < self.perception_range
                and not isinstance(obj, RoadSurface)]

    def emergency_stop_required(self, obstacles):
        return any(obj.speedDelta(ego) > 5 and obj.distanceTo(ego) < 10
                 for obj in obstacles)

    def adjust_for_construction(self):
        self.agent.target_speed *= self.construction_adjustment
        self.agent.setSteering(lane_centering_steer() + 0.15)

class AggressiveUrbanBehavior(Behavior):
    def __init__(self, base_speed, braking_probability):
        self.brake_prob = braking_probability
        self.base_speed = base_speed

    def step(self):
        if self.probability(self.brake_prob):
            self.execute_sudden_brake()
        else:
            self.maintain_speed()

    def execute_sudden_brake(self):
        self.agent.setBraking(Uniform(0.6, 1.0))
        self.agent.setThrottle(0)

# Environmental constraints
require:
    - Roadblock.intersects(IntersectionRegion)
    - ConstructionZone.overlaps(ego.path)
    - eventually ego exits IntersectionRegion within 60 seconds
``` 

**Chain of Thought Verification:**
1. **Coordinate Alignment:** Ego positioned at (0,0) with explicit coordinates rather than road-based positioning to match input specification
2. **Traffic Density:** Implemented through parametric vehicle count and spatial distribution around intersection
3. **Visibility Paradox:** Clear weather with low visibility modeled through perception_range parameter
4. **Dynamic Elements:** 
   - Sudden braking via probabilistic braking behavior
   - Roadblock as concrete region with collision constraints
   - Construction zone with lane narrowing effect
5. **Behavioral Validation:** 
   - Speed modulation through slow_lane_speed parameter
   - Collision avoidance through obstacle detection buffer
   - Construction zone navigation adjustments
```
    """
    convert_dsl_to_highway_env_code(final_answer)
