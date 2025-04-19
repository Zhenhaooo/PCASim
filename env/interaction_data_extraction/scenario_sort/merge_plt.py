import os
import matplotlib.pyplot as plt
import cairosvg
import numpy as np
from PIL import Image
import io

# 设置 SVG 文件路径
vehicle_dir = '/home/chuan/work/TypicalScenarioExtraction/data/interaction_data/scenario_data/pic/compare_per'
svg_files = [f for f in os.listdir(vehicle_dir) if f.endswith('.svg')]

# 计算需要的行列数
rows = 2  # 行数（可以根据需要调整）
cols = 4  # 列数（可以根据需要调整）

# 创建一个 3x3 的网格
fig, axs = plt.subplots(rows, cols, figsize=(15, 15))

# 遍历 SVG 文件并加载到子图中
for i, ax in enumerate(axs.flat):
    if i < len(svg_files):
        svg_path = os.path.join(vehicle_dir, svg_files[i])

        # 使用 cairosvg 将 SVG 转换为 PNG 数据（在内存中）
        png_data = cairosvg.svg2png(url=svg_path)

        # 将内存中的 PNG 数据加载为 PIL 图像
        img = Image.open(io.BytesIO(png_data))
        ax.imshow(img)
        ax.axis('off')  # 不显示坐标轴

    else:
        ax.axis('off')  # 如果 SVG 文件少于 9 个，关闭多余的子图

# 调整布局，避免子图重叠
plt.tight_layout()
if not os.path.exists('/home/chuan/work/TypicalScenarioExtraction//data/interaction_data/scenario_data/pic/combined'):
    os.makedirs('/home/chuan/work/TypicalScenarioExtraction//data/interaction_data/scenario_data/pic/combined')
# 保存拼接后的图像
save_path = '/home/chuan/work/TypicalScenarioExtraction//data/interaction_data/scenario_data/pic/combined/combined_image_compare_per.png'
plt.savefig(save_path, format='png')

# 显示合并后的图像
plt.show()
