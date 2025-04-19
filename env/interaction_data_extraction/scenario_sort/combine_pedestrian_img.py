import os
import matplotlib.pyplot as plt
from utils.config import pedestrian_pic_save_path
import cairosvg
from PIL import Image
import io

def save_compare_img(ped_img_path):
    mutual_path = os.path.join(ped_img_path, 'mutual')
    per_path = os.path.join(ped_img_path, 'per')
    combined_img_path = os.path.join(ped_img_path, 'combined_img')

    # 创建保存合并图像的文件夹（如果不存在）
    if not os.path.exists(combined_img_path):
        os.makedirs(combined_img_path)

    # 获取 mutual 和 per 文件夹中的所有 .svg 文件
    mutual_files = [f for f in os.listdir(mutual_path) if f.lower().endswith('.svg')]
    per_files = [f for f in os.listdir(per_path) if f.lower().endswith('.svg')]

    # 提取文件名（不含扩展名）并找到两者的交集
    mutual_names = set(os.path.splitext(f)[0] for f in mutual_files)
    per_names = set(os.path.splitext(f)[0] for f in per_files)
    common_names = sorted(mutual_names.intersection(per_names))

    if not common_names:
        print("未找到具有相同名称的 .svg 文件。")
        return

    images_mutual = []
    images_per = []

    for name in common_names:
        mutual_file = os.path.join(mutual_path, name + '.svg')
        per_file = os.path.join(per_path, name + '.svg')

        try:
            # 将 .svg 转换为 .png
            mutual_png = cairosvg.svg2png(url=mutual_file)
            per_png = cairosvg.svg2png(url=per_file)

            # 使用 PIL 打开转换后的 .png 图像
            mutual_image = Image.open(io.BytesIO(mutual_png))
            per_image = Image.open(io.BytesIO(per_png))

            images_mutual.append(mutual_image)
            images_per.append(per_image)
        except Exception as e:
            print(f"转换文件 {name}.svg 失败: {e}")
            continue

    if not images_mutual or not images_per:
        print("没有可用的图像进行绘制。")
        return

    # 创建一个2行5列的子图
    fig, axes = plt.subplots(2, 5, figsize=(20, 8))

    for idx in range(len(common_names)):
        # 绘制 mutual 图像在第一行
        axes[0, idx].imshow(images_mutual[idx])
        axes[0, idx].axis('off')
        axes[0, idx].set_title(f"Mutual: {common_names[idx]}")

        # 绘制 per 图像在第二行
        axes[1, idx].imshow(images_per[idx])
        axes[1, idx].axis('off')
        axes[1, idx].set_title(f"Per: {common_names[idx]}")

    # 隐藏多余的子图（如果不足5组）
    total_pairs = len(common_names)
    for idx in range(total_pairs, 5):
        axes[0, idx].axis('off')
        axes[1, idx].axis('off')

    plt.tight_layout()

    # 保存合并后的图像
    combined_image_file = os.path.join(combined_img_path, 'combined_plot.png')
    plt.savefig(combined_image_file)
    plt.close()
    print(f"合并后的图像已保存到 {combined_image_file}")

if __name__ == '__main__':
    save_compare_img(pedestrian_pic_save_path)
