import os
import re
import cv2
import pygame
import numpy as np
from threading import Thread
from env.NGSIM_env.road.graphics import RoadGraphics, WorldSurface
from env.NGSIM_env.vehicle.graphics import VehicleGraphics

def draw(road, frame_save_name="rendered_frames/scene_frame.png", cam_center=None):
    """
    渲染当前路网和车辆，保存为图像。

    参数:
    - road: Road 对象，包含车辆信息
    - frame_save_name: 保存图片的文件路径
    - cam_center: 可选，设定摄像机中心视角 (x, y)
    """
    pygame.init()
    screen_size = (1280, 720)
    pygame.display.set_mode(screen_size)

    raw_surface = pygame.Surface(screen_size)
    surface = WorldSurface(screen_size, 0, raw_surface)

    if cam_center is not None:
        surface.move_display_window_to(np.array(cam_center))

    RoadGraphics.display(road, surface)

    font = pygame.font.SysFont("Arial", 16)

    for veh in road.vehicles:
        VehicleGraphics.display(veh, surface)
        pos = veh.position
        screen_pos = surface.world_to_pixel(pos)
        vid = getattr(veh, 'vid', '?')
        text_surface = font.render(str(vid), True, (255, 255, 0))  # 黄色字体
        surface.blit(text_surface, screen_pos)

    pygame.image.save(surface, frame_save_name)


def save_videos(frame_folder="rendered_frames", save_video_path="exported_videos", video_save_name="scenario_demo"):
    """
    将指定文件夹中的帧图像合成为 mp4 视频。

    参数:
    - frame_folder: 包含帧图像的文件夹路径
    - save_video_path: 最终生成视频保存的目录
    - video_save_name: 视频文件名（不带扩展名）
    """
    if not os.path.exists(frame_folder):
        print(f"❌ 图像文件夹不存在: {frame_folder}")
        return

    def frame_key(fname):
        m = re.match(r"frame_(\d+)\.png$", os.path.basename(fname))
        return int(m.group(1)) if m else float('inf')

    img_files = sorted([
        os.path.join(frame_folder, f)
        for f in os.listdir(frame_folder)
        if f.endswith(".png") and f.startswith("frame_")
    ], key=frame_key)

    if not img_files:
        print("❌ 图像文件夹为空，无法生成视频")
        return

    imgs = [cv2.imread(f) for f in img_files if cv2.imread(f) is not None]
    if not imgs:
        print("❌ 没有图像可保存为视频")
        return

    os.makedirs(save_video_path, exist_ok=True)
    video_path = f"{save_video_path}/{video_save_name}.mp4"
    print(f"🎬 正在保存视频到 {video_path}")

    def img_2_video(video_path, imgs):
        height, width, _ = imgs[0].shape
        out = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'mp4v'), 30, (width, height))
        for img in imgs:
            out.write(img)
        out.release()
        print(f"✅ 视频保存成功: {video_path}")

    t = Thread(target=img_2_video, args=(video_path, imgs))
    t.setDaemon(True)
    t.start()
    t.join()
