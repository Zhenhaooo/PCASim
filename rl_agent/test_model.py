import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""  # ✅ 禁用 GPU
from glob import glob
from multiprocessing import Pool, cpu_count
from stable_baselines3 import PPO
from AdvCollisionEnv import AdvCollisionEnv
from tqdm import tqdm
import torch

# 配置
SCENE_ROOT = "data/json_test"
SCENE_TYPES = ["brake", "go_straight", "turn", "lane_change"]
OSM_PATH = "Data/map/DR_USA_Intersection_MA.osm"
MODEL_PATH = "adv_checkpoints/ppo_adv_model_ep_213"
MAX_STEPS = 500
N_CPUS = min(cpu_count(), 32)

def replay_and_count(json_path):
    # 限制线程
    torch.set_num_threads(1)

    # 每个进程单独加载模型（重要！）
    model = PPO.load(MODEL_PATH, device="cpu")
    env = AdvCollisionEnv(json_path=json_path, osm_path=OSM_PATH, max_step=MAX_STEPS)
    obs, _ = env.reset()
    for _ in range(env.scene_len):
        action, _ = model.predict(obs, deterministic=True)
        obs, _, terminated, truncated, _ = env.step(action)
        if terminated:
            return json_path, True, False
        if truncated:
            return json_path, False, True
    return json_path, False, True

def main():
    summary = {}
    for scene_type in SCENE_TYPES:
        folder = os.path.join(SCENE_ROOT, scene_type)
        json_files = sorted(glob(os.path.join(folder, "*.json")))
        print(f"\n🔍 测试类型: {scene_type} ({len(json_files)} 个样本)")

        with Pool(N_CPUS) as pool:
            results = list(tqdm(pool.imap(replay_and_count, json_files), total=len(json_files)))

        col, timeout = 0, 0
        for _, terminated, truncated in results:
            if terminated:
                col += 1
            elif truncated:
                timeout += 1

        total = col + timeout
        collision_rate = (col / total) if total else 0
        summary[scene_type] = (col, timeout, collision_rate)

    print("\n📊 测试结果汇总：")
    for stype, (c, t, rate) in summary.items():
        print(f"{stype:<15} 碰撞: {c:<3} 超时: {t:<3} 碰撞率: {rate:.2%}")

if __name__ == "__main__":
    main()
