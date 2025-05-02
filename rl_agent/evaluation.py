import os
from glob import glob
from tqdm import tqdm
from stable_baselines3 import PPO
from AdvCollisionEnv import AdvCollisionEnv
import numpy as np

SCENE_ROOT = "json_test"
SCENE_TYPES = ["brake", "go_straight", "turn", "lane_change"]
OSM_PATH = "Data/map/DR_USA_Intersection_MA.osm"
MAX_STEPS = 500
ADV_MODEL_PATH = "adv_checkpoints/ppo_adv_model_ep_213"
EGO_MODEL_PATH = "ego_checkpoints/ppo_ego_model_ep_119"

def replay_with_dual_control(adv_model, ego_model, json_path):
    env = AdvCollisionEnv(json_path=json_path, osm_path=OSM_PATH, max_step=MAX_STEPS)
    obs, _ = env.reset()
    done = False
    for _ in range(env.scene_len):
        # adv 由 PPO 控制
        action_adv, _ = adv_model.predict(obs, deterministic=True)

        # 在对抗阶段让 ego 也由 PPO 控制
        if env.phase in {1, 2} and env.ego_id in env.vehicles:
            ego = env.vehicles[env.ego_id]
            # 构造与训练一致的 ego 观测 (8维)
            pos = ego.position
            vel = ego.velocity
            heading = ego.heading
            rel_goal = [0.0, 0.0]
            ego_obs = [*pos, *vel, heading, *rel_goal, 0.0]  # 添加占位维度
            ego_obs = np.array(ego_obs, dtype=np.float32)

            ego_action, _ = ego_model.predict(ego_obs, deterministic=True)
            ego.action = {
                "acceleration": float(np.clip(ego_action[0], -1, 1) * 3.0),
                "steering": float(np.clip(ego_action[1], -1, 1) * 0.3)
            }
            ego.step(env.dt)

        obs, _, terminated, truncated, _ = env.step(action_adv)
        if terminated or truncated:
            return terminated, truncated
    return False, True  # 未中断视为 timeout

def main():
    adv_model = PPO.load(ADV_MODEL_PATH)
    ego_model = PPO.load(EGO_MODEL_PATH)
    summary = {}

    for scene_type in SCENE_TYPES:
        folder = os.path.join(SCENE_ROOT, scene_type)
        json_files = sorted(glob(os.path.join(folder, "*.json")))
        col, timeout = 0, 0
        print(f"\n🔍 测试类型: {scene_type} ({len(json_files)} 个样本)")
        for f in tqdm(json_files):
            try:
                terminated, truncated = replay_with_dual_control(adv_model, ego_model, f)
                if terminated:
                    col += 1
                elif truncated:
                    timeout += 1
            except Exception as e:
                print(f"⚠️ 跳过 {f}: {e}")
                timeout += 1

        total = col + timeout
        collision_rate = (col / total) if total else 0
        summary[scene_type] = (col, timeout, collision_rate)

    print("\n📊 测试结果汇总：")
    for stype, (c, t, rate) in summary.items():
        print(f"{stype:<15} 碰撞: {c:<3} 超时: {t:<3} 碰撞率: {rate:.2%}")

if __name__ == "__main__":
    main()
