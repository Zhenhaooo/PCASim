import os
import gymnasium as gym
import numpy as np
import torch
from glob import glob
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
from collections import defaultdict, Counter
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from AdvCollisionEnv import AdvCollisionEnv

# ========== 配置 ==========
JSON_ROOT = "data/json_train"
SCENE_TYPES = ["brake", "go_straight", "turn", "lane_change"]
OSM_PATH = "data/map/DR_USA_Intersection_MA.osm"
MODEL_DIR = "adv_checkpoints"
MODEL_PREFIX = "ppo_adv_model"
TOTAL_EPISODES = 2000
MAX_STEPS = 500
N_CPUS = min(cpu_count(), 28)
DEVICE = "cpu"
LEARNING_RATE = 2.5e-4
RESET_INTERVAL = 500  # 每500个episode重置一次

# 加权采样比例
SCENE_WEIGHTS = {
    "brake": 1,
    "go_straight": 2,
    "turn": 2,
    "lane_change": 4
}

os.makedirs(MODEL_DIR, exist_ok=True)

## 重置PPO policy
def reset_policy_network(model, num_last_layers_to_reset=2):
    policy_net = model.policy.mlp_extractor
    actor_net = policy_net.policy_net
    critic_net = policy_net.value_net

    # helper函数
    def reset_layers(network, num_layers):
        layers = [module for module in network.modules() if isinstance(module, torch.nn.Linear)]
        for layer in layers[-num_layers:]:
            torch.nn.init.kaiming_normal_(layer.weight, nonlinearity='relu')
            if layer.bias is not None:
                torch.nn.init.constant_(layer.bias, 0)

    reset_layers(actor_net, num_last_layers_to_reset)
    reset_layers(critic_net, num_last_layers_to_reset)

    # 重新初始化 optimizer (注意：lr_schedule要从model上拿)
    model.policy.optimizer = model.policy.optimizer_class(
        model.policy.parameters(),
        lr=model.lr_schedule(1),
        **model.policy.optimizer_kwargs
    )

    print(f"🔄 已重置 policy 最后 {num_last_layers_to_reset} 层参数，并重建 optimizer。")


def load_trajectories(json_path):
    import json
    from test import process_raw_trajectory

    with open(json_path, 'r') as f:
        raw_data = json.load(f)
    for vid in raw_data:
        try:
            raw_data[vid]['trajectory'] = process_raw_trajectory(raw_data[vid]['trajectory'])
        except Exception:
            return None
    return raw_data

def rollout(json_path, model_path):
    try:
        env = DummyVecEnv([lambda: AdvCollisionEnv(json_path=json_path, osm_path=OSM_PATH, max_step=MAX_STEPS)])
        model = PPO.load(model_path, env=env, device=DEVICE)

        obs = env.reset()
        done = [False]
        total_reward = 0.0
        for _ in range(MAX_STEPS):
            action, _ = model.predict(obs, deterministic=False)
            obs, reward, done, _ = env.step(action)
            total_reward += reward[0]
            if done[0]:
                break
        return total_reward
    except Exception as e:
        print(f"错误处理 {json_path}: {e}")
        return None

def get_latest_checkpoint():
    checkpoints = glob(os.path.join(MODEL_DIR, f"{MODEL_PREFIX}_ep_*.zip"))
    if not checkpoints:
        return 0, None
    episodes = [int(os.path.basename(f).split("_ep_")[1].split(".zip")[0]) for f in checkpoints]
    last_ep = max(episodes)
    last_model_path = os.path.join(MODEL_DIR, f"{MODEL_PREFIX}_ep_{last_ep}.zip")
    return last_ep, last_model_path

def train_all():
    # Step 1: 按类型收集数据
    scene_dict = defaultdict(list)
    for scene in SCENE_TYPES:
        files = glob(os.path.join(JSON_ROOT, scene, "*.json"))
        scene_dict[scene].extend(files)
    assert any(scene_dict.values()), "❌ 没有找到 JSON 文件"

    # Step 2: 加载模型
    last_ep, last_model_path = get_latest_checkpoint()
    env = DummyVecEnv([lambda: AdvCollisionEnv(scene_dict["brake"][0], osm_path=OSM_PATH, max_step=MAX_STEPS)])
    model = PPO(
        "MlpPolicy", env,
        device=DEVICE,
        n_steps=2048,
        batch_size=256,
        n_epochs=1,
        learning_rate=LEARNING_RATE,
        verbose=1
    )
    if last_model_path:
        print(f"加载上次模型: {last_model_path}")
        model = PPO.load(last_model_path, env=env, device=DEVICE)

    for ep in range(last_ep + 1, TOTAL_EPISODES + 1):
        print(f"\nSampling for EP {ep}:")

        # Step 3: 加权采样
        sampled_jsons = []
        total_weight = sum(SCENE_WEIGHTS.values())
        for scene, weight in SCENE_WEIGHTS.items():
            count = int(N_CPUS * weight / total_weight)
            sampled_jsons.extend(np.random.choice(scene_dict[scene], size=count, replace=True))
        while len(sampled_jsons) < N_CPUS:
            sampled_jsons.append(np.random.choice(scene_dict["lane_change"]))

        # 打印采样分布
        print("本轮采样分布:", Counter(os.path.basename(os.path.dirname(f)) for f in sampled_jsons))

        # Step 4: Rollout & Learn
        model_path = os.path.join(MODEL_DIR, f"{MODEL_PREFIX}_temp.zip")
        model.save(model_path)

        with Pool(N_CPUS) as pool:
            rewards = list(tqdm(pool.starmap(rollout, [(f, model_path) for f in sampled_jsons]), total=N_CPUS))

        valid_rewards = [r for r in rewards if r is not None]
        if valid_rewards:
            mean_reward = np.mean(valid_rewards)
            print(f"EP {ep} 平均奖励: {mean_reward:.2f}")

            # 如果达到重置条件，重置 policy 网络结构
            if ep % RESET_INTERVAL == 0:
                print(f"🔧 EP {ep} 达到重置条件，重置 policy 网络结构。")
                reset_policy_network(model, num_last_layers_to_reset=2)

            model.learn(total_timesteps=MAX_STEPS * N_CPUS, reset_num_timesteps=True)
            model.save(os.path.join(MODEL_DIR, f"{MODEL_PREFIX}_ep_{ep}.zip"))
        else:
            print(f"EP {ep} 无有效样本，跳过训练")

if __name__ == "__main__":
    train_all()