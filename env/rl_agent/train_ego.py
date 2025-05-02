import os, random, json, gymnasium as gym
import numpy as np
from glob import glob
from collections import defaultdict, Counter
from multiprocessing import Pool, cpu_count
from tqdm import tqdm

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from AdvCollisionEnv import AdvCollisionEnv

# ==========================================================
JSON_ROOT   = "data/json_train"
OSM_PATH    = "data/map/DR_USA_Intersection_MA.osm"
ADV_MODEL   = "adv_checkpoints/ppo_adv_model_ep_213.zip"
MODEL_DIR   = "ego_checkpoints"
MODEL_PREF  = "ppo_ego_model"
TOTAL_EPIS  = 2000
MAX_STEPS   = 500
N_CPUS      = min(cpu_count(), 28)
DEVICE      = "cpu"
LEARN_RATE  = 3e-4

SCENE_TYPES   = ["brake", "go_straight", "turn", "lane_change"]
SCENE_WEIGHTS = {"brake":1, "go_straight":2, "turn":2, "lane_change":4}

os.makedirs(MODEL_DIR, exist_ok=True)
class EgoViaAdvWrapper(gym.Wrapper):
    def __init__(self, adv_env: AdvCollisionEnv, adv_model: PPO):
        super().__init__(adv_env)
        self.adv_env  = adv_env
        self.adv_model= adv_model

        # 外部只需要管 ego → action space = ego 的 (lon_acc, lat_steer)
        self.action_space      = gym.spaces.Box(-1, 1, (2,), np.float32)
        # 观测直接复用 adv_env 的
        self.observation_space = adv_env.observation_space

    def reset(self, **kwargs):
        obs, info = self.adv_env.reset(**kwargs)
        return obs, info

    def step(self, ego_action):
        # --- ① adv_action ---
        adv_obs = self.adv_env._get_obs()
        adv_action, _ = self.adv_model.predict(adv_obs, deterministic=True)

        # --- ② ego 行为（直接修改 ego 车辆状态；此处做一个最简的纵向加速度控制） ---
        ego = self.adv_env.vehicles[self.adv_env.ego_id]
        acc_cmd  = float(np.clip(ego_action[0], -1, 1) * 3.0)
        steer_cmd= float(np.clip(ego_action[1], -1, 1) * 0.35)
        ego.action = {"acceleration": acc_cmd, "steering": steer_cmd}

        # --- ③ env.step 仍需 adv_action 作为 current_adv 的动作 ---
        obs, rew, term, trunc, info = self.adv_env.step(adv_action)
        # 把 reward 调成“ego 视角”：撞车=-1，存活=+0.001
        if term and rew == 1.0:          # 说明 ego 被撞
            rew = -1.0
        else:
            rew = 0.001
        return obs, rew, term, trunc, info

def make_env(json_path):
    base = AdvCollisionEnv(json_path=json_path,
                           osm_path=OSM_PATH,
                           max_step=MAX_STEPS,
                           use_rule_based=False)        # adv_env 本身让 current_adv 接 PPO
    adv_model = PPO.load(ADV_MODEL, env=base, device=DEVICE)
    return EgoViaAdvWrapper(base, adv_model)

def rollout(json_path, model_path):
    try:
        env   = DummyVecEnv([lambda: make_env(json_path)])
        model = PPO.load(model_path, env=env, device=DEVICE)

        obs = env.reset()
        total_reward, done = 0.0, [False]
        for _ in range(MAX_STEPS):
            act, _ = model.predict(obs, deterministic=False)
            obs, rew, done, _ = env.step(act)
            total_reward += rew[0]
            if done[0]:
                break
        return total_reward
    except Exception as e:
        print(f"❌ {json_path}: {e}")
        return None

def weighted_sample(scene_dict, n):
    total_w = sum(SCENE_WEIGHTS.values())
    keys, probs = zip(*[(k, w/total_w) for k,w in SCENE_WEIGHTS.items()])
    return [np.random.choice(scene_dict[random.choices(keys, probs)[0]]) for _ in range(n)]

def train_all():
    # 收集 json
    scene_dict = defaultdict(list)
    for sc in SCENE_TYPES:
        scene_dict[sc].extend(glob(os.path.join(JSON_ROOT, sc, "*.json")))
    assert any(scene_dict.values()), "❌ 无 JSON 数据"

    dummy_env = DummyVecEnv([lambda: make_env(scene_dict["brake"][0])])
    model = PPO("MlpPolicy", dummy_env, device=DEVICE,
                batch_size=256, n_steps=2048, n_epochs=1,
                learning_rate=LEARN_RATE, verbose=1)

    for ep in range(1, TOTAL_EPIS+1):
        print(f"\n🌀 Episode-{ep} 采样 ...")
        json_batch = weighted_sample(scene_dict, N_CPUS)

        tmp_path = os.path.join(MODEL_DIR, f"{MODEL_PREF}_tmp.zip")
        model.save(tmp_path)

        with Pool(N_CPUS) as pool:
            rewards = list(tqdm(pool.starmap(rollout, [(p, tmp_path) for p in json_batch]),
                                total=N_CPUS))

        valid = [r for r in rewards if r is not None]
        print(f"   🎯 平均评估奖励: {np.mean(valid):.3f}" if valid else "⚠️ 无有效回合")

        model.learn(total_timesteps=MAX_STEPS * N_CPUS,
                    reset_num_timesteps=True)

        model.save(os.path.join(MODEL_DIR, f"{MODEL_PREF}_ep_{ep}.zip"))

if __name__ == "__main__":
    train_all()
