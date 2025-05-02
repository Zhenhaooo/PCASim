import os
import gymnasium as gym
import numpy as np
import json
from gymnasium import spaces
from env.NGSIM_env.vehicle.kinematics import Vehicle as KinematicVehicle
from scripts.utils import OSMRoadNetworkLoader, process_raw_trajectory
from scripts.collision_utils import check_rotated_rectangle_collision

# ====== 可调常量 ======
NEAR_DIST        = 15.0          # 选取候选 adv 距离阈值
HEADING_THRESH   = np.pi / 4     # 轨迹方向相近阈值（暂未用）
MAX_STEER        = 0.35          # rad ≈20°
MAX_ACC          = 3.0           # m/s²
LANE_WIDTH_EST   = 3.5           # m，用来判断“同车道”
# ======================

class AdvCollisionEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self,
                 json_path: str,
                 osm_path,
                 road=None,
                 max_step: int = 500,
                 use_rule_based: bool = True):
        super().__init__()
        self.json_path = json_path
        self.max_step = max_step
        self.dt = 1 / 15.0
        self.ego_id = "ego"
        self.use_rule_based = use_rule_based

        # ---------- Road ----------
        if road is not None:
            self.road = road
        elif osm_path:
            loader = OSMRoadNetworkLoader(osm_path)
            loader.load_map()
            loader.create_road_network()
            self.road = loader.get_road()
        else:
            raise ValueError("必须提供 road 或 osm_path")

        # ---------- 数据 ----------
        self._load_data()
        self._parse_brake_window()        # 获取 brake_start/end 及 adv1_id

        # ---------- Gym 空间 ----------
        self.action_space = spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf,
                                            shape=(10,), dtype=np.float32)

    # ------------------------------------------------------------------
    # **********************   核心策略函数   **************************
    # ------------------------------------------------------------------
    def _rule_based_adv_control(self, adv, ego):
        # ---- 局部坐标 ----
        dx, dy = adv.position - ego.position
        cos_h, sin_h = np.cos(ego.heading), np.sin(ego.heading)
        lon = cos_h * dx + sin_h * dy          # ego 前为 +
        lat = -sin_h * dx + cos_h * dy         # ego 左为 +

        ego_speed = np.linalg.norm(ego.velocity)
        adv_speed = np.linalg.norm(adv.velocity)

        # ---- 默认指令 ----
        target_speed = adv_speed
        steer_cmd = 0.0

        # ---- 处理朝向差 (>90°) ----
        rel_angle = np.arctan2(dy, dx)         # ego → adv 的视线角
        heading_err = np.arctan2(np.sin(rel_angle - adv.heading),
                                 np.cos(rel_angle - adv.heading))

        if abs(heading_err) > np.pi / 2:
            # 背对 → 先掉头
            steer_cmd = np.clip(heading_err, -MAX_STEER, MAX_STEER)
            target_speed = max(0.0, adv_speed - 2.0)
        else:
            # ---- 正常位置决策 ----
            same_lane = abs(lat) < LANE_WIDTH_EST

            if same_lane:
                if lon > 0:                       # adv 在前 → 刹车
                    target_speed = max(0.0, ego_speed - 4.0)
                else:                             # adv 在后 → 追尾
                    target_speed = ego_speed + 4.0
            else:
                # 侧向并道压向 ego
                steer_cmd = np.clip(-lat / 12.0, -1.0, 1.0) * MAX_STEER
                if lon > 0:                       # 侧前方 → 轻刹
                    target_speed = max(0.0, ego_speed - 2.0)
                else:                             # 侧后方 → 加速插前
                    target_speed = ego_speed + 2.0

        # ---- P 控制纵向加速度 ----
        acc_cmd = np.clip((target_speed - adv_speed) * 0.8,
                          -MAX_ACC, MAX_ACC)
        return acc_cmd, steer_cmd

    # ------------------------------------------------------------------
    # **********************   数据与窗口解析   ************************
    # ------------------------------------------------------------------
    def _load_data(self):
        with open(self.json_path, "r") as f:
            self.raw_data = json.load(f)
        # trajectory 坐标转换
        for vid, info in self.raw_data.items():
            self.raw_data[vid]["trajectory"] = process_raw_trajectory(info["trajectory"])

    def _parse_brake_window(self):
        info = self.raw_data.get("ego", {}).get("scenario_info", {})
        folder_type = os.path.basename(os.path.dirname(self.json_path))

        if folder_type == "brake":
            binfo = info.get("brake_info", [])[0]
            self.adv1_id = binfo["brake_vehicle_id"]
            self.brake_start_t = int(binfo["brake_start"])
            self.brake_end_t = int(binfo["brake_end"])

        elif folder_type in {"go_straight", "turn"}:
            intersection_info = info.get("intersection", {})
            if intersection_info:
                adv1_id = list(intersection_info.keys())[0]
                inter = intersection_info[adv1_id]
                i_ego, i_adv = inter["index_cp"][0]
                first_ts = self.raw_data[self.ego_id]["trajectory"][0][4]
                self.brake_start_t = first_ts + (min(i_ego, i_adv) - 1) * 100
                self.brake_end_t = first_ts + (max(i_ego, i_adv) - 1) * 100
                self.adv1_id = adv1_id
            else:
                self._fallback_fixed_window()

        elif folder_type == "lane_change":
            lcinfo = info.get("laneChanging_info", {})
            start_idx = lcinfo["start_lane_change_index"]
            end_idx = lcinfo["end_lane_change_index"]
            first_ts = self.raw_data[self.ego_id]["trajectory"][0][4]
            self.brake_start_t = first_ts + (start_idx - 1) * 100
            self.brake_end_t = first_ts + (end_idx - 1) * 100
            self.adv1_id = lcinfo.get("last_lane_rear_v") or lcinfo.get("current_lane_front_v")
            if self.adv1_id is None:
                self._fallback_fixed_window()
        else:
            raise RuntimeError(f"❌ 不支持的场景类型: {folder_type}")

        self.brake_start_i = int(self.brake_start_t / 100)
        self.brake_end_i = int(self.brake_end_t / 100)
        self.scene_len = min(len(self.raw_data[self.ego_id]["trajectory"]), self.max_step)

    def _fallback_fixed_window(self):
        traj = self.raw_data[self.ego_id]["trajectory"]
        total_len = len(traj)
        first_ts = traj[0][4]
        self.brake_start_i = total_len // 3
        self.brake_end_i = total_len * 2 // 3
        self.brake_start_t = first_ts + (self.brake_start_i - 1) * 100
        self.brake_end_t = first_ts + (self.brake_end_i - 1) * 100
        # 选离 ego 最近的车为 adv1
        ego_pos = np.array(traj[self.brake_start_i][:2])
        min_dist, chosen = float("inf"), None
        for vid, data in self.raw_data.items():
            if vid == self.ego_id or self.brake_start_i >= len(data["trajectory"]):
                continue
            pos = np.array(data["trajectory"][self.brake_start_i][:2])
            dist = np.linalg.norm(pos - ego_pos)
            if dist < min_dist:
                min_dist, chosen = dist, vid
        self.adv1_id = chosen

    # ------------------------------------------------------------------
    # **********************   reset / step   **************************
    # ------------------------------------------------------------------
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.np_random, _ = gym.utils.seeding.np_random(seed)

        self.step_count = 0
        self.phase = 0                     # 0 / 1 / 2
        self.current_adv = None
        self.collision = False
        self.vehicles_removed = set()

        # -------- 车辆初始化 --------
        self.vehicles = {}
        for vid, data in self.raw_data.items():
            x, y, v, h, _ = data["trajectory"][0]
            veh = KinematicVehicle(self.road, np.array([x, y]), h, v)
            veh.LENGTH = data.get("length", 5.0)
            veh.WIDTH = data.get("width", 2.0)
            veh.vid = vid
            self.vehicles[vid] = veh

        self._maybe_pick_adv2()

        return self._get_obs(), {}

    def _pick_nearby_car(self):
        ego = self.vehicles[self.ego_id]
        best, best_dist = None, float("inf")
        for vid, data in self.raw_data.items():
            if vid in {self.ego_id, self.adv1_id} or vid in self.vehicles_removed:
                continue
            if self.step_count >= len(data["trajectory"]):
                continue
            pos = data["trajectory"][self.step_count][:2]
            dist = np.linalg.norm(pos - ego.position)
            # ---- 修正：需要同时满足两条件 ----
            if dist < best_dist and dist < NEAR_DIST:
                best, best_dist = vid, dist
        return best

    def _maybe_pick_adv2(self):
        self.current_adv = self._pick_nearby_car()

    def _maybe_pick_adv3(self):
        self.current_adv = self._pick_nearby_car()

    def _get_obs(self):
        ego = self.vehicles[self.ego_id]

        if self.current_adv and self.current_adv in self.vehicles:
            adv = self.vehicles[self.current_adv]
            rel_pos = adv.position - ego.position
            rel_vel = adv.velocity - ego.velocity
            adv_heading = adv.heading
        else:
            rel_pos = rel_vel = [0.0, 0.0]
            adv_heading = 0.0

        obs = [
            *rel_pos,                   # 2
            *rel_vel,                   # 2
            adv_heading,                # 1
            ego.heading,                # 1
            *ego.velocity,              # 2
            0.0, 0.0                    # 2 预留
        ]
        return np.array(obs, dtype=np.float32)

    # ------------------------------------------------------------
    # 核心 step
    # ------------------------------------------------------------
    def step(self, action):
        self.step_count += 1

        # ---------- 当前 adv 动作 ----------
        if self.current_adv and self.current_adv in self.vehicles:
            adv = self.vehicles[self.current_adv]
            ego = self.vehicles[self.ego_id]

            if self.use_rule_based:
                acc, steer = self._rule_based_adv_control(adv, ego)
            else:
                acc = float(np.clip(action[0], -1, 1) * MAX_ACC)
                steer = float(np.clip(action[1], -1, 1) * MAX_STEER)

            adv.action = {"acceleration": acc, "steering": steer}
            adv.step(self.dt)
        else:
            if self.current_adv and self.current_adv not in self.vehicles:
                print(f"⚠️ Warning: current_adv {self.current_adv} not found in vehicles, skipping adv control.")

        # ---------- 其余车辆：按轨迹重放 ----------
        for vid, data in self.raw_data.items():
            if vid in {self.current_adv, self.adv1_id}:
                if not (self.phase == 1 and vid == self.adv1_id):
                    continue
            if self.step_count >= len(data["trajectory"]):
                continue
            if vid not in self.vehicles:
                continue  # 车辆已经被删除了，跳过

            x, y, speed, yaw, _ = data["trajectory"][self.step_count]
            veh = self.vehicles[vid]
            veh.position = np.array([x, y])
            veh.heading = yaw
            veh.speed = speed

        # ---------- 阶段切换 ----------
        if self.phase == 0 and self.step_count >= self.brake_start_i:
            if self.current_adv and self.current_adv in self.vehicles:
                self.vehicles_removed.add(self.current_adv)
                self.vehicles.pop(self.current_adv, None)
            self.current_adv = self.adv1_id
            self.phase = 1

        elif self.phase == 1 and self.step_count >= self.brake_end_i:
            if self.adv1_id and self.adv1_id in self.vehicles:
                self.vehicles_removed.add(self.adv1_id)
                self.vehicles.pop(self.adv1_id, None)
            self.current_adv = None
            self.phase = 2
            self._maybe_pick_adv3()

        # ---------- 碰撞检测 ----------
        veh_list = list(self.vehicles.values())
        for i in range(len(veh_list)):
            v1 = veh_list[i]
            if np.allclose(v1.position, [0, 0]):
                continue
            for j in range(i + 1, len(veh_list)):
                v2 = veh_list[j]
                if np.allclose(v2.position, [0, 0]):
                    continue
                if check_rotated_rectangle_collision(v1, v2):
                    reward = 1.0 if self.ego_id in (v1.vid, v2.vid) else -1.0
                    return self._get_obs(), reward, True, False, {}

        # ---------- 超时 ----------
        if self.step_count >= self.scene_len:
            return self._get_obs(), -1.0, False, True, {}

        return self._get_obs(), 0.0, False, False, {}