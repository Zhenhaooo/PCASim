import random
import sys
import os
import time
project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(project_dir)
import simulation_environment
import gymnasium as gym
from gymnasium.wrappers import RecordVideo
if __name__ == '__main__':
    data_path = '/home/chuan/work/TypicalScenarioExtraction/data/pedestrian_data/trajectory_set/vehicle_tracks_016_trajectory_set_P2.json'
    osm_path = '/home/chuan/work/TypicalScenarioExtraction/data/datasets/Interaction/maps/DR_USA_Intersection_MA.osm'
    env = gym.make(
        'interaction-v0',
        render_mode="rgb_array",
        data_path=data_path,
        osm_path=osm_path,
        )
    env = RecordVideo(
        env, video_folder="./videos", episode_trigger=lambda e: True,
        name_prefix="vehicle_tracks_016_trajectory_set_P2",
    )
    env.unwrapped.set_record_video_wrapper(env)

    obs, info = env.reset()
    done = False
    while not done:
        action = random.choice(range(5))
        obs, reward, terminated, truncated, info = env.step(None)
        env.render()
        done = terminated or truncated
    env.close()