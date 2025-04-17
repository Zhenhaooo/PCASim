import os
import sys

from gymnasium.envs.registration import register


def _register_highway_envs():
    """Import the envs module so that envs register themselves."""

    # from .common.abstract import MultiAgentWrapper

    # highway_env.py
    register(
        id="intersection-adv-v0",
        entry_point="simulation_environment.sumo_envs.intersection_env:AdvScenarioEnv",
    )

    register(
        id="intersection-adv-v1",
        entry_point="simulation_environment.sumo_envs.intersection_train_env:IntersectionTrainEnv",
    )

    register(
        id="driving-side-by-side-v0",
        entry_point="simulation_environment.sumo_envs.driving_side_by_side_env:DrivingSideBySideEnv",
    )

    register(
        id="cutin-brake-cutout-v0",
        entry_point="simulation_environment.sumo_envs.cutin_brake_cutout_env:CutInBrakeCutOutEnv",
    )

    register(
        id="emergency-cut-in-v0",
        entry_point="simulation_environment.sumo_envs.emergency_cut_in_env:EmergencyCutInEnv",
    )

    register(
        id="interaction-v0",
        entry_point="simulation_environment.osm_envs.interaction_env:InterActionEnv",
    )


_register_highway_envs()