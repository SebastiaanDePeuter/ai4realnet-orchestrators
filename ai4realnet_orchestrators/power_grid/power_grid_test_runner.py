"""
Power Grid Test Runner - Base Classes and Common Functions
==========================================================

This module provides the base TestRunner for Power Grid KPIs and common
utility functions used by specialized KPI runners.

Base Classes:
    - PowerGridTestRunner: Template for all Power Grid KPIs

Authors: AI4REALNET Consortium
"""
import json
from abc import abstractmethod
import logging
import os
import sys
import tempfile
import zipfile
from typing import Dict, List

import requests
import grid2op
from lightsim2grid import LightSimBackend

from ai4realnet_orchestrators.test_runner import TestRunner

logger = logging.getLogger(__name__)

# ============================================================================
# Path Setup for Framework
# ============================================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
FRAMEWORK_PATH = os.path.join(SCRIPT_DIR, "framework")
CONFIG_PATH = os.path.join(SCRIPT_DIR, "configuration")

# Add framework to path for imports
if FRAMEWORK_PATH not in sys.path:
    sys.path.insert(0, FRAMEWORK_PATH)


# ============================================================================
# Base Class: PowerGridTestRunner (TEMPLATE - DO NOT MODIFY)
# ============================================================================

class PowerGridTestRunner(TestRunner):
    """
    Base TestRunner for Power Grid KPIs.

    Provides common functionality:
    - Agent loading (CurriculumAgent, RandomAgent)
    - Environment creation with LightSimBackend
    - Submission data handling

    Subclasses must implement: getResult(env, agent) -> dict
    """

    # To be overridden by specialized subclasses
    KPI_MAPPING: Dict = {}

    def __init__(self, test_id: str, scenario_ids: List[str], benchmark_id: str, use_weekly_subset: bool = False):
        """Initialize with KPI-specific configuration."""
        super().__init__(test_id=test_id, scenario_ids=scenario_ids, benchmark_id=benchmark_id)

        # Subset selection
        self.use_weekly_subset = use_weekly_subset
        self.selected_scenario_names = None

        # Get KPI info from mapping (subclasses should override KPI_MAPPING)
        self.kpi_info = self.KPI_MAPPING.get(test_id, {
            "name": "Unknown KPI",
            "metric_key": None,
            "description": ""
        })

        logger.info(
            f"Initialized {self.__class__.__name__}\n"
            f"  Test ID: {test_id}\n"
            f"  KPI: {self.kpi_info['name']}"
        )

    def init(self, submission_data_url: str, submission_id: str):
        """Initialize and load submission data from JSON URL."""
        super().init(submission_data_url=submission_data_url, submission_id=submission_id)
        self.submission_data = PowerGridTestRunner.load_submission_data(submission_data_url)

    def run_scenario(self, scenario_id: str, submission_id: str):
        """Run evaluation for a specific scenario."""
        if scenario_id not in self.submission_data["scenarios"]:
            raise ValueError(f"Unrecognized scenario ID: '{scenario_id}'")

        default_config = self.submission_data["default_config"]
        specific_config = self.submission_data["specific_config"].get(scenario_id, {})
        # Merge default and specific configs (specific overrides default)
        scenario_data = {**default_config, **specific_config}

        # Load mapping towards scenario and agent paths
        with open(os.path.join(CONFIG_PATH, "path-mapping.json"), "r") as f:
            mapping = json.load(f)

        # Create environment with fast backend
        scenario_name = scenario_data["scenario_name"]
        scenario_path = mapping["scenario_path"][scenario_name]
        env = grid2op.make(scenario_path, backend=LightSimBackend())

        # Create shift environment when provided
        # TODO: update scenario_shift_path in path-mapping.json, it's the same as scenario_path for now
        scenario_shift_name = scenario_data.get("scenario_shift_name")
        if scenario_shift_name:
            scenario_shift_path = mapping["scenario_path"][scenario_shift_name]
            env_shift = grid2op.make(scenario_shift_path, backend=LightSimBackend())
        else:
            env_shift = None

        # Create and load agent
        agent_type = scenario_data["agent_type"]
        agent_path = mapping["agent_path"][agent_type]
        agent = self.load_agent(agent_type, agent_path, env)

        # Select subset if requested
        if self.use_weekly_subset:
            chronics_path = os.path.join(env.get_path_env(), "chronics")
            self._select_weekly_scenarios(chronics_path, seed=9542)
            logger.info(f"Selected {len(self.selected_scenario_names)} scenarios for weekly subset: {self.selected_scenario_names}")

            # Filter environment chronics
            self._filter_env_chronics(env)
            if env_shift:
                self._filter_env_chronics(env_shift)

        return self.getResult(env, env_shift, agent)

    def _filter_env_chronics(self, env):
        """Filter the environment's chronics to only include the selected subset."""
        import numpy as np
        if hasattr(env, "chronics_handler") and hasattr(env.chronics_handler, "real_data"):
            real_data = env.chronics_handler.real_data
            if hasattr(real_data, "subpaths"):
                real_data.subpaths = np.array([
                    p for p in real_data.subpaths
                    if os.path.basename(p) in self.selected_scenario_names
                ])
                # Reset the chronics handler to start from the first scenario
                if hasattr(real_data, "reset"):
                    real_data.reset()
                logger.info(f"Environment chronics filtered to {len(real_data.subpaths)} scenarios and reset")

    def _select_weekly_scenarios(self, chronics_path: str, seed: int):
        """Select one random scenario per week."""
        import random
        all_dirs = sorted([d for d in os.listdir(chronics_path)
                          if os.path.isdir(os.path.join(chronics_path, d)) and d != "chronic_example"])

        weeks = {}
        for name in all_dirs:
            week = name.split('_')[0] if '_' in name else name
            if week not in weeks:
                weeks[week] = []
            weeks[week].append(name)

        rng = random.Random(seed)
        selected_names = []
        for week in sorted(weeks.keys()):
            selected_names.append(rng.choice(weeks[week]))

        self.selected_scenario_names = sorted(selected_names)

    @staticmethod
    def load_submission_data(submission_data_url: str) -> dict:
        """Load submission metadata from URL."""
        response = requests.get(submission_data_url)
        return response.json()

    @staticmethod
    def load_agent(agent_type: str, agent_path: str | None, env):
        """
        Load agent based on type.

        Supported types:
        - RandomAgent: Grid2Op random agent
        - CurriculumAgent: Trained curriculum learning agent
        - ExpertAgent: Trained expert learning agent
        """
        if agent_type == 'RandomAgent':
            from grid2op.Agent import RandomAgent
            agent = RandomAgent(env.action_space)

        elif agent_type == 'CurriculumAgent':
            from ai4realnet_orchestrators.power_grid.framework.modified_curriculum_classes.baseline import CurriculumAgent
            agent = CurriculumAgent(env.action_space, env.observation_space, 'curriculum_agent')

            if agent_path is not None:
                # Extract agent zip locally
                temp_dir = tempfile.mkdtemp()
                with zipfile.ZipFile(agent_path, 'r') as zip_ref:
                    zip_ref.extractall(temp_dir)

                # Load model and actions
                model_path = os.path.join(temp_dir, 'model')
                actions_path = os.path.join(temp_dir, 'actions')
                agent.load(model_path, actions_path, best_action_threshold=0.95)
                
        elif agent_type == "ExpertAgent":
            from ExpertAgent.utils.helper_functions import make_gymenv
            from ExpertAgent.ExpertAgent import ExpertAgentRL
            from stable_baselines3.ppo import MlpPolicy
            
            env_gym = make_gymenv(env, obs_attr_to_keep=["rho"], action_space_path="read_from_file", act_to_keep=("set_bus",))
            model_path = agent_path
            nn_kwargs = {
                "policy": MlpPolicy,
                "env": env_gym,
                "verbose": True,
                "learning_rate": 1e-3,
                "tensorboard_log": model_path,
                "policy_kwargs": {"net_arch": [800, 1000, 1000, 800]},
                "device": "auto"
            }
            agent = ExpertAgentRL(name="PPO_SB3",
                                  env=env,
                                  action_space=env.action_space,
                                  gymenv=env_gym,
                                  gym_act_space=env_gym.action_space,
                                  gym_obs_space=env_gym.observation_space,
                                  nn_kwargs=nn_kwargs
                                  )
            agent.load(model_path)
        else:
            raise SyntaxError(f'Unsupported agent type: {agent_type}')

        return agent

    @abstractmethod
    def _compute_all_metrics(self, env, env_shift, agent) -> Dict:
        """
        Compute all metrics for this category of KPIs.

        Must be implemented by generic subclasses (Operational, Reliability, etc.).

        Args:
            env: Grid2Op environment
            env_shift: Grid2Op shift environment (if applicable)
            agent: Loaded agent

        Returns:
            dict containing all computed metrics
        """
        pass

    def getResult(self, env, env_shift, agent) -> dict:
        """
        Compute and return KPI results using caching.

        This generic implementation handles caching results by submission_id
        and extracting the specific metric for the current KPI.

        Args:
            env: Grid2Op environment
            env_shift: Grid2Op shift environment
            agent: Loaded agent

        Returns:
            dict with "primary" key containing the KPI value
        """
        # Use submission_id for caching
        cache_key = getattr(self, 'submission_id', 'default')

        # Get KPI info (set by subclasses)
        kpi_info = getattr(self, 'kpi_info', {
            "name": "Unknown KPI",
            "metric_key": None,
            "description": ""
        })

        logger.info(
            f"Running evaluation for {kpi_info['name']}\n"
            f"  Cache key: {cache_key}"
        )

        # Check cache - avoid re-running expensive evaluation
        # Subclasses should define their own _metrics_cache class attribute
        cache = getattr(self, '_metrics_cache', None)
        if cache is None:
            # Fallback to instance cache if class cache not defined
            if not hasattr(self, '_instance_metrics_cache'):
                self._instance_metrics_cache = {}
            cache = self._instance_metrics_cache

        if cache_key not in cache:
            logger.info(f"Cache miss - running complete evaluation in {self.__class__.__name__}")
            all_metrics = self._compute_all_metrics(env, env_shift, agent)
            cache[cache_key] = all_metrics
            logger.info(f"Cached results for {cache_key}")
        else:
            logger.info(f"Cache hit - using existing results for {cache_key}")

        all_metrics = cache[cache_key]

        # Extract KPI-specific value
        metric_key = kpi_info.get('metric_key')
        if metric_key is None:
            logger.error(f"No metric_key defined for test_id {getattr(self, 'test_id', 'unknown')}")
            return {"primary": 0.0}

        kpi_value = all_metrics.get(metric_key, 0.0)

        logger.info(
            f"KPI Result: {kpi_info['name']} = {kpi_value}\n"
            f"  Description: {kpi_info['description']}"
        )

        return {"primary": float(kpi_value)}


