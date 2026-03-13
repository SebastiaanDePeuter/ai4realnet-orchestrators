"""
Multi-Attacker Robustness & Resilience TestRunner for Power Grid Domain

This module implements a SINGLE TestRunner that evaluates defender agents against
multiple adversarial attack types and returns different KPI values based on test_id.

KPIs Implemented:
    - KPI-DF-069: Drop-off in reward
    - KPI-FF-070: Frequency changed output AI agent
    - KPI-SF-071: Severity of changed output AI agent
    - KPI-SF-072: Steps survived with perturbations
    - KPI-VF-073: Vulnerability to perturbation
    - KPI-AF-074: Area between reward curves
    - KPI-DF-075: Degradation time
    - KPI-RF-076: Restorative time
    - KPI-SF-077: Similarity state to unperturbed situation

Framework Path: Relative to this file at ./framework/

Design: Single evaluation runs ALL attackers once, computes ALL metrics, 
then returns the appropriate metric based on which KPI is being evaluated.

Author: INESC TEC
"""

import logging
import os
import sys
import pickle
import tempfile
import zipfile
import requests
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

# Add framework to path BEFORE other imports
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
FRAMEWORK_PATH = os.path.join(SCRIPT_DIR, "framework")
if FRAMEWORK_PATH not in sys.path:
    sys.path.insert(0, FRAMEWORK_PATH)

# Parent directory (ONE level up) - where test_runner.py lives
_parent_dir = os.path.dirname(SCRIPT_DIR)  
if _parent_dir not in sys.path:
    sys.path.insert(0, _parent_dir)

from test_runner import TestRunner

logger = logging.getLogger(__name__)

# ============================================================================
# KPI ID Mapping
# ============================================================================

KPI_MAPPING = {
    # Robustness KPIs (Benchmark: 3810191b-8cfd-4b03-86b2-f7e530aab30d)
    "b8a9a411-7cfe-4c1d-b9a6-eef1c0efe920": {
        "name": "KPI-VF-073: Vulnerability to perturbation",
        "metric_key": "perturb_vulnerability",
        "description": "Proportion of features vulnerable to attack [0-1]"
    },
    "a121d8bd-1943-41ba-b3a7-472a0154f8f9": {
        "name": "KPI-SF-072: Steps survived with perturbations",
        "metric_key": "n_steps_survived",
        "description": "Number of timesteps before failure"
    },
    "3d033ec6-942a-4b03-b26e-f8152ba48022": {
        "name": "KPI-SF-071: Severity of changed output",
        "metric_key": "severity_of_change",
        "description": "Severity of action changes [0-1, higher=worse]"
    },
    "1cbb7783-47b4-4289-9abf-27939da69a2f": {
        "name": "KPI-DF-069: Drop-off in reward",
        "metric_key": "reward_drop_percent",
        "description": "Percentage decrease in reward [0-100]"
    },
    "acaf712a-c06c-4a04-a00f-0e7feeefb60c": {
        "name": "KPI-FF-070: Frequency changed output",
        "metric_key": "action_change_freq",
        "description": "Proportion of timesteps with changed actions [0-1]"
    },
    
    # Resilience KPIs (Benchmark: 31ea606b-681a-437a-85b9-7c81d4ccc287)
    "534f5a1f-7115-48a5-b58c-4deb044d425d": {
        "name": "KPI-AF-074: Area between reward curves",
        "metric_key": "area_between_curves",
        "description": "Integrated performance degradation"
    },
    "04a23bfc-fc44-4ec4-a732-c29214130a83": {
        "name": "KPI-DF-075: Degradation time",
        "metric_key": "degradation_time",
        "description": "Time until performance degrades"
    },
    "225aaee8-7c7f-4faf-810b-407b551e9f2a": {
        "name": "KPI-RF-076: Restorative time",
        "metric_key": "restoration_time",
        "description": "Time to restore performance"
    },
    "7fe4210f-1253-411c-ba03-49d8b37c71fa": {
        "name": "KPI-SF-077: Similarity to unperturbed state",
        "metric_key": "state_similarity",
        "description": "Cosine similarity to unperturbed states [-1 to 1]"
    },
}


class MultiAttackerRobustnessTestRunner(TestRunner):
    """
    Single TestRunner that handles ALL 9 robustness/resilience KPIs.
    
    This TestRunner:
    1. Runs evaluation ONCE against all attackers
    2. Computes ALL metrics
    3. Returns the appropriate metric based on test_id
    
    Advantages:
    - Efficient: Only one evaluation per submission
    - Maintainable: Single codebase for all KPIs
    - Cacheable: Results cached across KPI requests
    """
    
    # Evaluation configuration
    ATTACKER_TYPES = ["GEPerturb", "LambdaPIR", "Random", "PPO", "SAC_10", "SAC_5", "RLPerturb"]
    NUM_EPISODES = 50
    ENV_NAME = os.path.join(FRAMEWORK_PATH, "environments", "env_icaps")
    
    def __init__(self, test_id: str, scenario_ids: List[str], benchmark_id: str):
        """
        Initialize the TestRunner.
        
        Args:
            test_id: UUID for the specific KPI being evaluated
            scenario_ids: List of scenario UUIDs to evaluate
            benchmark_id: UUID for the benchmark suite
        """
        super().__init__(test_id=test_id, benchmark_id=benchmark_id)
        self.scenario_ids = scenario_ids
        
        # Validate test_id is one of our 9 KPIs
        if test_id not in KPI_MAPPING:
            raise ValueError(
                f"Unknown test_id: {test_id}. "
                f"Expected one of: {list(KPI_MAPPING.keys())}"
            )
        
        self.kpi_info = KPI_MAPPING[test_id]
        
        # Metrics cache (key: f"{scenario_id}_{submission_id}")
        self._metrics_cache = {}
        
        # Defender agent (loaded in init())
        self._defender_agent = None
        
        # Framework initialization flag
        self._framework_initialized = False
        
        logger.info(
            f"Initialized MultiAttackerRobustnessTestRunner\n"
            f"  Test ID: {test_id}\n"
            f"  KPI: {self.kpi_info['name']}\n"
            f"  Metric: {self.kpi_info['metric_key']}\n"
            f"  Scenarios: {scenario_ids}"
        )
    
    def _initialize_framework(self):
        """Add framework to Python path and validate it exists."""
        if self._framework_initialized:
            return
            
        if not os.path.exists(self.FRAMEWORK_PATH):
            raise FileNotFoundError(
                f"Framework not found at {self.FRAMEWORK_PATH}\n"
                f"Please ensure the framework folder exists in the power_grid directory."
            )
        
        # Add framework to path if not already there
        if self.FRAMEWORK_PATH not in sys.path:
            sys.path.insert(0, self.FRAMEWORK_PATH)
        
        logger.info(f"Framework initialized from: {self.FRAMEWORK_PATH}")
        
        self._framework_initialized = True
    
    def init(self, submission_data_url: str):
        """
        Initialize and load defender agent from submission.
        
        Args:
            submission_data_url: URL to download the submitted defender agent
        """
        super().init(submission_data_url=submission_data_url)
        
        logger.info(f"Loading defender agent from: {submission_data_url}")
        
        # Initialize framework
        self._initialize_framework()
        
        # Determine submission format and load agent
        try:
            if submission_data_url.endswith('.pkl'):
                self._defender_agent = self._load_agent_from_pickle(submission_data_url)
            elif submission_data_url.endswith('.zip'):
                self._defender_agent = self._load_agent_from_zip(submission_data_url)
            elif 'docker' in submission_data_url.lower():
                self._defender_agent = self._load_agent_from_docker(submission_data_url)
            else:
                # Default: try pickle
                self._defender_agent = self._load_agent_from_pickle(submission_data_url)
            
            logger.info("Defender agent loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load defender agent: {e}")
            raise
    
    def run_scenario(self, scenario_id: str, submission_id: str) -> Dict:
        """
        Run evaluation for a specific scenario and return the KPI value.
        
        Args:
            scenario_id: UUID of the scenario to evaluate
            submission_id: UUID of the submission
        
        Returns:
            Dictionary with "primary" key containing the KPI value
        """
        logger.info(
            f"Running scenario evaluation\n"
            f"  Scenario: {scenario_id}\n"
            f"  Submission: {submission_id}\n"
            f"  KPI: {self.kpi_info['name']}"
        )
        
        # Check cache
        cache_key = f"{scenario_id}_{submission_id}"
        
        if cache_key in self._metrics_cache:
            logger.info(f"Using cached metrics for {cache_key}")
            all_metrics = self._metrics_cache[cache_key]
        else:
            # Run complete evaluation
            logger.info(f"No cache found - running complete evaluation")
            all_metrics = self._run_complete_evaluation(scenario_id, submission_id)
            
            # Cache results
            self._metrics_cache[cache_key] = all_metrics
            logger.info(f"Cached metrics for {cache_key}")
        
        # Extract KPI-specific value
        metric_key = self.kpi_info['metric_key']
        kpi_value = all_metrics[metric_key]
        
        logger.info(
            f"KPI Result: {self.kpi_info['name']} = {kpi_value}\n"
            f"  Description: {self.kpi_info['description']}"
        )
        
        return {"primary": float(kpi_value)}
    
    def _run_complete_evaluation(self, scenario_id: str, submission_id: str) -> Dict:
        """
        Run complete multi-attacker evaluation and compute ALL metrics.
        
        Args:
            scenario_id: UUID of the scenario to evaluate
            submission_id: UUID of the submission
        
        Returns:
            Dictionary containing ALL computed metrics for all 9 KPIs
        """
        logger.info(
            f"Starting complete evaluation\n"
            f"  Attackers: {self.ATTACKER_TYPES}\n"
            f"  Episodes: {self.NUM_EPISODES}\n"
            f"  Environment: {self.ENV_NAME}"
        )
        
        # Import framework modules (now from framework/ folder)
        from evaluation_framework.result_getter import result_getter
        from evaluation_framework.metrics import metrics
        
        # Initialize environment
        env = self._initialize_environment(scenario_id)
        
        # Load attackers
        attackers = self._load_attackers(env)
        
        # Create temporary directory for results
        with tempfile.TemporaryDirectory() as temp_dir:
            logger.info(f"Running episodes in temp directory: {temp_dir}")
            
            # Run evaluation using result_getter
            rg = result_getter(
                env=env,
                defender=self._defender_agent,
                n_episodes=self.NUM_EPISODES,
                save_folder=temp_dir,
                attackers=attackers
            )
            
            # This runs all episodes and computes metrics
            rg.calculate_metrics()
            
            # Load computed metrics
            logger.info("Loading computed metrics from pickle files")
            
            # Load unperturbed data
            with open(f"{temp_dir}/unperturbed.pkl", "rb") as f:
                unperturbed_data = pickle.load(f)
            
            # Load metrics for each attacker
            metrics_dicts = []
            for attacker in attackers:
                with open(f"{temp_dir}/{attacker.pickle_file}", "rb") as f:
                    data_dict = pickle.load(f)
                
                m = metrics(
                    data_dict,
                    unperturbed_data,
                    env.do_nothing_action(),
                    env.get_similarity_score,
                    model_name=attacker.model_name
                )
                metrics_dicts.append(m)
            
            # Aggregate metrics across all attackers
            all_metrics = self._aggregate_metrics(metrics_dicts, unperturbed_data)
        
        logger.info("Complete evaluation finished")
        
        return all_metrics
    
    def _aggregate_metrics(self, metrics_dicts: List, unperturbed_data: Dict) -> Dict:
        """
        Aggregate metrics from all attackers into a single result.
        
        Args:
            metrics_dicts: List of metrics objects from each attacker
            unperturbed_data: Dictionary with unperturbed episode data
        
        Returns:
            Dictionary with aggregated metrics for all 9 KPIs
        """
        logger.info(f"Aggregating metrics from {len(metrics_dicts)} attackers")
        
        # Extract metrics from each attacker
        vulnerability_scores = []
        steps_survived = []
        similarity_scores = []
        reward_drops = []
        action_change_freqs = []
        areas_between_curves = []
        degradation_times = []
        restoration_times = []
        state_similarities = []
        
        for m in metrics_dicts:
            logger.info(f"\n{'='*60}")
            logger.info(f"ATTACKER: {m.model_name}")
            logger.info(f"{'='*60}")
            
            # Robustness metrics for this attacker
            vuln = m.perturb_vulnerability.mean()
            steps = m.metrics_robustness['n_steps'].mean()
            sim = m.metrics_robustness['similarity_score'].mean()
            
            # Reward drop
            total_reward_perturbed = m.metrics_robustness['total_reward'].sum()
            total_reward_unperturbed = sum([sum([r for r in ep if not np.isnan(r)]) 
                                        for ep in unperturbed_data['rewards']])
            if total_reward_unperturbed > 0:
                reward_drop = 100 * (total_reward_unperturbed - total_reward_perturbed) / total_reward_unperturbed
            else:
                reward_drop = 0.0
            
            # Action change frequency
            n_changed = m.metrics_robustness['n_actions_changed'].sum()
            n_total = m.metrics_robustness['n_steps_with_act'].sum()
            action_freq = n_changed / n_total if n_total > 0 else 0
            
            # Resilience metrics
            if 'area_per_1000_steps' in m.metrics_resilience.columns:
                area = m.metrics_resilience['area_per_1000_steps'].values[0]
            elif 'area' in m.metrics_resilience.columns:
                area = m.metrics_resilience['area'].values[0]
            else:
                area = 0.0
            
            degr = m.metrics_resilience['degradation_time'].values[0]
            rest = m.metrics_resilience['restoration_time'].values[0]
            
            # State similarity
            state_sim = np.mean([np.mean(ep) for ep in m.cos_similarity_all])
            
            # Print all metrics for this attacker
            logger.info(f"  Robustness Metrics:")
            logger.info(f"    - Vulnerability:        {vuln:.4f}")
            logger.info(f"    - Steps Survived:       {steps:.1f}")
            logger.info(f"    - Severity of Change:   {1.0 - sim:.4f}")
            logger.info(f"    - Reward Drop (%):      {reward_drop:.2f}")
            logger.info(f"    - Action Change Freq:   {action_freq:.4f}")
            logger.info(f"  Resilience Metrics:")
            logger.info(f"    - Area Between Curves:  {area:.4f}")
            logger.info(f"    - Degradation Time:     {degr:.1f}")
            logger.info(f"    - Restoration Time:     {rest:.1f}")
            logger.info(f"    - State Similarity:     {state_sim:.4f}")
            
            # Append to lists
            vulnerability_scores.append(vuln)
            steps_survived.append(steps)
            similarity_scores.append(sim)
            reward_drops.append(reward_drop)
            action_change_freqs.append(action_freq)
            areas_between_curves.append(area)
            degradation_times.append(degr)
            restoration_times.append(rest)
            state_similarities.append(state_sim)
        
        # Compute means across attackers
        aggregated = {
            # Robustness KPIs
            'perturb_vulnerability': np.mean(vulnerability_scores),
            'n_steps_survived': np.mean(steps_survived),
            'severity_of_change': 1.0 - np.mean(similarity_scores),  # Inverted (higher=worse)
            'reward_drop_percent': np.mean(reward_drops),
            'action_change_freq': np.mean(action_change_freqs),
            
            # Resilience KPIs
            'area_between_curves': np.mean(areas_between_curves),
            'degradation_time': np.mean(degradation_times),
            'restoration_time': np.mean(restoration_times),
            'state_similarity': np.mean(state_similarities),
        }
        
        logger.info(f"\n{'='*60}")
        logger.info(f"AGGREGATED METRICS (Average across {len(metrics_dicts)} attackers)")
        logger.info(f"{'='*60}")
        logger.info(
            "\n".join([f"  {k}: {v:.4f}" for k, v in aggregated.items()])
        )
        
        return aggregated
    
    def _initialize_environment(self, scenario_id: str):
        """
        Initialize Grid2Op environment for the given scenario.
        
        Args:
            scenario_id: UUID of the scenario
        
        Returns:
            Initialized Grid2Op environment with Environment wrapper
        """
        logger.info(f"Initializing environment for scenario: {scenario_id}")
        
        import grid2op
        from attack_models.Environment import Environment
        
        # Create base Grid2Op environment
        base_env = grid2op.make(self.ENV_NAME)
        
        # Wrap with Environment for attacker support
        env = Environment(base_env, self._defender_agent)
        
        logger.info(f"Environment initialized: {self.ENV_NAME}")
        
        return env
    
    def _load_attackers(self, env) -> List:
        """
        Load all attacker agents from the framework.
        
        Args:
            env: Environment wrapper instance
        
        Returns:
            List of attacker agent objects
        """
        logger.info(f"Loading {len(self.ATTACKER_TYPES)} attacker types")
        
        # Import attacker classes from framework
        from attack_models.SACAttacker import SACAttacker
        from attack_models.PPOAttacker import PPOAttacker
        from attack_models.RLPerturbAttacker import RLPerturbAttacker
        from attack_models.GEPerturbAttacker import GEPerturbAttacker
        from attack_models.RPerturbAttacker import RPerturbAttacker
        from attack_models.LambdaPIRAttacker import LambdaPIRAttacker
        
        attackers = []
        trained_models_path = os.path.join(self.FRAMEWORK_PATH, "trained_models")
        
        for attacker_type in self.ATTACKER_TYPES:
            logger.info(f"Loading attacker: {attacker_type}")
            
            try:
                if attacker_type == "SAC_5":
                    attacker = SACAttacker(
                        model_path=os.path.join(trained_models_path, "SAC.zip"),
                        factor=5,
                        model_name="SAC_5",
                        pickle_file="sac_5.pkl"
                    )
                elif attacker_type == "SAC_10":
                    attacker = SACAttacker(
                        model_path=os.path.join(trained_models_path, "SAC.zip"),
                        factor=10,  
                        model_name="SAC_10",
                        pickle_file="sac_10.pkl"
                    )
                elif attacker_type == "PPO":
                    attacker = PPOAttacker(
                        model_path=os.path.join(trained_models_path, "PPO.zip"),
                        model_name="PPO",
                        pickle_file="ppo.pkl"
                    )
                elif attacker_type == "RLPerturb":
                    attacker = RLPerturbAttacker(
                        model_path=os.path.join(trained_models_path, "RLPerturbAgent", "trained_rlpa_0.pth"),
                        target_path=os.path.join(trained_models_path, "RLPerturbAgent", "trained_rlpa_target_net_0.pth"),
                        env=env.env,  # Base Grid2Op env
                        agent=self._defender_agent
                    )
                elif attacker_type == "GEPerturb":
                    attacker = GEPerturbAttacker(
                        env=env.env,
                        agent=self._defender_agent, 
                        n_iter=10
                    )
                elif attacker_type == "Random":
                    attacker = RPerturbAttacker(
                        env=env.env,
                        prob_perturb=0.6
                    )
                elif attacker_type == "LambdaPIR":
                    attacker = LambdaPIRAttacker(
                        model_path=os.path.join(trained_models_path, "SAC.zip"),
                        env=env.env,
                        agent=self._defender_agent,
                        lambda_param=0.7,
                        initial_prob_policy=0.2,
                        epsilon=1.0,
                        gradient_step_size=0.1,
                        refinement_iterations=20,
                        decay_schedule="exponential",
                        name="LambdaPIR",
                        use_gpu=False
                    )
                else:
                    logger.warning(f"Unknown attacker type: {attacker_type}, skipping")
                    continue
                
                attackers.append(attacker)
                logger.info(f"Loaded attacker: {attacker_type}")
                
            except Exception as e:
                logger.error(f"Failed to load attacker {attacker_type}: {e}")
                continue
        
        logger.info(f"Successfully loaded {len(attackers)} attackers")
        
        return attackers
    
    def _load_agent_from_pickle(self, url: str):
        """
        Load agent from pickle file.
        
        Args:
            url: URL to pickle file
        
        Returns:
            Loaded agent object
        """
        logger.info(f"Loading agent from pickle: {url}")
        
        # Download file
        response = requests.get(url, timeout=300)
        response.raise_for_status()
        
        # Save temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pkl') as f:
            f.write(response.content)
            temp_path = f.name
        
        # Load pickle
        with open(temp_path, 'rb') as f:
            agent = pickle.load(f)
        
        # Clean up
        os.remove(temp_path)
        
        logger.info("Agent loaded from pickle successfully")
        
        return agent
    
    def _load_agent_from_zip(self, url: str):
        """
        Load agent from zip file.
        
        Args:
            url: URL to zip file containing agent code
        
        Returns:
            Loaded agent object
        """
        logger.info(f"Loading agent from zip: {url}")
        
        # Download file
        response = requests.get(url, timeout=300)
        response.raise_for_status()
        
        # Save and extract
        with tempfile.NamedTemporaryFile(delete=False, suffix='.zip') as f:
            f.write(response.content)
            temp_zip = f.name
        
        temp_dir = tempfile.mkdtemp()
        
        # Extract
        with zipfile.ZipFile(temp_zip, 'r') as zip_ref:
            zip_ref.extractall(temp_dir)
        
        # Clean up zip
        os.remove(temp_zip)
        
        # Load agent using CurriculumAgent
        import grid2op
        from modified_curriculum_classes.baseline import CurriculumAgent
        
        # Create environment to get action/observation space
        env = grid2op.make(self.ENV_NAME)
        
        # Create and load agent
        agent = CurriculumAgent(env.action_space, env.observation_space, "defender_agent")
        
        # Find model and actions paths
        model_path = os.path.join(temp_dir, "model")
        if not os.path.exists(model_path):
            model_path = temp_dir  # Model might be at root
        
        # Actions path - use framework's action definitions or from zip
        actions_path = os.path.join(temp_dir, "actions")
        if not os.path.exists(actions_path):
            actions_path = os.path.join(self.FRAMEWORK_PATH, "action_definitions")
        
        agent.load(model_path, actions_path)
        
        logger.info("Agent loaded from zip successfully")
        
        return agent
    
    def _load_agent_from_docker(self, url: str):
        """
        Load agent from Docker image.
        
        Args:
            url: Docker image URL
        
        Returns:
            Loaded agent object (wrapped Docker container)
        """
        logger.info(f"Loading agent from Docker: {url}")
        
        # Docker support requires additional setup
        raise NotImplementedError(
            "Docker agent loading not yet implemented. "
            "Please submit agents as pickle or zip files."
        )


# ============================================================================
# Convenience aliases for each KPI (all use the same TestRunner)
# These allow FAB orchestrator to reference specific KPIs
# ============================================================================

# Robustness KPIs
class TestRunner_KPI_DF_069_Power_Grid(MultiAttackerRobustnessTestRunner):
    """KPI-DF-069: Drop-off in reward"""
    pass

class TestRunner_KPI_FF_070_Power_Grid(MultiAttackerRobustnessTestRunner):
    """KPI-FF-070: Frequency changed output AI agent"""
    pass

class TestRunner_KPI_SF_071_Power_Grid(MultiAttackerRobustnessTestRunner):
    """KPI-SF-071: Severity of changed output AI agent"""
    pass

class TestRunner_KPI_SF_072_Power_Grid(MultiAttackerRobustnessTestRunner):
    """KPI-SF-072: Steps survived with perturbations"""
    pass

class TestRunner_KPI_VF_073_Power_Grid(MultiAttackerRobustnessTestRunner):
    """KPI-VF-073: Vulnerability to perturbation"""
    pass

# Resilience KPIs
class TestRunner_KPI_AF_074_Power_Grid(MultiAttackerRobustnessTestRunner):
    """KPI-AF-074: Area between reward curves"""
    pass

class TestRunner_KPI_DF_075_Power_Grid(MultiAttackerRobustnessTestRunner):
    """KPI-DF-075: Degradation time"""
    pass

class TestRunner_KPI_RF_076_Power_Grid(MultiAttackerRobustnessTestRunner):
    """KPI-RF-076: Restorative time"""
    pass

class TestRunner_KPI_SF_077_Power_Grid(MultiAttackerRobustnessTestRunner):
    """KPI-SF-077: Similarity state to unperturbed situation"""
    pass
