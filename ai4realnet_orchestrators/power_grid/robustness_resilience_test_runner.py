import pickle
import logging
import os
import tempfile
from typing import Dict, List

import numpy as np
from ai4realnet_orchestrators.power_grid.power_grid_test_runner import PowerGridTestRunner, FRAMEWORK_PATH

logger = logging.getLogger(__name__)

# KPI ID to metric mapping
ROBUSTNESS_RESILIENCE_KPI_MAPPING = {
    # Robustness KPIs (Benchmark: 3810191b-8cfd-4b03-86b2-f7e530aab30d)
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
    "3d033ec6-942a-4b03-b26e-f8152ba48022": {
        "name": "KPI-SF-071: Severity of changed output",
        "metric_key": "severity_of_change",
        "description": "Severity of action changes [0-1, higher=worse]"
    },
    "a121d8bd-1943-41ba-b3a7-472a0154f8f9": {
        "name": "KPI-SF-072: Steps survived with perturbations",
        "metric_key": "n_steps_survived",
        "description": "Number of timesteps before failure"
    },
    "b8a9a411-7cfe-4c1d-b9a6-eef1c0efe920": {
        "name": "KPI-VF-073: Vulnerability to perturbation",
        "metric_key": "perturb_vulnerability",
        "description": "Proportion of features vulnerable to attack [0-1]"
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


class RobustnessResilienceTestRunner(PowerGridTestRunner):
    """
    Extended TestRunner for Robustness & Resilience KPIs (069-077).
    
    Inherits from PowerGridTestRunner and implements getResult() to run
    multi-attacker evaluation against the defender agent.
    
    Attackers:
    - GEPerturb: Gradient estimation perturbation
    - LambdaPIR: Lambda policy iteration with refinement
    - Random: Random perturbations
    - PPO: PPO-trained attacker
    - SAC_5/SAC_10: SAC-trained attackers with different factors
    - RLPerturb: RL-based perturbation agent
    
    Single evaluation computes ALL 9 metrics, results are cached to avoid
    re-running when multiple KPIs are evaluated for the same submission.
    """
    
    # Evaluation configuration
    ATTACKER_TYPES = ["GEPerturb", "LambdaPIR", "Random", "PPO", "SAC_10", "SAC_5", "RLPerturb"]
    NUM_EPISODES = 50
    
    # Class-level cache: {submission_id: all_metrics_dict}
    _metrics_cache: Dict[str, Dict] = {}
    
    # Specific KPI mapping for this category
    KPI_MAPPING = ROBUSTNESS_RESILIENCE_KPI_MAPPING

    def _compute_all_metrics(self, env, env_shift, agent) -> Dict:
        """
        Run complete multi-attacker evaluation and compute ALL metrics.

        Args:
            env: Grid2Op environment
            env_shift: Shift environment (ignored here)
            agent: Defender agent

        Returns:
            Dictionary containing ALL computed metrics for all 9 KPIs
        """
        logger.info(
            f"Starting complete evaluation in {self.__class__.__name__}\n"
            f"  Attackers: {self.ATTACKER_TYPES}\n"
            f"  Episodes: {self.NUM_EPISODES}"
        )

        # Import framework modules
        from evaluation_framework.result_getter import result_getter
        from evaluation_framework.metrics import metrics
        from attack_models.Environment import Environment

        # Wrap environment for attacker support
        wrapped_env = Environment(env, agent)

        # Load attackers
        attackers = self._load_attackers(wrapped_env, agent)

        # Run evaluation in temporary directory
        with tempfile.TemporaryDirectory() as temp_dir:
            logger.info(f"Running episodes in: {temp_dir}")

            rg = result_getter(
                env=wrapped_env,
                defender=agent,
                n_episodes=self.NUM_EPISODES,
                save_folder=temp_dir,
                attackers=attackers
            )
            rg.calculate_metrics()

            # Load unperturbed baseline
            with open(os.path.join(temp_dir, "unperturbed.pkl"), "rb") as f:
                unperturbed_data = pickle.load(f)

            # Load and process metrics for each attacker
            metrics_list = []
            for attacker in attackers:
                pkl_path = os.path.join(temp_dir, attacker.pickle_file)
                with open(pkl_path, "rb") as f:
                    data_dict = pickle.load(f)

                m = metrics(
                    data_dict,
                    unperturbed_data,
                    wrapped_env.do_nothing_action(),
                    wrapped_env.get_similarity_score,
                    model_name=attacker.model_name
                )
                metrics_list.append(m)

            # Aggregate metrics across all attackers
            return self._aggregate_metrics(metrics_list, unperturbed_data)
    
    def _load_attackers(self, env, agent) -> List:
        """
        Load all attacker agents from the framework.
        
        Args:
            env: Environment wrapper instance
            agent: Defender agent
            
        Returns:
            List of attacker agent objects
        """
        logger.info(f"Loading {len(self.ATTACKER_TYPES)} attacker types")
        
        from attack_models.SACAttacker import SACAttacker
        from attack_models.PPOAttacker import PPOAttacker
        from attack_models.RLPerturbAttacker import RLPerturbAttacker
        from attack_models.GEPerturbAttacker import GEPerturbAttacker
        from attack_models.RPerturbAttacker import RPerturbAttacker
        from attack_models.LambdaPIRAttacker import LambdaPIRAttacker
        
        attackers = []
        trained_models_path = os.path.join(FRAMEWORK_PATH, "trained_models")
        
        # Attacker configurations
        attacker_configs = {
            "GEPerturb": lambda: GEPerturbAttacker(
                env=env.env, agent=agent, n_iter=10
            ),
            "LambdaPIR": lambda: LambdaPIRAttacker(
                model_path=os.path.join(trained_models_path, "SAC.zip"),
                env=env.env, agent=agent,
                lambda_param=0.7, initial_prob_policy=0.2, epsilon=1.0,
                gradient_step_size=0.1, refinement_iterations=20,
                decay_schedule="exponential", name="LambdaPIR", use_gpu=False
            ),
            "Random": lambda: RPerturbAttacker(
                env=env.env, prob_perturb=0.6
            ),
            "PPO": lambda: PPOAttacker(
                model_path=os.path.join(trained_models_path, "PPO.zip"),
                model_name="PPO", pickle_file="ppo.pkl"
            ),
            "SAC_10": lambda: SACAttacker(
                model_path=os.path.join(trained_models_path, "SAC.zip"),
                factor=10, model_name="SAC_10", pickle_file="sac_10.pkl"
            ),
            "SAC_5": lambda: SACAttacker(
                model_path=os.path.join(trained_models_path, "SAC.zip"),
                factor=5, model_name="SAC_5", pickle_file="sac_5.pkl"
            ),
            "RLPerturb": lambda: RLPerturbAttacker(
                model_path=os.path.join(trained_models_path, "RLPerturbAgent", "trained_rlpa_0.pth"),
                target_path=os.path.join(trained_models_path, "RLPerturbAgent", "trained_rlpa_target_net_0.pth"),
                env=env.env, agent=agent
            ),
        }
        
        for attacker_type in self.ATTACKER_TYPES:
            try:
                if attacker_type in attacker_configs:
                    attacker = attacker_configs[attacker_type]()
                    attackers.append(attacker)
                    logger.info(f"Loaded attacker: {attacker_type}")
            except Exception as e:
                logger.error(f"Failed to load attacker {attacker_type}: {e}")
        
        logger.info(f"Successfully loaded {len(attackers)} attackers")
        return attackers
    
    def _aggregate_metrics(self, metrics_list: List, unperturbed_data: Dict) -> Dict:
        """
        Aggregate metrics from all attackers into final results.
        
        Args:
            metrics_list: List of metrics objects from each attacker
            unperturbed_data: Dictionary with unperturbed episode data
            
        Returns:
            Dictionary with aggregated metrics for all 9 KPIs
        """
        logger.info(f"Aggregating metrics from {len(metrics_list)} attackers")
        
        # Collect metrics from each attacker
        vulnerability_scores = []
        steps_survived = []
        similarity_scores = []
        reward_drops = []
        action_change_freqs = []
        areas_between_curves = []
        degradation_times = []
        restoration_times = []
        state_similarities = []
        
        # Calculate total unperturbed reward once
        total_reward_unperturbed = sum([
            sum([r for r in ep if not np.isnan(r)]) 
            for ep in unperturbed_data['rewards']
        ])
        
        for m in metrics_list:
            logger.info(f"Processing metrics for attacker: {m.model_name}")
            
            # Robustness metrics
            vulnerability_scores.append(m.perturb_vulnerability.mean())
            steps_survived.append(m.metrics_robustness['n_steps'].mean())
            similarity_scores.append(m.metrics_robustness['similarity_score'].mean())
            
            # Reward drop calculation
            total_reward_perturbed = m.metrics_robustness['total_reward'].sum()
            if total_reward_unperturbed > 0:
                reward_drop = 100 * (total_reward_unperturbed - total_reward_perturbed) / total_reward_unperturbed
            else:
                reward_drop = 0.0
            reward_drops.append(reward_drop)
            
            # Action change frequency
            n_changed = m.metrics_robustness['n_actions_changed'].sum()
            n_total = m.metrics_robustness['n_steps_with_act'].sum()
            action_change_freqs.append(n_changed / n_total if n_total > 0 else 0)
            
            # Resilience metrics
            if 'area_per_1000_steps' in m.metrics_resilience.columns:
                areas_between_curves.append(m.metrics_resilience['area_per_1000_steps'].values[0])
            elif 'area' in m.metrics_resilience.columns:
                areas_between_curves.append(m.metrics_resilience['area'].values[0])
            else:
                areas_between_curves.append(0.0)
            
            degradation_times.append(m.metrics_resilience['degradation_time'].values[0])
            restoration_times.append(m.metrics_resilience['restoration_time'].values[0])
            state_similarities.append(np.mean([np.mean(ep) for ep in m.cos_similarity_all]))
        
        # Compute averages across all attackers
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
        
        logger.info(f"Aggregated metrics: {aggregated}")
        
        return aggregated


class TestRunner_KPI_DF_069_Power_Grid(RobustnessResilienceTestRunner):
    """KPI-DF-069: Drop-off in reward"""
    pass


class TestRunner_KPI_FF_070_Power_Grid(RobustnessResilienceTestRunner):
    """KPI-FF-070: Frequency changed output AI agent"""
    pass


class TestRunner_KPI_SF_071_Power_Grid(RobustnessResilienceTestRunner):
    """KPI-SF-071: Severity of changed output AI agent"""
    pass


class TestRunner_KPI_SF_072_Power_Grid(RobustnessResilienceTestRunner):
    """KPI-SF-072: Steps survived with perturbations"""
    pass


class TestRunner_KPI_VF_073_Power_Grid(RobustnessResilienceTestRunner):
    """KPI-VF-073: Vulnerability to perturbation"""
    pass


class TestRunner_KPI_AF_074_Power_Grid(RobustnessResilienceTestRunner):
    """KPI-AF-074: Area between reward curves"""
    pass


class TestRunner_KPI_DF_075_Power_Grid(RobustnessResilienceTestRunner):
    """KPI-DF-075: Degradation time"""
    pass


class TestRunner_KPI_RF_076_Power_Grid(RobustnessResilienceTestRunner):
    """KPI-RF-076: Restorative time"""
    pass


class TestRunner_KPI_SF_077_Power_Grid(RobustnessResilienceTestRunner):
    """KPI-SF-077: Similarity state to unperturbed situation"""
    pass
