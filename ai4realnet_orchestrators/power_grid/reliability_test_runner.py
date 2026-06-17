import logging
from typing import Dict, List

from domain_shift_kpis.adaptation_time import DsAdaptationTime
from ai4realnet_orchestrators.power_grid.power_grid_test_runner import PowerGridTestRunner

logger = logging.getLogger(__name__)

def evaluate_domain_shift_kpis(env, env_shift, agent) -> Dict:
    if env_shift is None:
        logger.warning("No shift environment provided. Domain shift KPIs cannot be computed.")
        return {
            "adaptation_time": 0.0,
            "performance_drop": 0.0
        }

    from ExpertAgent.utils.helper_functions import make_gymenv
    env_gym = make_gymenv(env, obs_attr_to_keep=["rho"], action_space_path="read_from_file", act_to_keep=("set_bus",))
    env_gym_shift = make_gymenv(env_shift, obs_attr_to_keep=["rho"], action_space_path="read_from_file", act_to_keep=("set_bus",))
    
    ds_kpi = DsAdaptationTime(agent=agent, 
                              trained_model_path=None, 
                              env=env_gym, 
                              env_shift=env_gym_shift
                             )
    
    # save_path = os.path.join(here, "..", "trained_models", "PPO_SB3_FINETUNE")
    
    train_kwargs = {
        "train_steps": int(1e3),
        "load_path": None,
        "save_path": None,
        "save_freq": 5000,
    }
    
    eval_kwargs = {
        "n_eval_episodes": 10,
        "render": False,
        "deterministic": True,
        "return_episode_rewards": True
    }
    
    results = ds_kpi.compute(acceptance_threshold=200.,
                             fine_tune_budget=int(15e3),
                             agent_train_fun=agent.train_static,
                             agent_train_kwargs=train_kwargs,
                             agent_eval_fun=agent.evaluate,
                             agent_eval_kwargs=eval_kwargs,
                             min_train_steps=int(1e3),
                             save_path=None
                             )
    return results

# KPI ID to metric mapping
RELIABILITY_KPI_MAPPING = {
    # Reliability KPIs (Benchmark: 43040944-39ac-47c9-b91d-bc8ca5693b3c)
    "855729a4-6729-4ae2-bb8d-443ef4867d94": {
        "name": "KPI-DF-052: Domain shift adaptation time",
        "metric_key": "adaptation_time",
        "description": "Iterations required for an agent to adapt its policy to domain shift"
    },
    "c5e4f893-4302-47e8-98d6-b5fbcb10963a": {
        "name": "KPI-DF-057: Domain shift success rate drop",
        "metric_key": "performance_drop",
        "description": "The performance drop when encountering a domain shift"
    },
}

class ReliabilityTestRunner(PowerGridTestRunner):
    """
    Extended TestRunner for Reliability KPIs (052-058 + 090).
    
    Inherits from PowerGridTestRunner and implements getResult() to run
    the reliability evaluation against a RL agent
    
    Single evaluation computes ALL the metrics
    """
    
    # Class-level cache: {submission_id: all_metrics_dict}
    _metrics_cache: Dict[str, Dict] = {}

    # Specific KPI mapping for this category
    KPI_MAPPING = RELIABILITY_KPI_MAPPING
    
    def _compute_all_metrics(self, env, env_shift, agent) -> Dict:
        """Evaluate reliability KPIs and return all metrics."""
        return evaluate_domain_shift_kpis(env, env_shift, agent)


class TestRunner_KPI_DF_052_Power_Grid(ReliabilityTestRunner):
    """KPI-DF-052: Domain shift adaptation time"""
    pass


class TestRunner_KPI_DF_057_Power_Grid(ReliabilityTestRunner):
    """KPI-DF-057: Domain shift Success Rate Drop (performance drop)"""
    pass
