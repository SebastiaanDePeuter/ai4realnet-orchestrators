import os
import random
import logging
import pandas as pd
from typing import Dict, List

from grid2op.utils import ScoreL2RPN2023
from ai4realnet_orchestrators.power_grid.power_grid_test_runner import PowerGridTestRunner

logger = logging.getLogger(__name__)

# KPI ID to metric mapping for Operational KPIs
OPERATIONAL_KPI_MAPPING = {
    # Operational KPIs (Benchmark: 4b0be731-8371-4e4e-a673-b630187b0bb8)
    "aba10b3f-0d5c-4f90-aec4-69460bbb098b": {
        "name": "KPI-AF-008: Assistant alert accuracy",
        "metric_key": "assistant_confidence_score",
        "description": "Assistant alert accuracy"
    },
    "ab91af79-ffc3-4da7-916a-6574609dc1b6": {
        "name": "KPI-CF-012: Carbon intensity",
        "metric_key": "nres_score",
        "description": "Carbon intensity"
    },
    "ae4dcac7-c559-457e-902d-ee35d064bb3f": {
        "name": "KPI-OF-036: Operation score",
        "metric_key": "op_score",
        "description": "Operation score"
    }
}


def get_scoring_config(chronics_path: str, seed: int, scenario_names: List[str] = None):
    """
    Retrieve scenario information (number, length) for operational KPI calculation and generate seeds.
    """
    if scenario_names is None:
        scenario_names = sorted([name for name in os.listdir(chronics_path)
                                 if os.path.isdir(os.path.join(chronics_path, name)) and name != "chronic_example"])
    else:
        scenario_names = sorted(scenario_names)

    scenario_lengths = [len(pd.read_csv(os.path.join(chronics_path, name + "/load_p.csv.bz2"), compression="bz2"))
                        for name in scenario_names]

    rng = random.Random(seed)
    config = {
        "nb_scenario": len(scenario_names),
        "total_timesteps": sum(scenario_lengths),
        "episodes_info": {}
    }

    for name, length in zip(scenario_names, scenario_lengths):
        config["episodes_info"][name] = {
            "length": length,
            "env_seed": rng.randint(0, 2**31 - 1),
            "agent_seed": rng.randint(0, 2**31 - 1)
        }

    return config


def evaluate_operational_kpis(env, agent) -> dict:
    """
    Evaluate operational KPIs using Grid2Op's ScoreL2RPN2023.

    Returns scores for:
    - op_score: Operation score
    - nres_score: Non-renewable energy score (carbon intensity)
    - assistant_confidence_score: Assistant alert accuracy
    """
    chronics_path = os.path.join(env.get_path_env(), "chronics")
    
    # Discover scenario names from the environment's chronics handler
    scenario_names = None
    if hasattr(env.chronics_handler, "real_data") and hasattr(env.chronics_handler.real_data, "subpaths"):
        scenario_names = [os.path.basename(p) for p in env.chronics_handler.real_data.subpaths]
        
    config = get_scoring_config(chronics_path, 4295, scenario_names=scenario_names)

    episodes_info = config["episodes_info"]
    scenario_names = list(episodes_info.keys())
    env_seeds = [int(episodes_info[name]["env_seed"]) for name in scenario_names]
    agent_seeds = [int(episodes_info[name]["agent_seed"]) for name in scenario_names]

    nb_scenario = int(config["nb_scenario"])

    scoring = ScoreL2RPN2023(
        env=env,
        env_seeds=env_seeds,
        agent_seeds=agent_seeds,
        nb_scenario=nb_scenario,
        min_losses_ratio=0.8,
        verbose=0,
        max_step=-1,
        nb_process_stats=1,
        add_nb_highres_sim=True,
        weight_op_score=0.6,
        weight_assistant_score=0.25,
        weight_nres_score=0.15,
        min_nres_score=-100,
        min_assistant_score=-300
    )

    all_scores, _, _, _ = scoring.get(agent)
    scores_per_episode = {
        "op_score": [float(score[1]) for score in all_scores],
        "nres_score": [float(score[2]) for score in all_scores],
        "assistant_confidence_score": [float(score[3]) for score in all_scores],
    }

    weights = [float(episodes_info[name]["length"]) / float(config["total_timesteps"]) for name in scenario_names]
    total_op_score = sum(w * s for w, s in zip(weights, scores_per_episode["op_score"]))
    total_nres_score = sum(w * s for w, s in zip(weights, scores_per_episode["nres_score"]))
    total_assistant_score = sum(w * s for w, s in zip(weights, scores_per_episode["assistant_confidence_score"]))

    return {
        "op_score": total_op_score,
        "nres_score": total_nres_score,
        "assistant_confidence_score": total_assistant_score,
    }


class OperationalTestRunner(PowerGridTestRunner):
    """
    Extended TestRunner for Operational KPIs (008, 012, 036).

    Inherits from PowerGridTestRunner and implements getResult() to run
    operational evaluation. Results are cached per submission.
    """

    # Class-level cache: {submission_id: all_metrics_dict}
    _metrics_cache: Dict[str, Dict] = {}

    # Specific KPI mapping for this category
    KPI_MAPPING = OPERATIONAL_KPI_MAPPING

    def _compute_all_metrics(self, env, env_shift, agent) -> Dict:
        """Evaluate operational KPIs and return all metrics."""
        return evaluate_operational_kpis(env, agent)


class TestRunner_KPI_AF_008_Power_Grid(OperationalTestRunner):
    """KPI-AF-008: Assistant alert accuracy"""
    pass


class TestRunner_KPI_CF_012_Power_Grid(OperationalTestRunner):
    """KPI-CF-012: Carbon intensity"""
    pass


class TestRunner_KPI_OF_036_Power_Grid(OperationalTestRunner):
    """KPI-OF-036: Operation score"""
    pass
