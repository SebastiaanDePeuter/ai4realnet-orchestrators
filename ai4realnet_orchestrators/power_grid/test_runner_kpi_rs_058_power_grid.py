import logging
from typing import Dict
import numpy as np

from grid2op.Agent import BaseAgent, RandomAgent
from grid2op.Runner import Runner

from ai4realnet_orchestrators.power_grid.power_grid_test_runner import PowerGridTestRunner

logger = logging.getLogger(__name__)

class InterveningAgent(BaseAgent):
    def __init__(self, action_space, base_policy, intervention_rate):
        super().__init__(action_space)
        self.intervention_policy = RandomAgent(action_space)
        self.intervention_rate = float(intervention_rate)
        self.base_policy = base_policy

    def reset(self, obs):
        self.intervention_policy.reset(obs)
        self.base_policy.reset(obs)

    def act(self, observation, reward, done):
        intervened_act = self.intervention_policy.act(observation, reward, done)
        base_policy_act = self.base_policy.act(observation, reward, done)
        if self.space_prng.rand() < self.intervention_rate:
            return intervened_act
        return base_policy_act
    
    def save_state(self, savestate_path):
        raise NotImplementedError()

    def load_state(self, loadstate_path):
        raise NotImplementedError()

# Reliability KPIs (Benchmark: 3810191b-8cfd-4b03-86b2-f7e530aab30d)
ROBUSTNESS_TO_OPERATOR_KPI_MAPPING = {
    "75cc9343-9371-4eb1-9613-22a26c67fc00": {
        "name": "KPI-RS-058: Robustness to operator input",
        "metric_key": "gained_reward",
        "description": "Reward gained by an Agent when paired with a (simulated) human compared over the Agent acting alone."
    }}

class TestRunner_KPI_RS_058_Power_Grid(PowerGridTestRunner):
    KPI_MAPPING = ROBUSTNESS_TO_OPERATOR_KPI_MAPPING

    _metrics_cache: Dict[str, Dict] = {}

    def _compute_all_metrics(self, env, env_shift, agent) -> Dict:
        def run_and_get_metrics(runner):
            mean_rew = []
            mean_rel_ep_len = []
            for _, _, cum_reward, nb_time_step, max_ts in runner.run(nb_episode=25, max_iter=env.chronics_handler.max_timestep()):
                mean_rew.append(cum_reward)
                mean_rel_ep_len.append(nb_time_step / max_ts)
            return np.mean(mean_rew), np.mean(mean_rel_ep_len)

        from grid2op.Agent import RecoPowerlineAgent
        agent = RecoPowerlineAgent(env.action_space)

        intervened_runner = Runner(**env.get_params_for_runner(), agentClass=None, agentInstance=InterveningAgent(env.action_space, agent, 0.05))
        reference_runner = Runner(**env.get_params_for_runner(), agentClass=None, agentInstance=agent)

        intervened_mean_rew, intervened_mean_rel_ep_len = run_and_get_metrics(intervened_runner)
        reference_mean_rew, reference_mean_rel_ep_len = run_and_get_metrics(reference_runner)

        return {
            'gained_reward': (intervened_mean_rew - reference_mean_rew) / reference_mean_rew,
            'gained_ep_completion_rate': intervened_mean_rel_ep_len - reference_mean_rel_ep_len,
        }

        return {}