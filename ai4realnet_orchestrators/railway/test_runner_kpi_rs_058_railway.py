import logging
from pathlib import Path
import shutil
import os

import numpy as np
from flatland.trajectories.trajectories import Trajectory

from ai4realnet_orchestrators.railway.abstract_test_runner_railway import AbtractTestRunnerRailway, DATA_VOLUME, SCENARIOS_VOLUME, DATA_VOLUME_MOUNTPATH, SCENARIOS_VOLUME_MOUNTPATH, RAILWAY_ORCHESTRATOR_RUN_LOCAL
logger = logging.getLogger(__name__)
   
# KPI-RS-058: Robustness to operator input (Railway)
class TestRunner_KPI_RS_058_Railway(AbtractTestRunnerRailway):
    """
    This implements KPI 058 (robustness to operator input). It assumes that submission_data_url points to a docker 
    image implemening the testing code of the KPI itself. How the docker image should be implemented is explained in 
    https://github.com/SebastiaanDePeuter/AI4RealNet_KPI-RS-058. A correctly implemented docker image will contain the 
    flatland simulator, the agent to be tested, and a random "human" intervention policy. Performance of the agent is 
    evaluated and key measurements are recorded into flatland's own data format for trajectory data and written to the 
    DATA_VOLUME. Robustness to an operator is measured by comparing the performance of the agent itself, with its 
    performance when part of its actions (10%) is overridden by the human policy.
    """

    def _run_scenario(self, scenario_id: str, submission_id: str, generate_policy_args, data_dir):
        self.exec(generate_policy_args, scenario_id, submission_id, data_dir)

        try:
            trajectory = Trajectory.load_existing(data_dir=Path(f"{DATA_VOLUME}/{data_dir}"), ep_id=scenario_id)
        except Exception as e: 
            logger.error(f"Failed to load trajectory: {e}")
            raise

        # return success_rate, punctuality
        if len(trajectory.trains_arrived) == 0:
            success_rate = 0.0
            punctuality = 0.0
        else:
            num_agents = trajectory.trains_rewards_dones_infos["agent_id"].max() + 1
            success_rate = trajectory.trains_arrived.iloc[0]["success_rate"]
            punctuality = mean_punctuality_aggregator([r for r in trajectory.trains_rewards_dones_infos.tail(num_agents)["reward"].to_list()])

        return success_rate, punctuality

    
    def run_scenario(self, scenario_id: str, submission_id: str):
        env_path = TestRunner_KPI_RS_058_Railway.load_scenario_data(scenario_id)
        data_dir = f"{submission_id}/{self.test_id}/{scenario_id}"
        basic_policy_args = [
            "--rewards-pkg", "flatland.envs.rewards", "--rewards-cls", "PunctualityRewards",
            "--ep-id", scenario_id,
            "--env-path", f"{SCENARIOS_VOLUME_MOUNTPATH}/{env_path}",
            "--snapshot-interval", "10",
        ]

        runner_data_dir_intervention = f"{data_dir}/intervened"
        intervened_policy_args = ["--data-dir", f"{DATA_VOLUME_MOUNTPATH}/{runner_data_dir_intervention}"] + basic_policy_args

        runner_data_dir_no_intervention = f"{data_dir}/non_intervened"
        non_intervened_policy_args = basic_policy_args + ["--data-dir", f"{DATA_VOLUME_MOUNTPATH}/{runner_data_dir_no_intervention}", "--policy", "intervened_policies.random_intervention_policy.NoOperatorInterventionPolicy"]

        success_rate, punctuality = self._run_scenario(scenario_id, submission_id, intervened_policy_args, runner_data_dir_intervention)
        base_success_rate, base_punctuality = self._run_scenario(scenario_id, submission_id, non_intervened_policy_args, runner_data_dir_no_intervention)

        self.upload_and_empty_local(submission_id=submission_id, scenario_id=scenario_id)

        return {
            'primary': success_rate - base_success_rate,
            'gained_success_rate': success_rate - base_success_rate,
            'gained_punctuality': punctuality - base_punctuality,
        }

    @staticmethod
    def load_scenario_data(scenario_id: str) -> str:
        return {
            "5a60713d-01f2-4d32-9867-21904629e254": "Test_00/Level_0.pkl",
            "0db72a40-43e8-477b-89b3-a7bd1224660a": "Test_00/Level_1.pkl",
            "7def3118-2e9c-4de7-8d61-f0e76fbeee5d": "Test_00/Level_2.pkl",
            "3ae60635-6995-4fb1-8309-61fded3d6fd8": "Test_00/Level_3.pkl",
            "eeef8445-723d-4740-b89f-4dbaf75f9ae6": "Test_00/Level_4.pkl",
            "94af1ed1-3686-4a9e-99f5-3a7ad908f125": "Test_00/Level_5.pkl",
            "8250d0e2-700e-4051-85c3-a8d0d95a5f0f": "Test_00/Level_6.pkl",
            "c58759a7-a64a-4cbf-970b-948bae0c2254": "Test_00/Level_7.pkl",
            "f94f517f-c0a4-4415-b726-186cdc75f9c6": "Test_00/Level_8.pkl",
            "c0e2c3e0-c171-48dd-a312-5de070e3f937": "Test_00/Level_9.pkl",
            "6fc5f67a-40fa-45ce-819e-35a85e08e560": "Test_02/Level_0.pkl",
            "66bce513-502c-43b4-a155-8a16c410a7c6": "Test_02/Level_1.pkl",
            "eff645bf-7ea8-490d-ae8a-ebb0d16a774c": "Test_02/Level_2.pkl",
            "8397e6d6-babc-469b-a239-7eabcbd510da": "Test_02/Level_3.pkl",
            "c359f13c-d222-4b04-ad0a-2bb30fb9da5f": "Test_02/Level_4.pkl",
            "97203764-6717-4ca6-bae9-c35c4eb38206": "Test_02/Level_5.pkl",
            "adc4bf52-096c-4369-a85f-c9bf4b86bc64": "Test_02/Level_6.pkl",
            "72f93d48-ecef-4bf7-9d97-cb008b47e566": "Test_02/Level_7.pkl",
            "b470667b-d9c9-4af4-b64e-c32102c34387": "Test_02/Level_8.pkl",
            "4aa9e1b8-8669-466e-b4b9-c7db2a098bec": "Test_02/Level_9.pkl",
            "8b308495-7ea6-4ddc-acb4-56eb5b3aec12": "Test_04/Level_0.pkl",
            "a8f69dc4-04a1-434a-ad97-27c745561b6a": "Test_04/Level_1.pkl",
            "8b244f56-50e1-411a-a7d8-a2b89dfab26e": "Test_04/Level_2.pkl",
            "8e6419c1-6470-4272-9c4b-43d9fe19dd3d": "Test_04/Level_3.pkl",
            "ec503b6e-3682-4dcd-9dc7-b194b67283d9": "Test_04/Level_4.pkl",
            "74fd9eab-d2e5-4222-8656-81fc2dde7c21": "Test_04/Level_5.pkl",
            "c16e54c1-33b2-45b8-95b0-33cc4f5400d5": "Test_04/Level_6.pkl",
            "c80effec-27b8-4103-b726-344a85f35407": "Test_04/Level_7.pkl",
            "9bec9335-3dd5-4d88-b2ac-c5d711bcab36": "Test_04/Level_8.pkl",
            "4a067d3c-75e6-4e91-a42d-cdf291016674": "Test_04/Level_9.pkl",
            "a5dff3c8-902e-4cb9-8466-d277d0ed4d67": "Test_06/Level_0.pkl",
            "aca25feb-6254-40b3-8d40-3c805797c69b": "Test_06/Level_1.pkl",
            "deb21442-0f94-4ff3-b78d-8d418415d646": "Test_06/Level_2.pkl",
            "6cf2cc89-d30e-4063-bced-051f3cdae92f": "Test_06/Level_3.pkl",
            "84bcbff5-346f-452c-87ab-08ceff6364f2": "Test_06/Level_4.pkl",
            "9acbe68e-2a45-420b-a142-34996dbcfb83": "Test_06/Level_5.pkl",
            "42786e4c-c80e-40f5-8237-bafc5f39979d": "Test_06/Level_6.pkl",
            "242a6240-b62c-48b4-a264-b6737e893fa5": "Test_06/Level_7.pkl",
            "3c38a1d3-2340-43ed-ac0b-4b76c6588b92": "Test_06/Level_8.pkl",
            "b89daede-405b-411a-a02b-ee32d7c9d020": "Test_06/Level_9.pkl",
            "9e0aac9e-ddf9-4575-bf1c-d08a923e15fa": "Test_08/Level_0.pkl",
            "befa97fb-2a74-4f2e-91c8-ea2879d08dcf": "Test_08/Level_1.pkl",
            "0cc18965-c967-4b58-ac7f-38a443b4cd16": "Test_08/Level_2.pkl",
            "d0f62f51-5a51-443b-bf7b-18e3d5b191dc": "Test_08/Level_3.pkl",
            "c2ebb179-0a2d-4e84-95be-2837be406716": "Test_08/Level_4.pkl",
            "3ac76f3c-f560-4666-af61-c693e4cd3ad4": "Test_08/Level_5.pkl",
            "484bbf93-bc67-4726-8b81-6c4ab608c861": "Test_08/Level_6.pkl",
            "11b19a5f-4d61-4b5d-980c-98cf0c16906a": "Test_08/Level_7.pkl",
            "edecaeb7-53d3-411a-a00c-2ce6226fde50": "Test_08/Level_8.pkl",
            "a43cb746-fa63-4d39-87cd-43a81fbf3a8e": "Test_08/Level_9.pkl",
        }[scenario_id]

def mean_punctuality_aggregator(scores):
    data = np.array(scores).transpose()
    # step rewards are (0,0), only take episode rewards with >0 agents in the second column:
    n_stops_on_time = data[0][data[1] > 0]
    n_stops = data[1][data[1] > 0]
    scenario_punctuality = np.divide(n_stops_on_time, n_stops)
    return np.mean(scenario_punctuality)
