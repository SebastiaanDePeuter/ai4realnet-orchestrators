import ast
import logging
import os
from collections import defaultdict
from pathlib import Path

import numpy as np
from flatland.envs.step_utils.states import TrainState
from flatland.trajectories.trajectories import Trajectory

from ai4realnet_orchestrators.railway.abstract_test_runner_railway import AbtractTestRunnerRailway

DATA_VOLUME_MOUNTPATH = os.environ.get("DATA_VOLUME_MOUNTPATH", "/app/data")
SCENARIOS_VOLUME_MOUNTPATH = os.environ.get("SCENARIOS_VOLUME_MOUNTPATH", "/app/scenarios")

logger = logging.getLogger(__name__)


# KPI-NF-045: Network Impact Propagation (Railway)
class TestRunner_KPI_NF_045_Railway(AbtractTestRunnerRailway):

  def run_scenario(self, scenario_id: str, submission_id: str):
    env_path, earliest_malfunction = TestRunner_KPI_NF_045_Railway.load_scenario_data(scenario_id)
    # here you would implement the logic to run the test for the scenario
    # data and other stuff initialized in the init method can be used here
    # for demonstration, we return a dummy result
    data_dir_no_malfunction = f"{DATA_VOLUME_MOUNTPATH}/{submission_id}/{self.test_id}/{scenario_id}/no_malfunction"
    generate_policy_args_no_malfunction = [
      "--data-dir", data_dir_no_malfunction,
      "--rewards-pkg", "flatland.envs.rewards", "--rewards-cls", "PunctualityRewards",
      # TODO https://github.com/flatland-association/flatland-rl/issues/278 disable malfunction generator and replace with effects generator - a bit hacky for now, clean up later...
      "--malfunction-interval", "-1",
      "--effects-generator-pkg", "flatland.core.effects_generator", "--effects-generator-cls", "EffectsGenerator",
      "--ep-id", scenario_id,
      "--env-path", f"{SCENARIOS_VOLUME_MOUNTPATH}/{env_path}"
    ]
    self.exec(generate_policy_args_no_malfunction, scenario_id, submission_id, f"{submission_id}/{self.test_id}/{scenario_id}/no_malfunction")

    data_dir_with_malfunction = f"{DATA_VOLUME_MOUNTPATH}/{submission_id}/{self.test_id}/{scenario_id}/with_malfunction"
    generate_policy_args_one_malfunction = [
      "--data-dir", data_dir_with_malfunction,
      "--rewards-pkg", "flatland.envs.rewards", "--rewards-cls", "PunctualityRewards",
      # TODO https://github.com/flatland-association/flatland-rl/issues/278 disable malfunction generator and replace with effects generator - a bit hacky for now, clean up later...
      "--malfunction-interval", "-1",
      "--effects-generator-pkg", "flatland.envs.malfunction_effects_generators", "--effects-generator-cls",
      "ConditionalMalfunctionEffectsGenerator",
      "--effects-generator-kwargs", "earliest_malfunction", f"{earliest_malfunction}",
      "--effects-generator-kwargs", "max_num_malfunctions", "1",
      "--effects-generator-kwargs", "min_duration", "20",
      "--effects-generator-kwargs", "max_duration", "50",
      "--effects-generator-kwargs", "malfunction_rate", "1.0",
      "--effects-generator-kwargs", "condition_pkg", "flatland.envs.malfunction_effects_generators",
      "--effects-generator-kwargs", "condition_cls", "on_map_state_condition",
      "--ep-id", scenario_id,
      "--env-path", f"{SCENARIOS_VOLUME_MOUNTPATH}/{env_path}",
      "--snapshot-interval", "10",
    ]
    self.exec(generate_policy_args_one_malfunction, scenario_id, submission_id, f"{submission_id}/{self.test_id}/{scenario_id}/with_malfunction")

    # no malfunction
    trajectory_no_malfunction = Trajectory.load_existing(data_dir=Path(data_dir_no_malfunction), ep_id=scenario_id)
    num_agents = trajectory_no_malfunction.trains_rewards_dones_infos["agent_id"].max() + 1
    for _, r in trajectory_no_malfunction.trains_rewards_dones_infos.iterrows():
      assert r["info"]["malfunction"] == 0
    tail_no_malfunction = trajectory_no_malfunction.trains_rewards_dones_infos.tail(num_agents)
    assert tail_no_malfunction["done"].values.sum() == num_agents
    logger.info(f"tail_no_malfunction.reward {tail_no_malfunction["reward"].to_list()}")
    punctuality_tuples_no_malfunction = [ast.literal_eval(r) if isinstance(r, str) else r for r in tail_no_malfunction["reward"].to_list()]
    logger.info(f"punctuality_tuples_no_malfunction {punctuality_tuples_no_malfunction}")

    betroffen1 = [num_punctual != num_waypoints for num_punctual, num_waypoints in punctuality_tuples_no_malfunction]
    num_betroffen1 = np.sum(betroffen1)
    logger.info(f"num_betroffen1 {num_betroffen1}")

    trajectory_with_malfunction = Trajectory.load_existing(data_dir=Path(data_dir_with_malfunction), ep_id=scenario_id)
    malfunction_agents = defaultdict(list)
    for _, r in trajectory_with_malfunction.trains_rewards_dones_infos.iterrows():
      if r["info"]["malfunction"] > 0:
        malfunction_agents[r["agent_id"]].append(r["info"]["malfunction"])
    assert len(malfunction_agents.keys()) == 1
    assert list(malfunction_agents.values())[0][0] >= 20
    assert list(malfunction_agents.values())[0][0] <= 50
    logger.info(f"malfunction_agents {malfunction_agents}")

    # with malfunction
    tail_with_malfunction = trajectory_with_malfunction.trains_rewards_dones_infos.tail(num_agents)
    logger.info(f"tail_with_malfunction.reward {tail_with_malfunction["reward"].to_list()}")
    assert tail_with_malfunction["done"].values.sum() == num_agents
    punctuality_tuples_with_malfunction = [ast.literal_eval(r) if isinstance(r, str) else r for r in tail_with_malfunction["reward"].to_list()]
    logger.info(f"punctuality_tuples_with_malfunction {punctuality_tuples_with_malfunction}")

    malfunction_states = set()
    for _, r in trajectory_with_malfunction.trains_rewards_dones_infos.iterrows():
      if r["info"]["malfunction"] > 0:
        malfunction_states.add(TrainState(r["info"]["state"]).name)
    logger.info(f"malfunction_states {malfunction_states}")
    assert len(malfunction_states) == 1

    betroffen2 = [num_punctual != num_waypoints for num_punctual, num_waypoints in punctuality_tuples_with_malfunction]
    num_betroffen2 = np.sum(betroffen2)
    logger.info(f"num_betroffen2 {num_betroffen2}")
    unclipped = 1 - ((num_betroffen2 - num_betroffen1) / num_agents)
    nip = np.clip(unclipped, 0, 1)
    logger.info(f"network impact propagation {nip} np.clip({unclipped}, 0, 1) = np.clip(1 - ({num_betroffen2}-{num_betroffen1}) / {num_agents}, 0, 1)")

    assert nip >= 0
    assert nip <= 1

    success_rate_1 = trajectory_no_malfunction.trains_arrived.iloc[0]["success_rate"]
    logger.info(f"success rate no malfunction: {success_rate_1}")
    success_rate_2 = trajectory_with_malfunction.trains_arrived.iloc[0]["success_rate"]
    logger.info(f"success rate no malfunction: {success_rate_2}")

    punctuality_1 = mean_punctuality_aggregator(punctuality_tuples_with_malfunction)
    logger.info(f"punctuality no malfunction: {punctuality_1}")

    punctuality_2 = mean_punctuality_aggregator(punctuality_tuples_with_malfunction)
    logger.info(f"punctuality no malfunction: {punctuality_2}")

    self.upload_and_empty_local(submission_id=submission_id, scenario_id=scenario_id)

    return {
      'network_impact_propagation': nip,
      'success_rate_1': success_rate_1,
      'punctuality_1': punctuality_1,
      'success_rate_2': success_rate_2,
      'punctuality_2': punctuality_2,
    }

  @staticmethod
  def load_scenario_data(scenario_id: str) -> str:
    return {
      'f84dcf0c-4bde-460b-9139-ea76e3694267': ['Test_00/Level_1.pkl', 22],
      '30286226-29a3-4aa6-8243-562b88967d76': ['Test_00/Level_4.pkl', 38],
      'ab2b11c8-66f4-47c3-9cd3-f765eb772dc7': ['Test_00/Level_7.pkl', 13],
      'cff75f1a-8ea2-4f1d-b516-60dd0d625fe1': ['Test_01/Level_0.pkl', 20],
      '70316412-5480-44ca-9c2b-c51426b0390e': ['Test_01/Level_3.pkl', 20],
      '43b053bb-5e9b-4538-a490-fee839344203': ['Test_01/Level_6.pkl', 22],
      'fdd89c15-3f8d-4381-9cd7-e8b773d06997': ['Test_01/Level_9.pkl', 25],
      '62e20486-eb7f-49d9-a9dc-7aa00fdfefb0': ['Test_02/Level_2.pkl', 47],
      '86f360de-8c4d-44d0-b089-3259a91dc3ea': ['Test_02/Level_5.pkl', 35],
      'dec5cd4e-10b7-4a7e-a803-10e50badaaf5': ['Test_02/Level_8.pkl', 33],
      '54601145-edd9-469f-8180-245e26dff069': ['Test_03/Level_1.pkl', 47],
      'fef8ec79-80da-4039-9484-6ec49a29263e': ['Test_03/Level_4.pkl', 23],
      'e44e2b83-ad54-4e9c-a2ea-c23c1a249c54': ['Test_03/Level_7.pkl', 27],
      'e28dc7e5-03ae-4687-ba37-c7ed5914c901': ['Test_04/Level_0.pkl', 55],
      'ae557fe8-2155-42b3-8d17-2e9de47dda4b': ['Test_04/Level_3.pkl', 44],
      '9c221d41-fda7-409e-9ceb-a0f94018a92c': ['Test_04/Level_6.pkl', 48],
      '48caf228-64f6-4b03-ad20-5a34cf8dd2ee': ['Test_04/Level_9.pkl', 50],
      '401a5b54-feb8-4eaf-92c3-426cb2f221ef': ['Test_05/Level_2.pkl', 71],
      '2a631e96-a912-4b27-b82e-57ca3dd4aacf': ['Test_05/Level_5.pkl', 55],
      '2060f4fe-4f43-4095-b14f-a3c8ce312a42': ['Test_05/Level_8.pkl', 73],
      'b022b575-103d-4ded-8da5-2e9a8f686da6': ['Test_06/Level_1.pkl', 79],
      'a410e586-219d-402f-9e34-6a1720ae46bc': ['Test_06/Level_4.pkl', 77],
      'e7f82820-1caf-4b4f-ae70-8ea4d95dcb0c': ['Test_06/Level_7.pkl', 112],
      '05826982-694c-4ba3-817f-979d69942d36': ['Test_07/Level_0.pkl', 64],
      '320b95b2-84d6-4827-ba37-0de57a1e6360': ['Test_07/Level_3.pkl', 78],
      'be970bda-465f-42b4-9223-c4ba742b24ea': ['Test_07/Level_6.pkl', 89],
      '23ce8d72-6c32-45b4-a04e-34e029eb509e': ['Test_07/Level_9.pkl', 111],
      '02183d8b-2328-4467-81e9-97afde5618c9': ['Test_08/Level_2.pkl', 131],
      '2f5a8e75-521e-4398-bfd2-e7ff7c9e0be4': ['Test_08/Level_5.pkl', 96],
      '20ef1912-26a4-48c1-ad1b-f08c95b144e2': ['Test_08/Level_8.pkl', 90],
      '848a39f3-e96b-4c41-83b8-78a2eb99403f': ['Test_09/Level_1.pkl', 159],
      '6129f9a8-853c-459e-8ac2-aa7cbc65802e': ['Test_09/Level_4.pkl', 173],
      'e86883e7-53be-4b58-99cb-efd5f23cdb5b': ['Test_09/Level_7.pkl', 124],
      '566f099a-2957-4fc9-8e13-f9564311ba33': ['Test_10/Level_0.pkl', 146],
      '2ff0136f-7995-4134-8e4e-9fd92da46ea8': ['Test_10/Level_3.pkl', 174],
      '387dafe7-97ad-4b37-88e4-10785748e04d': ['Test_10/Level_6.pkl', 154],
      '5a52729c-4f09-4f58-852d-48239e9ce217': ['Test_10/Level_9.pkl', 149],
      '49257861-320d-4aa2-aa62-b9c4e1751791': ['Test_11/Level_2.pkl', 291],
      '220bf75b-d271-4c46-bde2-67b313f10d13': ['Test_11/Level_5.pkl', 177],
      '492c01f3-a7d4-46e5-b140-e82a22b13cdf': ['Test_11/Level_8.pkl', 175],
      'd9f53db5-8cec-4ec4-8d62-d116cb198811': ['Test_12/Level_1.pkl', 236],
      '7a335c56-af9e-4562-a614-bfe84a75951f': ['Test_12/Level_4.pkl', 304],
      'bb2b5fe3-92ed-4270-9235-dafbfffa2d03': ['Test_12/Level_7.pkl', 253],
      '42b7c577-c4ba-43e0-8ff9-d1839d86a06c': ['Test_13/Level_0.pkl', 275],
      'ef64aa0d-ba0d-4af4-abf3-22395713f0d7': ['Test_13/Level_3.pkl', 438],
      '56e0d8cc-b7f0-4e9f-ae77-a7322cc9a2a5': ['Test_13/Level_6.pkl', 341],
      '8540245e-c841-4caa-aeb7-5dd6955bc43d': ['Test_13/Level_9.pkl', 268],
      'c7864f2d-ae67-42a9-bddb-59b6933b0c1c': ['Test_14/Level_2.pkl', 264],
      '87e0d07c-cd5f-4506-b112-619d298ce924': ['Test_14/Level_5.pkl', 293],
      '97bbc19c-de0c-4deb-838d-5675d9525eb8': ['Test_14/Level_8.pkl', 258]
    }[scenario_id]


def mean_punctuality_aggregator(scores):
  data = np.array(scores).transpose()
  scenario_punctuality = data[0] / data[1]
  return np.mean(scenario_punctuality)


def gen_earlieast_malfunction_snippet():
  from pathlib import Path
  from flatland.envs.persistence import RailEnvPersister
  for k, v in {
    'f84dcf0c-4bde-460b-9139-ea76e3694267': 'Test_00/Level_1.pkl',
    '30286226-29a3-4aa6-8243-562b88967d76': 'Test_00/Level_4.pkl',
    'ab2b11c8-66f4-47c3-9cd3-f765eb772dc7': 'Test_00/Level_7.pkl',
    'cff75f1a-8ea2-4f1d-b516-60dd0d625fe1': 'Test_01/Level_0.pkl',
    '70316412-5480-44ca-9c2b-c51426b0390e': 'Test_01/Level_3.pkl',
    '43b053bb-5e9b-4538-a490-fee839344203': 'Test_01/Level_6.pkl',
    'fdd89c15-3f8d-4381-9cd7-e8b773d06997': 'Test_01/Level_9.pkl',
    '62e20486-eb7f-49d9-a9dc-7aa00fdfefb0': 'Test_02/Level_2.pkl',
    '86f360de-8c4d-44d0-b089-3259a91dc3ea': 'Test_02/Level_5.pkl',
    'dec5cd4e-10b7-4a7e-a803-10e50badaaf5': 'Test_02/Level_8.pkl',
    '54601145-edd9-469f-8180-245e26dff069': 'Test_03/Level_1.pkl',
    'fef8ec79-80da-4039-9484-6ec49a29263e': 'Test_03/Level_4.pkl',
    'e44e2b83-ad54-4e9c-a2ea-c23c1a249c54': 'Test_03/Level_7.pkl',
    'e28dc7e5-03ae-4687-ba37-c7ed5914c901': 'Test_04/Level_0.pkl',
    'ae557fe8-2155-42b3-8d17-2e9de47dda4b': 'Test_04/Level_3.pkl',
    '9c221d41-fda7-409e-9ceb-a0f94018a92c': 'Test_04/Level_6.pkl',
    '48caf228-64f6-4b03-ad20-5a34cf8dd2ee': 'Test_04/Level_9.pkl',
    '401a5b54-feb8-4eaf-92c3-426cb2f221ef': 'Test_05/Level_2.pkl',
    '2a631e96-a912-4b27-b82e-57ca3dd4aacf': 'Test_05/Level_5.pkl',
    '2060f4fe-4f43-4095-b14f-a3c8ce312a42': 'Test_05/Level_8.pkl',
    'b022b575-103d-4ded-8da5-2e9a8f686da6': 'Test_06/Level_1.pkl',
    'a410e586-219d-402f-9e34-6a1720ae46bc': 'Test_06/Level_4.pkl',
    'e7f82820-1caf-4b4f-ae70-8ea4d95dcb0c': 'Test_06/Level_7.pkl',
    '05826982-694c-4ba3-817f-979d69942d36': 'Test_07/Level_0.pkl',
    '320b95b2-84d6-4827-ba37-0de57a1e6360': 'Test_07/Level_3.pkl',
    'be970bda-465f-42b4-9223-c4ba742b24ea': 'Test_07/Level_6.pkl',
    '23ce8d72-6c32-45b4-a04e-34e029eb509e': 'Test_07/Level_9.pkl',
    '02183d8b-2328-4467-81e9-97afde5618c9': 'Test_08/Level_2.pkl',
    '2f5a8e75-521e-4398-bfd2-e7ff7c9e0be4': 'Test_08/Level_5.pkl',
    '20ef1912-26a4-48c1-ad1b-f08c95b144e2': 'Test_08/Level_8.pkl',
    '848a39f3-e96b-4c41-83b8-78a2eb99403f': 'Test_09/Level_1.pkl',
    '6129f9a8-853c-459e-8ac2-aa7cbc65802e': 'Test_09/Level_4.pkl',
    'e86883e7-53be-4b58-99cb-efd5f23cdb5b': 'Test_09/Level_7.pkl',
    '566f099a-2957-4fc9-8e13-f9564311ba33': 'Test_10/Level_0.pkl',
    '2ff0136f-7995-4134-8e4e-9fd92da46ea8': 'Test_10/Level_3.pkl',
    '387dafe7-97ad-4b37-88e4-10785748e04d': 'Test_10/Level_6.pkl',
    '5a52729c-4f09-4f58-852d-48239e9ce217': 'Test_10/Level_9.pkl',
    '49257861-320d-4aa2-aa62-b9c4e1751791': 'Test_11/Level_2.pkl',
    '220bf75b-d271-4c46-bde2-67b313f10d13': 'Test_11/Level_5.pkl',
    '492c01f3-a7d4-46e5-b140-e82a22b13cdf': 'Test_11/Level_8.pkl',
    'd9f53db5-8cec-4ec4-8d62-d116cb198811': 'Test_12/Level_1.pkl',
    '7a335c56-af9e-4562-a614-bfe84a75951f': 'Test_12/Level_4.pkl',
    'bb2b5fe3-92ed-4270-9235-dafbfffa2d03': 'Test_12/Level_7.pkl',
    '42b7c577-c4ba-43e0-8ff9-d1839d86a06c': 'Test_13/Level_0.pkl',
    'ef64aa0d-ba0d-4af4-abf3-22395713f0d7': 'Test_13/Level_3.pkl',
    '56e0d8cc-b7f0-4e9f-ae77-a7322cc9a2a5': 'Test_13/Level_6.pkl',
    '8540245e-c841-4caa-aeb7-5dd6955bc43d': 'Test_13/Level_9.pkl',
    'c7864f2d-ae67-42a9-bddb-59b6933b0c1c': 'Test_14/Level_2.pkl',
    '87e0d07c-cd5f-4506-b112-619d298ce924': 'Test_14/Level_5.pkl',
    '97bbc19c-de0c-4deb-838d-5675d9525eb8': 'Test_14/Level_8.pkl'
  }.items():
    p = Path(f"../flatland-scenarios/scenarios/{v}")
    env, _ = RailEnvPersister.load_new(p)
    earliest_malfunction = int(env._max_episode_steps * 0.1)
    print(f'"{k}": ["{v}", {earliest_malfunction}],')
