import logging
import os
import time
from pathlib import Path
from typing import List

from flatland.trajectories.trajectories import Trajectory

from ai4realnet_orchestrators.fab_exec_utils import exec_with_logging
from ai4realnet_orchestrators.railway.abstract_test_runner_railway import AbtractTestRunnerRailway
from ai4realnet_orchestrators.railway.test_runner_kpi_pf_026_railway import mean_punctuality_aggregator

DATA_VOLUME = os.environ.get("DATA_VOLUME")
SCENARIOS_VOLUME = os.environ.get("SCENARIOS_VOLUME")
SUDO = os.environ.get("SUDO", "true").lower() == "true"

DATA_VOLUME_MOUNTPATH = os.environ.get("DATA_VOLUME_MOUNTPATH", "/app/data")
SCENARIOS_VOLUME_MOUNTPATH = os.environ.get("SCENARIOS_VOLUME_MOUNTPATH", "/app/scenarios")
RAILWAY_ORCHESTRATOR_RUN_LOCAL = os.environ.get("RAILWAY_ORCHESTRATOR_RUN_LOCAL", False)

logger = logging.getLogger(__name__)


# KPI-AF-051: AI-Agent Scalability Testing (Railway)
class TestRunner_KPI_AF_051_Railway(AbtractTestRunnerRailway):

  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self._last_simulation_time = None

  def exec(self, generate_policy_args: List[str], scenario_id: str, submission_id: str, subdir: str):
    """Override to time only the simulation, excluding Docker setup overhead."""
    if DATA_VOLUME:
      # Setup (not timed): mkdir, chmod, pull
      args = ["docker", "run", "--rm", "-v", f"{DATA_VOLUME}:/vol", "alpine:latest", "mkdir", "-p", f"/vol/{subdir}"]
      exec_with_logging(args if not SUDO else ["sudo"] + args)
      args = ["docker", "run", "--rm", "-v", f"{DATA_VOLUME}:/vol", "alpine:latest", "chmod", "-R", "a=rwx",
              f"/vol/{submission_id}/{self.test_id}/{scenario_id}"]
      exec_with_logging(args if not SUDO else ["sudo"] + args)
      args = ["docker", "pull", self.submission_data_url, ]
      exec_with_logging(args if not SUDO else ["sudo"] + args)

      # Simulation (timed)
      args = [
               "docker", "run",
               "--rm",
               "-v", f"{DATA_VOLUME}:{DATA_VOLUME_MOUNTPATH}",
               "-v", f"{SCENARIOS_VOLUME}:{SCENARIOS_VOLUME_MOUNTPATH}",
               "--security-opt=no-new-privileges",
               "-e", "PYTHONUNBUFFERED=1",
               "-e", "OAUTHLIB_INSECURE_TRANSPORT=1",
               self.submission_data_url,
               "flatland-trajectory-generate-from-policy",
             ] + generate_policy_args
      start = time.time()
      exec_with_logging(args if not SUDO else ["sudo"] + args, log_level_stdout=logging.DEBUG)
      self._last_simulation_time = time.time() - start

      # Cleanup (not timed)
      args = ["docker", "run", "--rm", "-v", f"{DATA_VOLUME}:/vol", "alpine:latest", "chmod", "-R", "a=rwx",
              f"/vol/{submission_id}/{self.test_id}/{scenario_id}"]
      exec_with_logging(args if not SUDO else ["sudo"] + args)
    else:
      from flatland.trajectories.policy_runner import generate_trajectory_from_policy
      Path(f"{DATA_VOLUME_MOUNTPATH}/{subdir}").mkdir(parents=True, exist_ok=False)
      try:
        start = time.time()
        generate_trajectory_from_policy(generate_policy_args)
        self._last_simulation_time = time.time() - start
      except SystemExit as e_info:
        self._last_simulation_time = time.time() - start
        if e_info.code != 0:
          print(e_info)
        assert e_info.code == 0

  def run_scenario(self, scenario_id: str, submission_id: str):
    env_path, complexity_metrics = TestRunner_KPI_AF_051_Railway.load_scenario_data(scenario_id)

    data_dir = f"{DATA_VOLUME_MOUNTPATH}/{submission_id}/{self.test_id}/{scenario_id}"
    generate_policy_args = [
      "--data-dir", data_dir,
      "--rewards-pkg", "flatland.envs.rewards", "--rewards-cls", "PunctualityRewards",
      "--ep-id", scenario_id,
      "--env-path", f"{SCENARIOS_VOLUME_MOUNTPATH}/{env_path}",
    ]
    self.exec(generate_policy_args, scenario_id, submission_id, f"{submission_id}/{self.test_id}/{scenario_id}")
    total_time_seconds = self._last_simulation_time
    logger.info(f"total_time_seconds: {total_time_seconds}")

    trajectory = Trajectory.load_existing(data_dir=Path(data_dir), ep_id=scenario_id)
    num_agents = trajectory.trains_rewards_dones_infos["agent_id"].max() + 1
    num_timesteps = len(trajectory.trains_rewards_dones_infos) // num_agents
    logger.info(f"num_timesteps: {num_timesteps}, num_agents: {num_agents}")

    avg_time_per_timestep = total_time_seconds / num_timesteps
    logger.info(f"avg_time_per_timestep: {avg_time_per_timestep}")

    # punctuality and success rate (same pattern as KPI-PF-026)
    df_trains_arrived = trajectory.trains_arrived
    assert len(df_trains_arrived) == 1
    success_rate = df_trains_arrived.iloc[0]["success_rate"]
    logger.info(f"success_rate: {success_rate}")

    agent_scores = trajectory.trains_rewards_dones_infos["reward"].to_list()
    punctuality = mean_punctuality_aggregator(agent_scores)
    logger.info(f"punctuality: {punctuality}")

    self.upload_and_empty_local(submission_id=submission_id, scenario_id=scenario_id)

    return {
      'primary': avg_time_per_timestep,
      'total_time_seconds': total_time_seconds,
      'num_timesteps': float(num_timesteps),
      'avg_time_per_timestep': avg_time_per_timestep,
      'punctuality': punctuality,
      'success_rate': success_rate,
      'n_cities': float(complexity_metrics['n_cities']),
      'n_agents': float(complexity_metrics['n_agents']),
    }

  @staticmethod
  def load_scenario_data(scenario_id: str):
    """
    70 scenarios: Test_{00,01,02,03,04,06,08} x Level_0-Level_9.
    """
    return {
      # Test_00: 2 cities, 7 agents, 30x30
      "bb6302f1-0dc2-43ed-976b-4e5d3126006a": ["Test_00/Level_0.pkl", {'n_cities': 2, 'n_agents': 7}],
      "f84dcf0c-4bde-460b-9139-ea76e3694267": ["Test_00/Level_1.pkl", {'n_cities': 2, 'n_agents': 7}],
      "89ea38d1-e42e-430e-8a72-f426f1cc0be7": ["Test_00/Level_2.pkl", {'n_cities': 2, 'n_agents': 7}],
      "ac3d32bf-2694-4405-953b-01849e7923ef": ["Test_00/Level_3.pkl", {'n_cities': 2, 'n_agents': 7}],
      "30286226-29a3-4aa6-8243-562b88967d76": ["Test_00/Level_4.pkl", {'n_cities': 2, 'n_agents': 7}],
      "18276866-5a94-412b-b09c-9cac2ca5add0": ["Test_00/Level_5.pkl", {'n_cities': 2, 'n_agents': 7}],
      "02e163b8-d8a3-44cb-9fb0-65501dfa35b7": ["Test_00/Level_6.pkl", {'n_cities': 2, 'n_agents': 7}],
      "ab2b11c8-66f4-47c3-9cd3-f765eb772dc7": ["Test_00/Level_7.pkl", {'n_cities': 2, 'n_agents': 7}],
      "f3ae4180-86f3-409a-a51e-c1deb7e005cd": ["Test_00/Level_8.pkl", {'n_cities': 2, 'n_agents': 7}],
      "7a3ae3eb-b783-44a3-80d4-aa9cb0bd55fb": ["Test_00/Level_9.pkl", {'n_cities': 2, 'n_agents': 7}],
      # Test_01: 2 cities, 7 agents, 30x30
      "cff75f1a-8ea2-4f1d-b516-60dd0d625fe1": ["Test_01/Level_0.pkl", {'n_cities': 2, 'n_agents': 7}],
      "aa4fd74f-4680-405b-a184-c9392f9218e3": ["Test_01/Level_1.pkl", {'n_cities': 2, 'n_agents': 7}],
      "01a82553-8d2c-4f84-94df-ccb9f3250734": ["Test_01/Level_2.pkl", {'n_cities': 2, 'n_agents': 7}],
      "70316412-5480-44ca-9c2b-c51426b0390e": ["Test_01/Level_3.pkl", {'n_cities': 2, 'n_agents': 7}],
      "60a6acda-9a1a-4a0a-8c04-75de02304713": ["Test_01/Level_4.pkl", {'n_cities': 2, 'n_agents': 7}],
      "db614cef-8b86-467d-a638-64c25a91ec78": ["Test_01/Level_5.pkl", {'n_cities': 2, 'n_agents': 7}],
      "43b053bb-5e9b-4538-a490-fee839344203": ["Test_01/Level_6.pkl", {'n_cities': 2, 'n_agents': 7}],
      "e01032e4-2047-455a-a329-175a40a8de24": ["Test_01/Level_7.pkl", {'n_cities': 2, 'n_agents': 7}],
      "3b68eeb2-96f6-4a87-8a2f-5decaf3cb3f0": ["Test_01/Level_8.pkl", {'n_cities': 2, 'n_agents': 7}],
      "fdd89c15-3f8d-4381-9cd7-e8b773d06997": ["Test_01/Level_9.pkl", {'n_cities': 2, 'n_agents': 7}],
      # Test_02: 3 cities, 20 agents, 30x30
      "1d8f2bda-38a4-41de-a614-291b9e4697e4": ["Test_02/Level_0.pkl", {'n_cities': 3, 'n_agents': 20}],
      "7277b987-4cc2-4cb5-a308-bb226c832747": ["Test_02/Level_1.pkl", {'n_cities': 3, 'n_agents': 20}],
      "62e20486-eb7f-49d9-a9dc-7aa00fdfefb0": ["Test_02/Level_2.pkl", {'n_cities': 3, 'n_agents': 20}],
      "ae7a8233-8a80-496a-a2b3-0afd9a28ebe6": ["Test_02/Level_3.pkl", {'n_cities': 3, 'n_agents': 20}],
      "2b4b92d2-6871-4c20-ad58-11dc51718379": ["Test_02/Level_4.pkl", {'n_cities': 3, 'n_agents': 20}],
      "86f360de-8c4d-44d0-b089-3259a91dc3ea": ["Test_02/Level_5.pkl", {'n_cities': 3, 'n_agents': 20}],
      "f4b1aaeb-a498-428e-8f8f-2ed07aee0641": ["Test_02/Level_6.pkl", {'n_cities': 3, 'n_agents': 20}],
      "e5968696-5497-496b-8fe4-f40a837f7129": ["Test_02/Level_7.pkl", {'n_cities': 3, 'n_agents': 20}],
      "dec5cd4e-10b7-4a7e-a803-10e50badaaf5": ["Test_02/Level_8.pkl", {'n_cities': 3, 'n_agents': 20}],
      "139b31bd-22e4-495c-8e21-5e6e34cd5a20": ["Test_02/Level_9.pkl", {'n_cities': 3, 'n_agents': 20}],
      # Test_03: 3 cities, 50 agents, 30x35
      "9b603e03-3e2e-4366-8127-96307d3b2ce1": ["Test_03/Level_0.pkl", {'n_cities': 3, 'n_agents': 50}],
      "54601145-edd9-469f-8180-245e26dff069": ["Test_03/Level_1.pkl", {'n_cities': 3, 'n_agents': 50}],
      "34fa69f7-e0f7-4fd4-adf0-2ed8a47d6abc": ["Test_03/Level_2.pkl", {'n_cities': 3, 'n_agents': 50}],
      "51081d92-2ff1-40a4-b557-38215c125051": ["Test_03/Level_3.pkl", {'n_cities': 3, 'n_agents': 50}],
      "fef8ec79-80da-4039-9484-6ec49a29263e": ["Test_03/Level_4.pkl", {'n_cities': 3, 'n_agents': 50}],
      "d9d80121-bf6c-41ee-bc18-dade0e853ada": ["Test_03/Level_5.pkl", {'n_cities': 3, 'n_agents': 50}],
      "5832961b-942f-4d33-8614-c6dd4861ef46": ["Test_03/Level_6.pkl", {'n_cities': 3, 'n_agents': 50}],
      "e44e2b83-ad54-4e9c-a2ea-c23c1a249c54": ["Test_03/Level_7.pkl", {'n_cities': 3, 'n_agents': 50}],
      "8ca33f45-3839-4ccc-aa88-146b41fee9f3": ["Test_03/Level_8.pkl", {'n_cities': 3, 'n_agents': 50}],
      "e15fa3d0-0da3-4513-a5bd-6082806039a3": ["Test_03/Level_9.pkl", {'n_cities': 3, 'n_agents': 50}],
      # Test_04: 5 cities, 80 agents, 35x30
      "e28dc7e5-03ae-4687-ba37-c7ed5914c901": ["Test_04/Level_0.pkl", {'n_cities': 5, 'n_agents': 80}],
      "ef52b0f5-a147-4333-9817-fbd7e53143ee": ["Test_04/Level_1.pkl", {'n_cities': 5, 'n_agents': 80}],
      "45b93b12-57cc-40ff-b277-82de8ceaec32": ["Test_04/Level_2.pkl", {'n_cities': 5, 'n_agents': 80}],
      "ae557fe8-2155-42b3-8d17-2e9de47dda4b": ["Test_04/Level_3.pkl", {'n_cities': 5, 'n_agents': 80}],
      "3b2f7caf-2e32-4db9-8377-e01f50e436c2": ["Test_04/Level_4.pkl", {'n_cities': 5, 'n_agents': 80}],
      "9ae1a2b5-fe89-4027-b1d8-8c3888862a5e": ["Test_04/Level_5.pkl", {'n_cities': 5, 'n_agents': 80}],
      "9c221d41-fda7-409e-9ceb-a0f94018a92c": ["Test_04/Level_6.pkl", {'n_cities': 5, 'n_agents': 80}],
      "7da98e75-8c84-4cfc-98f4-0fedf1aec08f": ["Test_04/Level_7.pkl", {'n_cities': 5, 'n_agents': 80}],
      "7b42a1cc-ce70-4d9a-804f-ac9027a1ee48": ["Test_04/Level_8.pkl", {'n_cities': 5, 'n_agents': 80}],
      "48caf228-64f6-4b03-ad20-5a34cf8dd2ee": ["Test_04/Level_9.pkl", {'n_cities': 5, 'n_agents': 80}],
      # Test_06: 9 cities, 80 agents, 40x60
      "f262ffb3-86b9-4db8-8657-f4a96915cb83": ["Test_06/Level_0.pkl", {'n_cities': 9, 'n_agents': 80}],
      "b022b575-103d-4ded-8da5-2e9a8f686da6": ["Test_06/Level_1.pkl", {'n_cities': 9, 'n_agents': 80}],
      "ba5308f8-7e12-4c6e-8e4f-42f6280f537c": ["Test_06/Level_2.pkl", {'n_cities': 9, 'n_agents': 80}],
      "fec35ca6-d093-4081-8cf1-2f3b8f445bed": ["Test_06/Level_3.pkl", {'n_cities': 9, 'n_agents': 80}],
      "a410e586-219d-402f-9e34-6a1720ae46bc": ["Test_06/Level_4.pkl", {'n_cities': 9, 'n_agents': 80}],
      "a82beae8-9b1a-4a00-bea2-4891b56f3014": ["Test_06/Level_5.pkl", {'n_cities': 9, 'n_agents': 80}],
      "14a20672-ea9c-4205-961a-4f2a3585eea2": ["Test_06/Level_6.pkl", {'n_cities': 9, 'n_agents': 80}],
      "e7f82820-1caf-4b4f-ae70-8ea4d95dcb0c": ["Test_06/Level_7.pkl", {'n_cities': 9, 'n_agents': 80}],
      "989cc7e2-1d39-4334-8130-b704fd7c6c7b": ["Test_06/Level_8.pkl", {'n_cities': 9, 'n_agents': 80}],
      "63b933c7-b63c-4ab7-b602-69aa5f91aec9": ["Test_06/Level_9.pkl", {'n_cities': 9, 'n_agents': 80}],
      # Test_08: 17 cities, 80 agents, 60x60
      "f185083b-3f74-4221-b5c8-7b2e561ae2e4": ["Test_08/Level_0.pkl", {'n_cities': 17, 'n_agents': 80}],
      "dc4da56a-bf2d-4351-b280-d418736844de": ["Test_08/Level_1.pkl", {'n_cities': 17, 'n_agents': 80}],
      "02183d8b-2328-4467-81e9-97afde5618c9": ["Test_08/Level_2.pkl", {'n_cities': 17, 'n_agents': 80}],
      "d85cbc22-951e-4926-94f9-9c7a703b54eb": ["Test_08/Level_3.pkl", {'n_cities': 17, 'n_agents': 80}],
      "d4b12805-017e-46d9-8fe7-220569a21477": ["Test_08/Level_4.pkl", {'n_cities': 17, 'n_agents': 80}],
      "2f5a8e75-521e-4398-bfd2-e7ff7c9e0be4": ["Test_08/Level_5.pkl", {'n_cities': 17, 'n_agents': 80}],
      "701c5b31-06d0-4e54-82b9-08e1612f1042": ["Test_08/Level_6.pkl", {'n_cities': 17, 'n_agents': 80}],
      "6acbd77e-18a8-41c9-af81-1ff0ac1a9b0f": ["Test_08/Level_7.pkl", {'n_cities': 17, 'n_agents': 80}],
      "20ef1912-26a4-48c1-ad1b-f08c95b144e2": ["Test_08/Level_8.pkl", {'n_cities': 17, 'n_agents': 80}],
      "890c8007-a763-4a94-86d5-28d8c11c573f": ["Test_08/Level_9.pkl", {'n_cities': 17, 'n_agents': 80}],
    }[scenario_id]
