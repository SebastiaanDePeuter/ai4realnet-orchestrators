import logging
import os
import time
from pathlib import Path

from ai4realnet_orchestrators.test_runner import TestRunner
from grid2evaluate.operation_score_kpi import OperationScoreKpi

POWERGRID_ORCHESTRATOR_RUN_LOCAL = os.environ.get("POWERGRID_ORCHESTRATOR_RUN_LOCAL", "True")
ENV_BASE_PATH = os.environ.get("ENV_PATH", "/opt/ai4realnet/grid2op-scenario")
RECORDER_PATH = os.environ.get("RECORDER_PATH", "/tmp/ai4realnet")

logger = logging.getLogger(__name__)


# KPI-AF-051: AI-Agent Scalability Testing (Power Grid)
class TestRunner_KPI_AF_051_Power_Grid(TestRunner):

    def __init__(self, *args, agent=None, **kwargs):
        super().__init__(*args, **kwargs)
        self._agent = agent

    def run_scenario(self, scenario_id: str, submission_id: str):
        env_name = TestRunner_KPI_AF_051_Power_Grid.load_scenario_data(scenario_id)

        input_directory = Path(ENV_BASE_PATH, env_name)
        record_directory = Path(RECORDER_PATH, env_name, scenario_id)
        os.makedirs(record_directory, exist_ok=True)

        if eval(POWERGRID_ORCHESTRATOR_RUN_LOCAL):
            total_time_seconds, powerflow_time = self._run_local(input_directory, record_directory)
        else:
            from ai4realnet_orchestrators.fab_exec_utils import exec_with_logging
            import subprocess
            pull_result = subprocess.run(["docker", "pull", self.submission_data_url], capture_output=True)
            if pull_result.returncode != 0:
                logger.info(f"docker pull skipped (local image): {self.submission_data_url}")

            args = [
                "docker", "run",
                "--rm",
                "-e", f"ENV_NAME={env_name}",
                "-e", f"SCENARIO_ID={scenario_id}",
                "-v", f"{ENV_BASE_PATH}:/opt/ai4realnet/grid2op-scenario",
                "-v", f"{RECORDER_PATH}:/tmp/ai4realnet",
                self.submission_data_url,
            ]
            start = time.time()
            exec_with_logging(args)
            total_time_seconds = time.time() - start
            powerflow_time = -1.0

        logger.info(f"total_time_seconds: {total_time_seconds}")

        num_timesteps = self._count_timesteps(record_directory)
        operational_score = self._compute_operational_score(record_directory)
        avg_time_per_timestep = total_time_seconds / num_timesteps if num_timesteps > 0 else 0.0
        agent_time = total_time_seconds - powerflow_time if powerflow_time >= 0 else -1.0

        logger.info(f"num_timesteps: {num_timesteps}")
        logger.info(f"avg_time_per_timestep: {avg_time_per_timestep}")
        logger.info(f"powerflow_time: {powerflow_time}")
        logger.info(f"agent_time: {agent_time}")
        logger.info(f"operational_score: {operational_score}")

        n_sub, n_line = self._get_complexity_metrics(record_directory)

        return {
            'primary': avg_time_per_timestep,
            'total_time_seconds': total_time_seconds,
            'num_timesteps': float(num_timesteps),
            'avg_time_per_timestep': avg_time_per_timestep,
            'powerflow_time': powerflow_time,
            'agent_time': agent_time,
            'operational_score': operational_score,
            'n_sub': float(n_sub),
            'n_line': float(n_line),
        }

    def _run_local(self, input_directory, record_directory):
        import grid2op
        from lightsim2grid import LightSimBackend
        from grid2op.Environment.EnvRecorder import EnvRecorder
        from grid2op.Agent import DoNothingAgent

        env = grid2op.make(
            input_directory,
            test=True,
            backend=LightSimBackend(),
            _add_to_name="test"
        )
        agent = self._agent or DoNothingAgent(env.action_space)
        with EnvRecorder(env, record_directory) as env_rec:
            obs = env_rec.reset()
            reward = 0
            done = False
            start = time.time()
            while not done:
                action = agent.act(obs, reward, done)
                obs, reward, done, info = env_rec.step(action)
            total_time_seconds = time.time() - start
            powerflow_time = env._time_powerflow
        return total_time_seconds, powerflow_time

    def _get_complexity_metrics(self, record_directory):
        import json
        import pandas as pd
        env_json = Path(record_directory, "env.json")
        with open(env_json) as f:
            n_sub = json.load(f)["n_sub"]
        line_parquet = Path(record_directory, "line.parquet")
        n_line = len(pd.read_parquet(line_parquet))
        return n_sub, n_line

    def _count_timesteps(self, record_directory):
        import pandas as pd
        actions_file = Path(record_directory, "actions.parquet")
        try:
            df = pd.read_parquet(actions_file)
            return len(df)
        except Exception:
            logger.warning("Could not determine timestep count from recording")
            return 0

    def _compute_operational_score(self, record_directory):
        kpi = OperationScoreKpi()
        kpi_result = kpi.evaluate(record_directory)
        n_redispatch = kpi_result[1]
        e_redispatch = kpi_result[2]
        e_balancing = kpi_result[3]
        n_curtailment = kpi_result[4]
        e_curtailment = kpi_result[5]
        e_losses = kpi_result[6]
        e_blackout = kpi_result[7]
        redispatch_fraction = e_redispatch / n_redispatch if n_redispatch != 0 else 0
        curtailment_fraction = e_curtailment / n_curtailment if n_curtailment != 0 else 0
        return redispatch_fraction + e_balancing + curtailment_fraction + e_losses + e_blackout

    @staticmethod
    def load_scenario_data(scenario_id: str):
        return {
            "5950ad04-76e5-4c4d-aa44-435d01d250eb": "l2rpn_case14_sandbox",
            # "6037056e-f720-4ec6-b867-24cd3483cc0c": "ai4realnet_small",
            # "c2413ae7-e973-4846-b61b-d404cb518dfb": "ai4realnet_large",
        }[scenario_id]


if __name__ == "__main__":
    import sys
    import uuid

    logging.basicConfig(level=logging.INFO)

    SCENARIO_IDS = [
        "5950ad04-76e5-4c4d-aa44-435d01d250eb",  # l2rpn_case14_sandbox
        # "6037056e-f720-4ec6-b867-24cd3483cc0c",  # ai4realnet_small
        # "c2413ae7-e973-4846-b61b-d404cb518dfb",  # ai4realnet_large
    ]

    runner = TestRunner_KPI_AF_051_Power_Grid(
        test_id="1409dbf6-0f66-4570-97df-fda84c46c71d",
        scenario_ids=SCENARIO_IDS,
        benchmark_id="16706c82-75df-4969-932d-a7f5c941eca2"
    )
    submission_data_url = os.environ.get("SUBMISSION_DATA_URL", "None")
    runner.init(submission_data_url=submission_data_url, submission_id=str(uuid.uuid4()))

    try:
        results = runner.run()
        print("\nRESULTS:")
        for scenario_id, key, value in results:
            print(f"  {scenario_id} | {key}: {value}")
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
