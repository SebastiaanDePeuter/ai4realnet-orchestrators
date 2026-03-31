import os
import logging
import uuid
import json
import shutil

from ai4realnet_orchestrators.fab_exec_utils import exec_with_logging
from ai4realnet_orchestrators.test_runner import TestRunner

# For docker
DATA_VOLUME = os.environ.get("DATA_VOLUME", ".")
DATA_VOLUME_MOUNTPATH = os.environ.get("DATA_VOLUME_MOUNTPATH", "/app/data")

logger = logging.getLogger(__name__)

class TestRunner_KPI_RS_058_Power_Grid(TestRunner):
    """
    This implements KPI 058 (robustness to operator input). It assumes that submission_data_url points to a docker 
    image implemening the testing code of the KPI itself. How the docker image should be implemented is explained in 
    https://github.com/SebastiaanDePeuter/AI4RealNet_KPI-RS-058. A correctly implemented docker image will contain the 
    Grid2Op simulator, the agent to be tested, and a random "human" intervention policy. Performance of the agent is 
    evaluated and key measurements are written into a JSON file located in the DATA_VOLUME. Robustness to an operator 
    is measured by comparing the performance of the agent itself, with its performance when part of its actions (10%) 
    is overridden by the human policy.
    """

    def init(self, submission_data_url: str, submission_id: str):
        self.submission_data_url = submission_data_url
        self.submission_id = submission_id
        self.submission_data = {}

    def exec(self, within_docker_save_path, intervention_rate):
        args = ["docker", "run", "--rm", 
                "-v", f"{DATA_VOLUME}:{DATA_VOLUME_MOUNTPATH}", 
                "--security-opt=no-new-privileges", 
                "-e", "PYTHONUNBUFFERED=1", 
                self.submission_data_url, 
                "--intervention_rate", str(intervention_rate), 
                "--save_path", f"{DATA_VOLUME_MOUNTPATH}/{within_docker_save_path}"]
        exec_with_logging(args, log_level_stdout=logging.DEBUG)

        try:
            with open(f"{DATA_VOLUME}/{within_docker_save_path}", "r") as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load trajectory: {e}")
            raise

    def run_scenario(self, scenario_id: str, submission_id: str):
        # make sure folder exists at DATA_VOLUME
        data_dir = f"{submission_id}/{self.test_id}/{scenario_id}"
        os.makedirs(f"{DATA_VOLUME}/{data_dir}", exist_ok=True)

        # pull docker image
        exec_with_logging(["docker", "pull", self.submission_data_url])

        intervened_results = self.exec(f"{data_dir}/intervened_results.json", 0.05)
        base_results = self.exec(f"{data_dir}/base_results.json", 0.0)

        # clean up the generated files
        try:
            shutil.rmtree(f"{DATA_VOLUME}/{data_dir}")
        except Exception as e:
            logger.error(f"Failed to load trajectory: {e}")
            raise

        return {
            'primary': intervened_results["mean_reward"] - base_results["mean_reward"],
            'gained_reward': intervened_results["mean_reward"] - base_results["mean_reward"],
            'gained_ep_completion': intervened_results["mean_rel_ep_len"] - base_results["mean_rel_ep_len"],
        }