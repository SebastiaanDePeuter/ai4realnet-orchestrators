import os
import logging
import uuid
import json

from ai4realnet_orchestrators.fab_exec_utils import exec_with_logging

# from ai4realnet_orchestrators.s3_utils import s3_utils, S3_BUCKET, AI4REALNET_S3_UPLOAD_ROOT
from ai4realnet_orchestrators.test_runner import TestRunner

# For docker
# DATA_VOLUME_MOUNTPATH = os.environ.get("DATA_VOLUME_MOUNTPATH", "/app/data")

# TODO: This should not set to be TRUE by default, otherwise the test will never be actually run
# FOR debugging purposed, it is set to True
POWERGRID_ORCHESTRATOR_RUN_LOCAL = os.environ.get("POWERGRID_ORCHESTRATOR_RUN_LOCAL", "True")

SUBMISSION_PATH = os.environ.get("SUBMISSION_PATH", "./submission")
CONFIG_PATH = os.environ.get("CONFIG_PATH", os.path.join(SUBMISSION_PATH, "config.ini"))
ENV_PATH = os.environ.get("ENV_PATH", "/home/milad/repos/ai4realnet/grid2op-scenario/l2rpn_case14_sandbox")
ENV_MOUNT_PATH = os.environ.get("ENV_MOUNT_PATH", "/root/data_grid2op/l2rpn_case14_sandbox")


logger = logging.getLogger(__name__)

def load_scenario_data(scenario_id: str):
    # TODO : this may return the chronic ids instead of env_name. 
    return {
        "5950ad04-76e5-4c4d-aa44-435d01d250eb": "l2rpn_case14_sandbox",
        "6037056e-f720-4ec6-b867-24cd3483cc0c": "ai4realnet_small",
        "c2413ae7-e973-4846-b61b-d404cb518dfb": "ai4realnet_large"
    }[scenario_id]
  

# KPI-DF-052: Domain shift adaptation time (Power Grid)
class TestRunner_KPI_DF_052_Power_Grid(TestRunner):
    """
    The domain shift adaptation runner
    """
    # def init(self, submission_data_url: str, submission_id: str):
    #     super().init(submission_data_url=submission_data_url, submission_id=submission_id)
        # submission_data = load_submission_data(submission_data_url)
        # self.model = load_model(submission_data)

    def run_scenario(self, scenario_id: str, submission_id: str):
        # here you would implement the logic to run the test for the scenario:
        # env_path = load_scenario_data(scenario_id)
        # the path where the results should be saved
        # data_dir = f"{DATA_VOLUME_MOUNTPATH}/{submission_id}/{self.test_id}/{scenario_id}"
        
        if eval(POWERGRID_ORCHESTRATOR_RUN_LOCAL):
            # run local when the the environment variable is set to `True`
            
            # TODO: constant value to test the execution, to be replaced with a docker that computes the KPI
            results = {}
            results["status"] = str(True)
            results["performance_drop"] = float(50)
            results["adaptation_time"] = int(1000)
        else:
            #env = grid2op.make(env_path)
            args = ["docker", "pull", "mleyliabadi/ds-kpi:v2"]
            exec_with_logging(args)
            
            args = [
                "docker", "run",
                "--rm",
                "--name", "ds-kpi",
                "-v", f"{CONFIG_PATH}:{'/usr/src/config.ini'}",
                # "-v", f"{ENV_PATH}:{ENV_MOUNT_PATH}",               
                "-v", f"{SUBMISSION_PATH}:{os.path.join('/usr/src', SUBMISSION_PATH)}",
                "-v", f"{os.path.join(SUBMISSION_PATH, 'test_results')}:{'/usr/src/results'}",
                "mleyliabadi/ds-kpi:v2"
            ]
            exec_with_logging(args)
            
            results_path = os.path.join(SUBMISSION_PATH, 'test_results', "kpi_results.json")
            with open(file=results_path, mode="r") as fs:
                results = json.load(fs)
            # make a directory to save the results
            # Path(data_dir).mkdir(parents=True, exist_ok=False)
            # 1) run the model on the specific environment and scenario
            # 2) compute the KPI 
            # 3) return the KPI
            

        if eval(str(results["status"])):
            final_result = results["adaptation_time"]
        else:
            final_result = None
        
        # data and other stuff initialized in the init method can be used here
        # for demonstration, we return a dummy result
        return {
            "primary": final_result
        }
        # return {
        #     "status": results["status"],
        #     "adaptation_time": results["adaptation_time"],
        #     "performance_drop": results["performance_drop"]
        # }
      
if __name__ == "__main__":
    test_runner = TestRunner_KPI_DF_052_Power_Grid(test_id="855729a4-6729-4ae2-bb8d-443ef4867d94",
                                                   scenario_ids=["5950ad04-76e5-4c4d-aa44-435d01d250eb"])
    test_runner.init(submission_data_url="None", submission_id=str(uuid.uuid4()))
    results = test_runner.run()
    print(results)
    