from abc import abstractmethod
import os
from ai4realnet_orchestrators.test_runner import TestRunner
from grid2evaluate.agent_runnner import AgentRunner
from pathlib import Path
import json
import requests

class PowerGridTestRunner(TestRunner):

  def init(self, submission_data_url: str, submission_id: str):
    super().init(submission_data_url=submission_data_url, submission_id=submission_id)
    self.submission_data = PowerGridTestRunner.load_submission_data(submission_data_url)
    print(self.submission_data)

  def run_scenario(self, scenario_id: str, submission_id: str):
    # here you would implement the logic to run the test for the scenario:
    scenario_data = PowerGridTestRunner.load_scenario_data(scenario_id)
    input_directory=Path(scenario_data['scenario_base_path'], self.submission_data[scenario_id]['scenario_name'])
    record_directory=Path(scenario_data['scenario_recorder_path'], self.submission_data[scenario_id]['scenario_name'])

    if not os.path.exists(record_directory):
        os.makedirs(record_directory)

    agent_runner = AgentRunner()
    agent_runner.run(input_directory, record_directory)
    return self.getResult(record_directory)

  @staticmethod
  def load_scenario_data(scenario_id: str) -> str:
    with open('./ai4realnet_orchestrators/power_grid/runner-config.json') as config_file:
      runner_config = json.load(config_file)

    print(runner_config)
    return runner_config[scenario_id]
  
  @staticmethod
  def load_submission_data(submission_data_url: str) -> str:
    print('read submission data from url = ' + submission_data_url)
    response = requests.get(submission_data_url)
    print(response.content)
    return response.json()

@abstractmethod
def getResult(self, record_directory: Path):
  raise NotImplementedError("This method should be overridden by subclasses.")
