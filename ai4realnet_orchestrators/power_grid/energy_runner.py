from ai4realnet_orchestrators.test_runner import TestRunner

class EnergyTestRunner(TestRunner):
  def init(self, submission_data_url: str, submission_id: str):
    super().init(submission_data_url=submission_data_url, submission_id=submission_id)
    # submission_data = load_submission_data(submission_data_url)
    # self.model = load_model(submission_data)

  def run_scenario(self, scenario_id: str, submission_id: str):
    # here you would implement the logic to run the test for the scenario:
    # scenario_data = load_scenario_data(scenario_id)
    # model = self.model


    # data and other stuff initialized in the init method can be used here
    # for demonstration, we return a dummy result
    return {
      "normalized_reward": 52,
      "percentage_complete": 99
    }


