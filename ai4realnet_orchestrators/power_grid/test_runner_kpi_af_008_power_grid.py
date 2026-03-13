from ai4realnet_orchestrators.power_grid.power_grid_test_runner import PowerGridTestRunner
from grid2evaluate.assistant_alert_accuracy_kpi import AssistantAlertAccuracyKpi
from pathlib import Path

class TestRunner_KPI_AF_008_Power_Grid(PowerGridTestRunner):

  def run_scenario(self, scenario_id: str, submission_id: str):
    # here you would implement the logic to run the test for the scenario:
    scenario_data = PowerGridTestRunner.load_scenario_data(scenario_id)
    input_directory=Path(scenario_data['scenario_base_path'], self.submission_data[scenario_id]['scenario_name'])

    return self.getResult(input_directory)

  def getResult(self, input_directory: Path):
    kpi = AssistantAlertAccuracyKpi()
    kpi_result = kpi.evaluate(input_directory)

    if len(kpi_result)==0:
      return{
        "primary": 0
      }

    TP=sum([kpi_result[i][0] for i in range(len(kpi_result))])
    TN=sum([kpi_result[i][1] for i in range(len(kpi_result))])
    FP=sum([kpi_result[i][2] for i in range(len(kpi_result))])
    FN=sum([kpi_result[i][3] for i in range(len(kpi_result))])

    if TP==0:
      return{
        "primary": 0
      }

    P = TP/(TP+FP)
    R = TP/(TP+FN)
    primary_kpi_value=2*(P*R)/(P+R)

    return {
      "primary": primary_kpi_value
    }
