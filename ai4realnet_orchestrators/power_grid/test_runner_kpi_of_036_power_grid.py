from ai4realnet_orchestrators.power_grid.power_grid_test_runner import PowerGridTestRunner
from grid2evaluate.agent_runnner import AgentRunner
from grid2evaluate.operation_score_kpi import OperationScoreKpi
from pathlib import Path

class TestRunner_KPI_OF_036_Power_Grid(PowerGridTestRunner):

  def getResult(self, record_directory: Path):
    kpi = OperationScoreKpi()
    kpi_result = kpi.evaluate(record_directory)
    n_redispatch = kpi_result[1]
    e_redispatch = kpi_result[2]
    e_balancing = kpi_result[3]
    n_curtailment = kpi_result[4]
    e_curtailment = kpi_result[5]
    e_losses = kpi_result[6]
    e_blackout = kpi_result[7]
    redispatch_fraction = e_redispatch / n_redispatch if n_redispatch !=0 else 0
    curtailment_fraction = e_curtailment / n_curtailment if n_curtailment !=0 else 0
    primary_kpi_value = redispatch_fraction + e_balancing + curtailment_fraction + e_losses + e_blackout

    return {
      "primary": primary_kpi_value
    }


