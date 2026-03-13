from ai4realnet_orchestrators.power_grid.power_grid_test_runner import PowerGridTestRunner
from grid2evaluate.agent_runnner import AgentRunner
from grid2evaluate.network_utilization_kpi import NetworkUtilizationKpi
from pathlib import Path

class TestRunner_KPI_NF_024_Power_Grid(PowerGridTestRunner):

  def getResult(self, record_directory: Path):
    kpi = NetworkUtilizationKpi()
    kpi_result = kpi.evaluate(record_directory)
    rho_N_avg = kpi_result[2]
    rho_N_1_avg = kpi_result[3]
    primary_kpi_value = rho_N_avg + rho_N_1_avg

    return {
      "primary": primary_kpi_value
    }

