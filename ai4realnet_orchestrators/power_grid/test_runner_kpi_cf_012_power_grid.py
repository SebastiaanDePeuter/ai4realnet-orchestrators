from ai4realnet_orchestrators.power_grid.power_grid_test_runner import PowerGridTestRunner
from grid2evaluate.agent_runnner import AgentRunner
from grid2evaluate.carbon_intensity_kpi import CarbonIntensityKpi
from pathlib import Path

class TestRunner_KPI_CF_012_Power_Grid(PowerGridTestRunner):

  def getResult(self, record_directory: Path):
    kpi = CarbonIntensityKpi()
    kpi_result = kpi.evaluate(record_directory)
    primary_kpi_value = kpi_result[0] 

    return {
      "primary": primary_kpi_value
    }

