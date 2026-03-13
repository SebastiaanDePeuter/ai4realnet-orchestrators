from ai4realnet_orchestrators.power_grid.power_grid_test_runner import PowerGridTestRunner
from grid2evaluate.agent_runnner import AgentRunner
from grid2evaluate.topological_action_complexity_kpi import TopologicalActionComplexityKpi
from pathlib import Path

class TestRunner_KPI_TF_034_Power_Grid(PowerGridTestRunner):

  def getResult(self, record_directory: Path):
    kpi = TopologicalActionComplexityKpi()
    kpi_result = kpi.evaluate(record_directory)
    avg_topology = kpi_result[2]
    primary_kpi_value = avg_topology

    return {
      "primary": primary_kpi_value
    }

