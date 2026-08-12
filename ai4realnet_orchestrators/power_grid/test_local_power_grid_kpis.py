"""
Local Test Script for Power Grid KPI Evaluation Framework
========================================================

Runs Operational, Reliability, and Robustness & Resilience evaluations LOCALLY 
without the FAB orchestrator. Use for development, debugging, and testing 
agents before submission.

Usage:
    python test_local_power_grid_kpis.py

Note: 
    This bypasses Celery/RabbitMQ orchestration. For production, use FAB.

Author: AI4REALNET Consortium
"""

import sys
import os
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore")
os.environ["KMP_WARNINGS"] = "0"

# Add paths for imports
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ORCHESTRATORS_DIR = os.path.dirname(SCRIPT_DIR)  # ai4realnet_orchestrators
PARENT_DIR = os.path.dirname(ORCHESTRATORS_DIR)  # ai4realnet-orchestrators
sys.path.insert(0, PARENT_DIR)
sys.path.insert(0, ORCHESTRATORS_DIR)

from ai4realnet_orchestrators.power_grid.operational_test_runner import (
    OperationalTestRunner,
    OPERATIONAL_KPI_MAPPING
)
from ai4realnet_orchestrators.power_grid.reliability_test_runner import (
    ReliabilityTestRunner,
    RELIABILITY_KPI_MAPPING
)
from ai4realnet_orchestrators.power_grid.robustness_resilience_test_runner import (
    RobustnessResilienceTestRunner,
    ROBUSTNESS_RESILIENCE_KPI_MAPPING
)
from ai4realnet_orchestrators.power_grid.test_runner_kpi_rs_058_power_grid import (
    TestRunner_KPI_RS_058_Power_Grid,
    ROBUSTNESS_TO_OPERATOR_KPI_MAPPING
)

EXPERT_SUBMISSION = "https://raw.githubusercontent.com/flatland-association/ai4realnet-orchestrators/refs/heads/milad-merged-powergrid-kpis/ai4realnet_orchestrators/power_grid/configuration/expert-ai4realnet-small.json"
CURRICULUM_SUBMISSION = "https://raw.githubusercontent.com/flatland-association/ai4realnet-orchestrators/refs/heads/milad-merged-powergrid-kpis/ai4realnet_orchestrators/power_grid/configuration/curriculum-ai4realnet-small.json"

reliability_runner = ReliabilityTestRunner(test_id="855729a4-6729-4ae2-bb8d-443ef4867d94",
                                           scenario_ids=['81f18394-0164-4896-9408-4315bcfcc5e0'], 
                                           benchmark_id="43040944-39ac-47c9-b91d-bc8ca5693b3c")

reliability_runner.init(
    submission_data_url=EXPERT_SUBMISSION,
    submission_id="local_test_reliability"
)

operational_runner = OperationalTestRunner(test_id="ae4dcac7-c559-457e-902d-ee35d064bb3f",
                                           scenario_ids=['fc090c38-8740-4911-96aa-2defd06f8715'],
                                           benchmark_id="4b0be731-8371-4e4e-a673-b630187b0bb8")
operational_runner.init(
    submission_data_url=CURRICULUM_SUBMISSION,
    submission_id="local_test_operational"
)

robustness_runner = RobustnessResilienceTestRunner(test_id="1cbb7783-47b4-4289-9abf-27939da69a2f",
                                                  scenario_ids=['900d5489-2539-4a49-b3fb-3ae2039be92f'],
                                                  benchmark_id="3810191b-8cfd-4b03-86b2-f7e530aab30d")
robustness_runner.init(
    submission_data_url=CURRICULUM_SUBMISSION,
    submission_id="local_test_robustness"
)

robustness_to_operator_runner = TestRunner_KPI_RS_058_Power_Grid(test_id="75cc9343-9371-4eb1-9613-22a26c67fc00",
                                                  scenario_ids=['0c0730f2-e795-4c9d-8220-9bee29c46dc6'],
                                                  benchmark_id="3810191b-8cfd-4b03-86b2-f7e530aab30d")
robustness_to_operator_runner.init(
    submission_data_url=CURRICULUM_SUBMISSION,
    submission_id="local_test_robustness_to_operator"
)

print(f"✅ Test runners initialized")

try:
    # ============================================================
    # 1. OPERATIONAL KPIs
    # ============================================================
    print("\n" + "=" * 60)
    print("🔄 Running Operational evaluation...")
    print("=" * 60)
    operational_runner.run_scenario(
        scenario_id="fc090c38-8740-4911-96aa-2defd06f8715",
        submission_id="local_test_operational"
    )
    
    print("\n📈 Operational KPIs Results:")
    op_cache = OperationalTestRunner._metrics_cache.get("local_test_operational")
    if op_cache:
        for kpi_id, info in OPERATIONAL_KPI_MAPPING.items():
            val = op_cache.get(info['metric_key'], 0.0)
            print(f"  - {info['name']}: {val:.4f}")
            print(f"    Description: {info['description']}")
    else:
        print("  ⚠️ No operational results found in cache.")

    # ============================================================
    # 2. RELIABILITY KPIs
    # ============================================================
    print("\n" + "=" * 60)
    print("🔄 Running Reliability evaluation (Domain Shift)...")
    print("=" * 60)
    reliability_runner.run_scenario(
        scenario_id="81f18394-0164-4896-9408-4315bcfcc5e0",
        submission_id="local_test_reliability"
    )

    print("\n📈 Reliability KPIs Results:")
    rel_cache = ReliabilityTestRunner._metrics_cache.get("local_test_reliability")
    if rel_cache:
        for kpi_id, info in RELIABILITY_KPI_MAPPING.items():
            val = rel_cache.get(info['metric_key'], 0.0)
            print(f"  - {info['name']}: {val:.4f}")
            print(f"    Description: {info['description']}")
    else:
        print("  ⚠️ No reliability results found in cache.")

    # ============================================================
    # 3. ROBUSTNESS & RESILIENCE KPIs
    # ============================================================
    print("\n" + "=" * 60)
    print("🔄 Running Robustness & Resilience evaluation...")
    print("=" * 60)
    robustness_runner.run_scenario(
        scenario_id="900d5489-2539-4a49-b3fb-3ae2039be92f",
        submission_id="local_test_robustness"
    )
    
    print("\n📈 Robustness & Resilience KPIs Results:")
    rob_cache = RobustnessResilienceTestRunner._metrics_cache.get("local_test_robustness")
    if rob_cache:
        # Separate Robustness and Resilience for display
        print("\n  [Robustness]")
        for kpi_id, info in ROBUSTNESS_RESILIENCE_KPI_MAPPING.items():
            kpi_num = info['name'].split(':')[0].split('-')[-1]
            if "074" <= kpi_num <= "077":
                continue # Skip resilience for now
            val = rob_cache.get(info['metric_key'], 0.0)
            print(f"    - {info['name']}: {val:.4f}")

        print("\n  [Resilience]")
        for kpi_id, info in ROBUSTNESS_RESILIENCE_KPI_MAPPING.items():
            kpi_num = info['name'].split(':')[0].split('-')[-1]
            if "074" <= kpi_num <= "077":
                val = rob_cache.get(info['metric_key'], 0.0)
                print(f"    - {info['name']}: {val:.4f}")
    else:
        print("  ⚠️ No robustness results found in cache.")

    # ============================================================
    # 4. ROBUSTNESS TO OPERATOR INPUT KPI
    # ============================================================
    print("\n" + "=" * 60)
    print("🔄 Running Robustness to Operator Input evaluation...")
    print("=" * 60)
    robustness_to_operator_runner.run_scenario(
        scenario_id="0c0730f2-e795-4c9d-8220-9bee29c46dc6",
        submission_id="local_test_robustness_to_operator"
    )
    
    print("\n📈 Operational KPIs Results:")
    rob_op_inpt_cache = TestRunner_KPI_RS_058_Power_Grid._metrics_cache.get("local_test_robustness_to_operator")
    if rob_op_inpt_cache:
        for kpi_id, info in ROBUSTNESS_TO_OPERATOR_KPI_MAPPING.items():
            val = rob_op_inpt_cache.get(info['metric_key'], 0.0)
            print(f"  - {info['name']}: {val:.4f}")
            print(f"    Description: {info['description']}")
    else:
        print("  ⚠️ No operational results found in cache.")

    print("\n" + "=" * 60)
    print("✅ ALL EVALUATIONS COMPLETE!")
    print("=" * 60)
    
except Exception as e:
    print(f"\n❌ Error: {e}")
    import traceback
    traceback.print_exc()
