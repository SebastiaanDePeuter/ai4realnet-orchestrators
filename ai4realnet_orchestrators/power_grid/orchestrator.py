# based on https://github.com/codalab/codabench/blob/develop/compute_worker/compute_worker.py
import logging
import os
import ssl
from typing import List

from celery import Celery

from ai4realnet_orchestrators.orchestrator import Orchestrator

# NOTE: import YourTestRunner implementations here
# from ai4realnet_orchestrators.power_grid.test_runner import YourTestRunner
from ai4realnet_orchestrators.power_grid.test_runner_robustness_resilience_kpi_069_077 import (
    TestRunner_KPI_DF_069_Power_Grid,
    TestRunner_KPI_FF_070_Power_Grid,
    TestRunner_KPI_SF_071_Power_Grid,
    TestRunner_KPI_SF_072_Power_Grid,
    TestRunner_KPI_VF_073_Power_Grid,
    TestRunner_KPI_AF_074_Power_Grid,
    TestRunner_KPI_DF_075_Power_Grid,
    TestRunner_KPI_RF_076_Power_Grid,
    TestRunner_KPI_SF_077_Power_Grid,
)

from ai4realnet_orchestrators.power_grid.test_runner_kpi_cf_012_power_grid import TestRunner_KPI_CF_012_Power_Grid
from ai4realnet_orchestrators.power_grid.test_runner_kpi_nf_024_power_grid import TestRunner_KPI_NF_024_Power_Grid
from ai4realnet_orchestrators.power_grid.test_runner_kpi_of_036_power_grid import TestRunner_KPI_OF_036_Power_Grid
from ai4realnet_orchestrators.power_grid.test_runner_kpi_tf_034_power_grid import TestRunner_KPI_TF_034_Power_Grid
from ai4realnet_orchestrators.power_grid.test_runner_kpi_af_008_power_grid import TestRunner_KPI_AF_008_Power_Grid

from ai4realnet_orchestrators.power_grid.test_runner_kpi_df_052_power_grid import TestRunner_KPI_DF_052_Power_Grid

logger = logging.getLogger(__name__)

app = Celery(
    broker=os.environ.get('BROKER_URL'),
    backend=os.environ.get('BACKEND_URL'),
    queue=os.environ.get("BENCHMARK_ID"),
    broker_use_ssl={
        'keyfile': os.environ.get("RABBITMQ_KEYFILE"),
        'certfile': os.environ.get("RABBITMQ_CERTFILE"),
        'ca_certs': os.environ.get("RABBITMQ_CA_CERTS"),
        'cert_reqs': ssl.CERT_REQUIRED
    }
)

# NOTE: Uncomment and implement the test runners you want
# Generated with https://github.com/flatland-association/flatland-benchmarks/blob/main/definitions/ai4realnet/gen_ai4realnet_benchmarks_sql.py
# from https://inesctecpt.sharepoint.com/:x:/r/sites/AI4REALNET/Shared%20Documents/General/WP4%20-%20Validation%20and%20impact%20assessment/Validation%20campaigns/Overview%20tests%20for%20KPI%20on%20validation%20campaign%20hub.xlsx?d=w947339379458465eaaf243a750315375&csf=1&web=1&e=RnrCdf
power_grid_orchestrator = Orchestrator(
    test_runners={

        #     # KPI-AS-001: Ability to anticipate (Power Grid)
        #     "98413684-d114-4f88-a5e9-1e118b106d67": TestRunner_KPI_AS_001_Power_Grid(
        #         test_id="98413684-d114-4f88-a5e9-1e118b106d67", scenario_ids=['ef0af7e2-0212-454d-9391-41de03bd7e57'], benchmark_id="3237ba20-ccff-45b0-af23-44719e584f41"
        #     ),
        #
        #     # KPI-AS-009: Assistant disturbance (Power Grid)
        #     "6e005336-6c4c-43ae-ac8d-330d51ab48d4": TestRunner_KPI_AS_009_Power_Grid(
        #         test_id="6e005336-6c4c-43ae-ac8d-330d51ab48d4", scenario_ids=['57c11d6d-0001-4623-9c1e-fbfc0744c8d5'], benchmark_id="3237ba20-ccff-45b0-af23-44719e584f41"
        #     ),
        #
        #     # KPI-DS-015: Decision support satisfaction (Power Grid)
        #     "4a9db7b3-0dd9-4ac2-a211-ef5a991e4ab9": TestRunner_KPI_DS_015_Power_Grid(
        #         test_id="4a9db7b3-0dd9-4ac2-a211-ef5a991e4ab9", scenario_ids=['3ac00dab-5671-4e60-bf35-99309075ee76'], benchmark_id="3237ba20-ccff-45b0-af23-44719e584f41"
        #     ),
        #
        #     # KPI-HS-022: Human motivation (Power Grid)
        #     "d1ba73c3-6796-44db-8d87-129558e3535d": TestRunner_KPI_HS_022_Power_Grid(
        #         test_id="d1ba73c3-6796-44db-8d87-129558e3535d", scenario_ids=['ee4e4f46-9bc1-4da8-94b5-371ffddfab7e'], benchmark_id="3237ba20-ccff-45b0-af23-44719e584f41"
        #     ),
        #
        #     # KPI-HS-023: Human response time (Power Grid)
        #     "dbd85bfb-2e78-4646-b2da-b4d16e90e657": TestRunner_KPI_HS_023_Power_Grid(
        #         test_id="dbd85bfb-2e78-4646-b2da-b4d16e90e657", scenario_ids=['d64548d6-2eb6-4e1e-8069-73c0ece64318'], benchmark_id="3237ba20-ccff-45b0-af23-44719e584f41"
        #     ),
        #
        #     # KPI-SS-031: Situation awareness (Power Grid)
        #     "4166c200-b50e-4e2e-a3a7-05f31e7f78f8": TestRunner_KPI_SS_031_Power_Grid(
        #         test_id="4166c200-b50e-4e2e-a3a7-05f31e7f78f8", scenario_ids=['5096d57a-00a3-4018-bdc3-02cd28443a85'], benchmark_id="3237ba20-ccff-45b0-af23-44719e584f41"
        #     ),
        #
        #     # KPI-WS-040: Workload (Power Grid)
        #     "66c08000-9a5e-42ca-a328-8e6295069142": TestRunner_KPI_WS_040_Power_Grid(
        #         test_id="66c08000-9a5e-42ca-a328-8e6295069142", scenario_ids=['06f61acd-da79-493c-8da0-4a1327b5fe6a'], benchmark_id="3237ba20-ccff-45b0-af23-44719e584f41"
        #     ),
        #
        #     # KPI-CS-049: Cognitive Performance & Stress (Power Grid)
        #     "62bcfa08-99ce-4bfe-b7a4-08343ef3a316": TestRunner_KPI_CS_049_Power_Grid(
        #         test_id="62bcfa08-99ce-4bfe-b7a4-08343ef3a316", scenario_ids=['6ae869dc-40e3-4cd0-a02c-6541a838b5b3'], benchmark_id="3237ba20-ccff-45b0-af23-44719e584f41"
        #     ),
        #
        #     # KPI-AS-002: Acceptance (Power Grid)
        #     "de7f38de-1f0b-450e-9951-b925523a563d": TestRunner_KPI_AS_002_Power_Grid(
        #         test_id="de7f38de-1f0b-450e-9951-b925523a563d", scenario_ids=['b4b167ff-56be-43a1-9a80-121eaaf8108f'], benchmark_id="2da3781a-25a9-4c89-8b43-9269844f3fef"
        #     ),
        #
        #     # KPI-AS-005: Agreement score (Power Grid)
        #     "923fddc9-99f6-4195-a857-00d1493886e6": TestRunner_KPI_AS_005_Power_Grid(
        #         test_id="923fddc9-99f6-4195-a857-00d1493886e6", scenario_ids=['631244e4-ade3-4cbb-afd6-a19e56c001d6'], benchmark_id="2da3781a-25a9-4c89-8b43-9269844f3fef"
        #     ),
        #
        #     # KPI-CS-013: Comprehensibility (Power Grid)
        #     "66d9dc9e-a163-4afc-b293-9ece9d45f3cf": TestRunner_KPI_CS_013_Power_Grid(
        #         test_id="66d9dc9e-a163-4afc-b293-9ece9d45f3cf", scenario_ids=['9d6c321f-25c4-4b31-91e6-0208c1da3455'], benchmark_id="2da3781a-25a9-4c89-8b43-9269844f3fef"
        #     ),
        #
        #     # KPI-TS-038: Trust in AI solutions score (Power Grid)
        #     "b10a5c41-7f19-414c-9ffe-3b73774ce1d9": TestRunner_KPI_TS_038_Power_Grid(
        #         test_id="b10a5c41-7f19-414c-9ffe-3b73774ce1d9", scenario_ids=['5314d4ab-35d4-4bc3-ade8-3b17bd39dd82'], benchmark_id="2da3781a-25a9-4c89-8b43-9269844f3fef"
        #     ),
        #
        #     # KPI-TS-039: Trust towards the AI tool (Power Grid)
        #     "9db74767-f2c3-4f04-aa83-cab1746ab83f": TestRunner_KPI_TS_039_Power_Grid(
        #         test_id="9db74767-f2c3-4f04-aa83-cab1746ab83f", scenario_ids=['2d5f2c8a-5cfd-4f32-9174-7ae81a82f0be'], benchmark_id="2da3781a-25a9-4c89-8b43-9269844f3fef"
        #     ),
        #
        #     # KPI-HS-003: Human intervention frequency (Power Grid)
        #     "bc503fa3-b1ea-4de5-8760-21bd3ede927f": TestRunner_KPI_HS_003_Power_Grid(
        #         test_id="bc503fa3-b1ea-4de5-8760-21bd3ede927f", scenario_ids=['81cb1769-35b5-4b97-aff0-b0d070dd6e50'], benchmark_id="8d5c876e-22c2-49e7-bdd5-4c1840d309f0"
        #     ),
        #
        #     # KPI-SS-030: Significance of human revisions (Power Grid)
        #     "f22c4c5d-1957-4262-b763-12736dd692f9": TestRunner_KPI_SS_030_Power_Grid(
        #         test_id="f22c4c5d-1957-4262-b763-12736dd692f9", scenario_ids=['79f0ed9e-094f-4637-921c-814707f3b02e'], benchmark_id="8d5c876e-22c2-49e7-bdd5-4c1840d309f0"
        #     ),
        #
        #     # KPI-PS-089: Perceived decision novelty (Power Grid)
        #     "44758e37-c1d5-4932-8c17-54147903f214": TestRunner_KPI_PS_089_Power_Grid(
        #         test_id="44758e37-c1d5-4932-8c17-54147903f214", scenario_ids=['b3b52505-caa6-4438-90b8-6ac84e9880d9'], benchmark_id="8d5c876e-22c2-49e7-bdd5-4c1840d309f0"
        #     ),
        #
        #     # KPI-AS-006: AI co-learning capability (Power Grid)
        #     "b2213a56-7841-41e1-afbc-6ef541d1597c": TestRunner_KPI_AS_006_Power_Grid(
        #         test_id="b2213a56-7841-41e1-afbc-6ef541d1597c", scenario_ids=['91b63e8f-7cf1-461c-98ea-0269b26bb3b4'], benchmark_id="65547935-f436-49fa-8d20-f320c6bd46dc"
        #     ),
        #
        #     # KPI-HS-021: Human learning (Power Grid)
        #     "67f77d51-0893-4cbd-b349-b73bd2f73db2": TestRunner_KPI_HS_021_Power_Grid(
        #         test_id="67f77d51-0893-4cbd-b349-b73bd2f73db2", scenario_ids=['6958bf4c-39ff-484f-b609-25500e9e314a'], benchmark_id="65547935-f436-49fa-8d20-f320c6bd46dc"
        #     ),
        #
        # KPI-AF-008: Assistant alert accuracy (Power Grid)
        "aba10b3f-0d5c-4f90-aec4-69460bbb098b": TestRunner_KPI_AF_008_Power_Grid(
            test_id="aba10b3f-0d5c-4f90-aec4-69460bbb098b", scenario_ids=['729cc815-ac93-4209-9f62-b57b920c2d0a'], benchmark_id="3b1bdca6-ed90-4938-bd63-fd657aa7dcd7"
        ),
        #
        # KPI-NF-024: Network utilization (Power Grid)
        "5d1db79c-a7a4-4060-bb03-4629d64b1a43": TestRunner_KPI_NF_024_Power_Grid(
                test_id="5d1db79c-a7a4-4060-bb03-4629d64b1a43", scenario_ids=['ed8ba2fc-853e-4e79-a984-b1986b9b6e97'], benchmark_id="3b1bdca6-ed90-4938-bd63-fd657aa7dcd7"
            ),
        #
        #     # KPI-TS-035: Total decision time (Power Grid)
        #     "58ce79e0-5c14-4c51-8d09-89f856361259": TestRunner_KPI_TS_035_Power_Grid(
        #         test_id="58ce79e0-5c14-4c51-8d09-89f856361259", scenario_ids=['1294d425-66bd-4510-b4b3-d9f64ca0e4f9'], benchmark_id="3b1bdca6-ed90-4938-bd63-fd657aa7dcd7"
        #     ),
        #

        # KPI-CF-012: Carbon intensity (Power Grid)
        "ab91af79-ffc3-4da7-916a-6574609dc1b6": TestRunner_KPI_CF_012_Power_Grid(
            test_id="ab91af79-ffc3-4da7-916a-6574609dc1b6", scenario_ids=['75d20248-740b-4d84-86e7-1de89f10fc1e'], benchmark_id="4b0be731-8371-4e4e-a673-b630187b0bb8"
        ),
        #
        # KPI-TF-034: Topological action complexity (Power Grid)
        "0b8c02c6-0120-431c-872f-0fb4bc8d5fba": TestRunner_KPI_TF_034_Power_Grid(
            test_id="0b8c02c6-0120-431c-872f-0fb4bc8d5fba", scenario_ids=['5dd33cc9-a4aa-4a61-bd3f-5fae1c1bf701'], benchmark_id="4b0be731-8371-4e4e-a673-b630187b0bb8"
        ),
        #
        #     # KPI-OF-036: Operation score (Power Grid)
        "ae4dcac7-c559-457e-902d-ee35d064bb3f": TestRunner_KPI_OF_036_Power_Grid(
            test_id="ae4dcac7-c559-457e-902d-ee35d064bb3f", scenario_ids=['fc090c38-8740-4911-96aa-2defd06f8715'], benchmark_id="4b0be731-8371-4e4e-a673-b630187b0bb8"
        ),
        #
        #     # KPI-AS-068: Assistant adaptation to user preferences (Power Grid)
        #     "c69ff5e9-497b-41e8-adff-2221bb823365": TestRunner_KPI_AS_068_Power_Grid(
        #         test_id="c69ff5e9-497b-41e8-adff-2221bb823365", scenario_ids=['a68e7062-1329-4a34-ac44-4f6075929902'], benchmark_id="4b0be731-8371-4e4e-a673-b630187b0bb8"
        #     ),
        #
        #     # KPI-HS-018: Human control/autonomy over the process (Power Grid)
        #     "ed250353-1f97-413e-9971-de83937fe4d9": TestRunner_KPI_HS_018_Power_Grid(
        #         test_id="ed250353-1f97-413e-9971-de83937fe4d9", scenario_ids=['edcf64a8-f1b8-48ec-ab6a-229d5abc1be4'], benchmark_id="d65cd37a-4830-468c-9100-0f60ee9ff72e"
        #     ),
        #
        #     # KPI-IS-041: Impact on workload (Power Grid)
        #     "6d372248-1221-4996-ad99-628f056f0799": TestRunner_KPI_IS_041_Power_Grid(
        #         test_id="6d372248-1221-4996-ad99-628f056f0799", scenario_ids=['df02b09a-b431-49ee-af6b-ffd709d47670'], benchmark_id="d65cd37a-4830-468c-9100-0f60ee9ff72e"
        #     ),
        #
        #     # KPI-AF-050: AI-Agent Scalability Training (Power Grid)
        #     "5af6ffd9-b0a6-4f53-94bf-058fc1383ecd": TestRunner_KPI_AF_050_Power_Grid(
        #         test_id="5af6ffd9-b0a6-4f53-94bf-058fc1383ecd", scenario_ids=['7d2d75c8-49e0-433d-809d-b0811c8e2f06'], benchmark_id="16706c82-75df-4969-932d-a7f5c941eca2"
        #     ),
        #
        #     # KPI-AF-051: AI-Agent Scalability Testing (Power Grid)
        #     "1409dbf6-0f66-4570-97df-fda84c46c71d": TestRunner_KPI_AF_051_Power_Grid(
        #         test_id="1409dbf6-0f66-4570-97df-fda84c46c71d", scenario_ids=['547f8244-d091-40da-892d-ee24a26ee29f'], benchmark_id="16706c82-75df-4969-932d-a7f5c941eca2"
        #     ),
        #
            # KPI-DF-052: Domain shift adaptation time (Power Grid)
        "855729a4-6729-4ae2-bb8d-443ef4867d94": TestRunner_KPI_DF_052_Power_Grid(
            test_id="855729a4-6729-4ae2-bb8d-443ef4867d94", scenario_ids=['81f18394-0164-4896-9408-4315bcfcc5e0'], benchmark_id="43040944-39ac-47c9-b91d-bc8ca5693b3c"
        ),
        #
        #     # KPI-DF-053: Domain shift generalization gap (Power Grid)
        #     "17b805b2-b773-4c22-8ba9-598780e7a40d": TestRunner_KPI_DF_053_Power_Grid(
        #         test_id="17b805b2-b773-4c22-8ba9-598780e7a40d", scenario_ids=['9fdfbb00-0754-444a-88c8-c8549e2cc6f9'], benchmark_id="43040944-39ac-47c9-b91d-bc8ca5693b3c"
        #     ),
        #
        #     # KPI-DF-054: Domain shift out of domain detection accuracy (Power Grid)
        #     "7149a428-b0a7-48f7-ba55-4d553932af41": TestRunner_KPI_DF_054_Power_Grid(
        #         test_id="7149a428-b0a7-48f7-ba55-4d553932af41", scenario_ids=['747cba94-8353-4982-9964-ba0b8361e689'], benchmark_id="43040944-39ac-47c9-b91d-bc8ca5693b3c"
        #     ),
        #
        #     # KPI-DF-055: Domain shift policy robustness (Power Grid)
        #     "115d6c9d-c0d1-423d-b74e-140f9e5608c5": TestRunner_KPI_DF_055_Power_Grid(
        #         test_id="115d6c9d-c0d1-423d-b74e-140f9e5608c5", scenario_ids=['f393653b-850e-4c17-bed1-7fe1ca51c854'], benchmark_id="43040944-39ac-47c9-b91d-bc8ca5693b3c"
        #     ),
        #
        #     # KPI-DF-056: Domain shift robustness to domain parameters (Power Grid)
        #     "07f8625d-c39a-4fd1-9633-012f342352e9": TestRunner_KPI_DF_056_Power_Grid(
        #         test_id="07f8625d-c39a-4fd1-9633-012f342352e9", scenario_ids=['82aed30d-9b28-4b8f-ba9a-fd05d6defec6'], benchmark_id="43040944-39ac-47c9-b91d-bc8ca5693b3c"
        #     ),
        #
        #     # KPI-DF-057: Domain shift success rate drop (Power Grid)
        #     "c5e4f893-4302-47e8-98d6-b5fbcb10963a": TestRunner_KPI_DF_057_Power_Grid(
        #         test_id="c5e4f893-4302-47e8-98d6-b5fbcb10963a", scenario_ids=['4d2b00cd-447a-4c7e-8cab-863f0402cb67'], benchmark_id="43040944-39ac-47c9-b91d-bc8ca5693b3c"
        #     ),
        #
        #     # KPI-DF-090: Domain shift forgetting rate (Power Grid)
        #     "648afbec-80ad-4490-869f-6c3d8088d50f": TestRunner_KPI_DF_090_Power_Grid(
        #         test_id="648afbec-80ad-4490-869f-6c3d8088d50f", scenario_ids=['99dfde1e-2798-4741-b3eb-610a3e847bc8'], benchmark_id="43040944-39ac-47c9-b91d-bc8ca5693b3c"
        #     ),
        #
        #     # KPI-RS-058: Robustness to operator input (Power Grid)
        #     "75cc9343-9371-4eb1-9613-22a26c67fc00": TestRunner_KPI_RS_058_Power_Grid(
        #         test_id="75cc9343-9371-4eb1-9613-22a26c67fc00", scenario_ids=['0c0730f2-e795-4c9d-8220-9bee29c46dc6'], benchmark_id="3810191b-8cfd-4b03-86b2-f7e530aab30d"
        #     ),
        #
             # KPI-DF-069: Drop-off in reward (Power Grid)
             "1cbb7783-47b4-4289-9abf-27939da69a2f": TestRunner_KPI_DF_069_Power_Grid(
                 test_id="1cbb7783-47b4-4289-9abf-27939da69a2f", scenario_ids=['900d5489-2539-4a49-b3fb-3ae2039be92f'], benchmark_id="3810191b-8cfd-4b03-86b2-f7e530aab30d"
             ),
        #
             # KPI-FF-070: Frequency changed output AI agent (Power Grid)
             "acaf712a-c06c-4a04-a00f-0e7feeefb60c": TestRunner_KPI_FF_070_Power_Grid(
                 test_id="acaf712a-c06c-4a04-a00f-0e7feeefb60c", scenario_ids=['fdaac433-3ef0-4667-afb8-8014d0c1afa3'], benchmark_id="3810191b-8cfd-4b03-86b2-f7e530aab30d"
             ),
        #
             # KPI-SF-071: Severity of changed output AI agent (Power Grid)
             "3d033ec6-942a-4b03-b26e-f8152ba48022": TestRunner_KPI_SF_071_Power_Grid(
                 test_id="3d033ec6-942a-4b03-b26e-f8152ba48022", scenario_ids=['70d937d5-742b-4838-a456-4a95ff994788'], benchmark_id="3810191b-8cfd-4b03-86b2-f7e530aab30d"
             ),
        #
             # KPI-SF-072: Steps survived with perturbations (Power Grid)
             "a121d8bd-1943-41ba-b3a7-472a0154f8f9": TestRunner_KPI_SF_072_Power_Grid(
                 test_id="a121d8bd-1943-41ba-b3a7-472a0154f8f9", scenario_ids=['9cd1a5e0-8445-4b9d-859b-76b096d33049'], benchmark_id="3810191b-8cfd-4b03-86b2-f7e530aab30d"
             ),
        #
             # KPI-VF-073: Vulnerability to perturbation (Power Grid)
             "b8a9a411-7cfe-4c1d-b9a6-eef1c0efe920": TestRunner_KPI_VF_073_Power_Grid(
                 test_id="b8a9a411-7cfe-4c1d-b9a6-eef1c0efe920", scenario_ids=['61063867-df62-4024-be42-c57507a15d7c'], benchmark_id="3810191b-8cfd-4b03-86b2-f7e530aab30d"
             ),
        #
        #     # KPI-RF-078: Reward per action (Power Grid)
        #     "95ba1e9a-8d72-4c0e-9526-7676f70ff067": TestRunner_KPI_RF_078_Power_Grid(
        #         test_id="95ba1e9a-8d72-4c0e-9526-7676f70ff067", scenario_ids=['a999eb93-2efe-4f73-a2d8-eab51f158ae8'], benchmark_id="3810191b-8cfd-4b03-86b2-f7e530aab30d"
        #     ),
        #
        #     # KPI-EF-086: Explainability Robustness (Power Grid)
        #     "89919375-8b53-4e3f-8382-a97e0af7eb56": TestRunner_KPI_EF_086_Power_Grid(
        #         test_id="89919375-8b53-4e3f-8382-a97e0af7eb56", scenario_ids=['e36d1ef8-939c-4cd8-a660-0a224ce24aa0'], benchmark_id="3810191b-8cfd-4b03-86b2-f7e530aab30d"
        #     ),
        #
        #     # KPI-EF-087: Explainability Faithfulness (Power Grid)
        #     "ce7d0394-1aa0-41c2-84b7-dfcfe006eb8b": TestRunner_KPI_EF_087_Power_Grid(
        #         test_id="ce7d0394-1aa0-41c2-84b7-dfcfe006eb8b", scenario_ids=['53b0db0e-7092-455b-9e2c-327ee017f776'], benchmark_id="3810191b-8cfd-4b03-86b2-f7e530aab30d"
        #     ),
        #
             # KPI-AF-074: Area between reward curves (Power Grid)
             "534f5a1f-7115-48a5-b58c-4deb044d425d": TestRunner_KPI_AF_074_Power_Grid(
                 test_id="534f5a1f-7115-48a5-b58c-4deb044d425d", scenario_ids=['bbcf8224-c768-4469-8ff5-939d977383b4'], benchmark_id="31ea606b-681a-437a-85b9-7c81d4ccc287"
             ),
        #
             # KPI-DF-075: Degradation time (Power Grid)
             "04a23bfc-fc44-4ec4-a732-c29214130a83": TestRunner_KPI_DF_075_Power_Grid(
                 test_id="04a23bfc-fc44-4ec4-a732-c29214130a83", scenario_ids=['b355482b-30a2-431e-9536-8e3dd29d06d1'], benchmark_id="31ea606b-681a-437a-85b9-7c81d4ccc287"
             ),
        #
             # KPI-RF-076: Restorative time (Power Grid)
             "225aaee8-7c7f-4faf-810b-407b551e9f2a": TestRunner_KPI_RF_076_Power_Grid(
                 test_id="225aaee8-7c7f-4faf-810b-407b551e9f2a", scenario_ids=['2eaf04e3-090a-4c13-b923-ac86de1b6db1'], benchmark_id="31ea606b-681a-437a-85b9-7c81d4ccc287"
             ),
        #
             # KPI-SF-077: Similarity state to unperturbed situation (Power Grid)
             "7fe4210f-1253-411c-ba03-49d8b37c71fa": TestRunner_KPI_SF_077_Power_Grid(
                 test_id="7fe4210f-1253-411c-ba03-49d8b37c71fa", scenario_ids=['4523d73e-427a-42a1-b841-c9668373fafb'], benchmark_id="31ea606b-681a-437a-85b9-7c81d4ccc287"
             ),
        #
        #     # KPI-RS-091: Reflection on operator trust  (Power Grid)
        #     "0eb49c18-fbf7-4797-b4c5-6dabc9795ddb": TestRunner_KPI_RS_091_Power_Grid(
        #         test_id="0eb49c18-fbf7-4797-b4c5-6dabc9795ddb", scenario_ids=['b26f54c0-146a-473c-82a6-fcecbedc9cbc'], benchmark_id="df309815-8ec0-4a6f-9d0b-dc3dbfc9055a"
        #     ),
        #
        #     # KPI-RS-092: Reflection on operator agency  (Power Grid)
        #     "b7d124f7-998a-416e-81a0-424cb6192755": TestRunner_KPI_RS_092_Power_Grid(
        #         test_id="b7d124f7-998a-416e-81a0-424cb6192755", scenario_ids=['284e2178-d34c-4dcb-8868-21b4d2310744'], benchmark_id="df309815-8ec0-4a6f-9d0b-dc3dbfc9055a"
        #     ),
        #
        #     # KPI-RS-093: Reflection on operator de-skilling  (Power Grid)
        #     "e860b491-42b1-416a-b2fb-bea9bd8a249d": TestRunner_KPI_RS_093_Power_Grid(
        #         test_id="e860b491-42b1-416a-b2fb-bea9bd8a249d", scenario_ids=['f253c68c-bd5e-43a2-8d72-77ab9caccb0d'], benchmark_id="df309815-8ec0-4a6f-9d0b-dc3dbfc9055a"
        #     ),
        #
        #     # KPI-RS-094: Reflection on over-reliance  (Power Grid)
        #     "ff1099da-8b25-4dff-8f64-c7ce1d5c9d9d": TestRunner_KPI_RS_094_Power_Grid(
        #         test_id="ff1099da-8b25-4dff-8f64-c7ce1d5c9d9d", scenario_ids=['44daf1f7-1449-408e-baa6-9d5c29dc50f0'], benchmark_id="df309815-8ec0-4a6f-9d0b-dc3dbfc9055a"
        #     ),
        #
        #     # KPI-RS-095: Reflection on additional training  (Power Grid)
        #     "afa5288f-3a6f-4933-8bd9-1761d4c97c34": TestRunner_KPI_RS_095_Power_Grid(
        #         test_id="afa5288f-3a6f-4933-8bd9-1761d4c97c34", scenario_ids=['9cfb4458-2418-4a83-a7f3-0c27758c4752'], benchmark_id="df309815-8ec0-4a6f-9d0b-dc3dbfc9055a"
        #     ),
        #
        #     # KPI-RS-096: Reflection on biases  (Power Grid)
        #     "b67c3a5a-c157-420c-9d3c-8c949413a79b": TestRunner_KPI_RS_096_Power_Grid(
        #         test_id="b67c3a5a-c157-420c-9d3c-8c949413a79b", scenario_ids=['7dfff29e-c251-459e-b591-295d796b328a'], benchmark_id="df309815-8ec0-4a6f-9d0b-dc3dbfc9055a"
        #     ),
        #
        #     # KPI-PS-097: Predicted long-term adoption  (Power Grid)
        #     "f322d495-95e0-49ab-9439-156228593328": TestRunner_KPI_PS_097_Power_Grid(
        #         test_id="f322d495-95e0-49ab-9439-156228593328", scenario_ids=['d885d30d-408c-4b88-9a25-add134aca45b'], benchmark_id="df309815-8ec0-4a6f-9d0b-dc3dbfc9055a"
        #     ),
        #
    }
)


# https://docs.celeryq.dev/en/stable/userguide/tasks.html#bound-tasks: A task being bound means the first argument to the task will always be the task instance (self).
# https://docs.celeryq.dev/en/stable/userguide/tasks.html#names: Every task must have a unique name.
@app.task(name="PowerGrid", bind=True)
def orchestrator(self, submission_data_url: str, tests: List[str] = None, **kwargs):
    submission_id = self.request.id
    benchmark_id = orchestrator.name
    logger.info(
        f"Queue/task {benchmark_id} received submission {submission_id} with submission_data_url={submission_data_url} for tests={tests}"
    )
    return power_grid_orchestrator.run(
        submission_id=submission_id,
        submission_data_url=submission_data_url,
        tests=tests,
    )

