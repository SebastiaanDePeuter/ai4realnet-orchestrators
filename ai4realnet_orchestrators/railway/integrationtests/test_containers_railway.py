import json
import logging
import os
import ssl
import subprocess
import time
import uuid
from io import TextIOWrapper, BytesIO
from typing import List, Optional

import pytest
from celery import Celery
from testcontainers.compose import DockerCompose

from ai4realnet_orchestrators.fab_oauth_utils import backend_application_flow
from ai4realnet_orchestrators.s3_utils import s3_utils
from fab_clientlib import DefaultApi, Configuration, ApiClient
from fab_clientlib.models.submissions_post_request import SubmissionsPostRequest

logger = logging.getLogger(__name__)


@pytest.fixture(scope="module")
def test_containers_fixture():
  # set env var ATTENDED to True if docker-demo.yml is already up and running
  if os.environ.get("ATTENDED", "False").lower() == "true":
    yield
    return

  global basic

  start_time = time.time()
  basic = DockerCompose(context="../..", profiles=["full"])
  logger.info("/ start docker compose down")
  basic.stop()
  duration = time.time() - start_time
  logger.info(f"\\ end docker compose down. Took {duration:.2f} seconds.")
  start_time = time.time()
  logger.info("/ start docker compose up")
  try:
    basic.start()
    duration = time.time() - start_time
    logger.info(f"\\ end docker compose up. Took {duration:.2f} seconds.")

    submission_id = str(uuid.uuid4())
    yield submission_id

    # TODO workaround for testcontainers not supporting streaming to logger
    start_time = time.time()
    logger.info("/ start get docker compose logs")
    stdout, stderr = basic.get_logs()
    logger.info("stdout from docker compose")
    logger.info(stdout)
    logger.warning("stderr from docker compose")
    logger.warning(stderr)
    duration = time.time() - start_time
    logger.info(f"\\ end get docker compose logs. Took {duration:.2f} seconds.")

    start_time = time.time()
    logger.info("/ start docker compose down")
    basic.stop()
    duration = time.time() - start_time
    logger.info(f"\\ end docker down. Took {duration:.2f} seconds.")
  except BaseException as e:
    print("An exception occurred during running docker compose:")
    print(e)
    stdout, stderr = basic.get_logs()
    print(stdout)
    print(stderr)
    raise e


def run_task(task_queue_name: str, submission_id: str, submission_data_url: str, tests: List[str], **kwargs):
  start_time = time.time()
  app = Celery(
    broker="amqps://guest:guest@localhost:5671",
    backend="rpc://",
    broker_use_ssl={
      'keyfile': "../../docker/rabbitmq/certs/client_localhost_key.pem",
      'certfile': "../../docker/rabbitmq/certs/client_localhost_certificate.pem",
      'ca_certs': "../../docker/rabbitmq/certs/ca_certificate.pem",
      'cert_reqs': ssl.CERT_REQUIRED
    }
  )
  logger.info(f"/ Start waiting for submission from portal for submission_id={submission_id}.....")
  time.sleep(3)
  inspect = app.control.inspect()
  while True:
    logger.info(inspect.active().values())
    active = [e2 for e in inspect.active().values() for e2 in e]
    logger.info(active)
    if len(active) > 0:
      seconds = 5
      time.sleep(seconds)
    else:
      break

  duration = time.time() - start_time
  logger.info(
    f"\\ End waiting for submission from portal for submission_id={submission_id}. Took {duration} seconds.")



@pytest.mark.usefixtures("test_containers_fixture")
@pytest.mark.integration
def test_runner_kpi_pf_026_railway():
  benchmark_id = "3b1bdca6-ed90-4938-bd63-fd657aa7dcd7"
  task_queue_name = 'Railway'  # Celery: queue name = task name
  test_id = "98ceb866-5479-47e6-a735-81292de8ca65"  # Celery: passed in "tests" key of kwargs when Celery task is submitted
  # TODO use released version
  submission_data_url = "ghcr.io/flatland-association/flatland-baselines-deadlock-avoidance-heuristic:latest"  # Celery: passed in "submission_data_url" key of kwargs when Celery task is submitted

  def _verify_kpi_pf_026(test_results):
    assert len(test_results.body) == 1
    test_results = test_results.body[0]
    assert test_results.scenario_scorings[0].scorings[0].field_key == "punctuality"
    assert test_results.scenario_scorings[0].scorings[0].score == 0.9285714285714286
    assert test_results.scenario_scorings[0].scorings[1].field_key == "success_rate"
    assert test_results.scenario_scorings[0].scorings[1].score == 1.0
    assert test_results.scenario_scorings[1].scorings[0].field_key == "punctuality"
    assert test_results.scenario_scorings[1].scorings[0].score == 1.0
    assert test_results.scenario_scorings[1].scorings[1].field_key == "success_rate"
    assert test_results.scenario_scorings[1].scorings[1].score == 1.0
    assert test_results.scorings[0].field_key == "punctuality"
    assert test_results.scorings[0].score == 0.9642857142857143

  submission_id = _generic_run(benchmark_id, submission_data_url, task_queue_name, test_id, _verify_kpi_pf_026)

  s3 = s3_utils.get_boto_client("minioadmin", "minioadmin", "http://localhost:9000")
  for scenario_id, expected_keys in {
    '5a60713d-01f2-4d32-9867-21904629e254': {
      f"ai4realnet/submissions/{submission_id}/98ceb866-5479-47e6-a735-81292de8ca65/5a60713d-01f2-4d32-9867-21904629e254/serialised_state/5a60713d-01f2-4d32-9867-21904629e254_step0180.pkl",
      f"ai4realnet/submissions/{submission_id}/98ceb866-5479-47e6-a735-81292de8ca65/5a60713d-01f2-4d32-9867-21904629e254/serialised_state/5a60713d-01f2-4d32-9867-21904629e254_step0160.pkl",
      f"ai4realnet/submissions/{submission_id}/98ceb866-5479-47e6-a735-81292de8ca65/5a60713d-01f2-4d32-9867-21904629e254/serialised_state/5a60713d-01f2-4d32-9867-21904629e254_step0020.pkl",
      f"ai4realnet/submissions/{submission_id}/98ceb866-5479-47e6-a735-81292de8ca65/5a60713d-01f2-4d32-9867-21904629e254/serialised_state/5a60713d-01f2-4d32-9867-21904629e254_step0100.pkl",
      f"ai4realnet/submissions/{submission_id}/98ceb866-5479-47e6-a735-81292de8ca65/5a60713d-01f2-4d32-9867-21904629e254/serialised_state/5a60713d-01f2-4d32-9867-21904629e254_step0140.pkl",
      f"ai4realnet/submissions/{submission_id}/98ceb866-5479-47e6-a735-81292de8ca65/5a60713d-01f2-4d32-9867-21904629e254/serialised_state/5a60713d-01f2-4d32-9867-21904629e254_step0070.pkl",
      f"ai4realnet/submissions/{submission_id}/98ceb866-5479-47e6-a735-81292de8ca65/5a60713d-01f2-4d32-9867-21904629e254/serialised_state/5a60713d-01f2-4d32-9867-21904629e254_step0030.pkl",
      f"ai4realnet/submissions/{submission_id}/98ceb866-5479-47e6-a735-81292de8ca65/5a60713d-01f2-4d32-9867-21904629e254/serialised_state/5a60713d-01f2-4d32-9867-21904629e254_step0010.pkl",
      f"ai4realnet/submissions/{submission_id}/98ceb866-5479-47e6-a735-81292de8ca65/5a60713d-01f2-4d32-9867-21904629e254/serialised_state/5a60713d-01f2-4d32-9867-21904629e254_step0170.pkl",
      f"ai4realnet/submissions/{submission_id}/98ceb866-5479-47e6-a735-81292de8ca65/5a60713d-01f2-4d32-9867-21904629e254/serialised_state/5a60713d-01f2-4d32-9867-21904629e254_step0000.pkl",
      f"ai4realnet/submissions/{submission_id}/98ceb866-5479-47e6-a735-81292de8ca65/5a60713d-01f2-4d32-9867-21904629e254/serialised_state/5a60713d-01f2-4d32-9867-21904629e254_step0090.pkl",
      f"ai4realnet/submissions/{submission_id}/98ceb866-5479-47e6-a735-81292de8ca65/5a60713d-01f2-4d32-9867-21904629e254/serialised_state/5a60713d-01f2-4d32-9867-21904629e254_step0150.pkl",
      f"ai4realnet/submissions/{submission_id}/98ceb866-5479-47e6-a735-81292de8ca65/5a60713d-01f2-4d32-9867-21904629e254/serialised_state/5a60713d-01f2-4d32-9867-21904629e254_step0110.pkl",
      f"ai4realnet/submissions/{submission_id}/98ceb866-5479-47e6-a735-81292de8ca65/5a60713d-01f2-4d32-9867-21904629e254/serialised_state/5a60713d-01f2-4d32-9867-21904629e254.pkl",
      f"ai4realnet/submissions/{submission_id}/98ceb866-5479-47e6-a735-81292de8ca65/5a60713d-01f2-4d32-9867-21904629e254/serialised_state/5a60713d-01f2-4d32-9867-21904629e254_step0080.pkl",
      f"ai4realnet/submissions/{submission_id}/98ceb866-5479-47e6-a735-81292de8ca65/5a60713d-01f2-4d32-9867-21904629e254/serialised_state/5a60713d-01f2-4d32-9867-21904629e254_step0130.pkl",
      f"ai4realnet/submissions/{submission_id}/98ceb866-5479-47e6-a735-81292de8ca65/5a60713d-01f2-4d32-9867-21904629e254/serialised_state/5a60713d-01f2-4d32-9867-21904629e254_step0050.pkl",
      f"ai4realnet/submissions/{submission_id}/98ceb866-5479-47e6-a735-81292de8ca65/5a60713d-01f2-4d32-9867-21904629e254/serialised_state/5a60713d-01f2-4d32-9867-21904629e254_step0060.pkl",
      f"ai4realnet/submissions/{submission_id}/98ceb866-5479-47e6-a735-81292de8ca65/5a60713d-01f2-4d32-9867-21904629e254/serialised_state/5a60713d-01f2-4d32-9867-21904629e254_step0120.pkl",
      f"ai4realnet/submissions/{submission_id}/98ceb866-5479-47e6-a735-81292de8ca65/5a60713d-01f2-4d32-9867-21904629e254/serialised_state/5a60713d-01f2-4d32-9867-21904629e254_step0040.pkl",
      f"ai4realnet/submissions/{submission_id}/98ceb866-5479-47e6-a735-81292de8ca65/5a60713d-01f2-4d32-9867-21904629e254/event_logs/ActionEvents.discrete_action.tsv",
      f"ai4realnet/submissions/{submission_id}/98ceb866-5479-47e6-a735-81292de8ca65/5a60713d-01f2-4d32-9867-21904629e254/event_logs/TrainMovementEvents.trains_arrived.tsv",
      f"ai4realnet/submissions/{submission_id}/98ceb866-5479-47e6-a735-81292de8ca65/5a60713d-01f2-4d32-9867-21904629e254/event_logs/TrainMovementEvents.trains_positions.tsv",
      f"ai4realnet/submissions/{submission_id}/98ceb866-5479-47e6-a735-81292de8ca65/5a60713d-01f2-4d32-9867-21904629e254/event_logs/TrainMovementEvents.trains_rewards_dones_infos.tsv"},
    '0db72a40-43e8-477b-89b3-a7bd1224660a': {
      f"ai4realnet/submissions/{submission_id}/98ceb866-5479-47e6-a735-81292de8ca65/0db72a40-43e8-477b-89b3-a7bd1224660a/serialised_state/0db72a40-43e8-477b-89b3-a7bd1224660a_step0160.pkl",
      f"ai4realnet/submissions/{submission_id}/98ceb866-5479-47e6-a735-81292de8ca65/0db72a40-43e8-477b-89b3-a7bd1224660a/serialised_state/0db72a40-43e8-477b-89b3-a7bd1224660a_step0020.pkl",
      f"ai4realnet/submissions/{submission_id}/98ceb866-5479-47e6-a735-81292de8ca65/0db72a40-43e8-477b-89b3-a7bd1224660a/serialised_state/0db72a40-43e8-477b-89b3-a7bd1224660a_step0100.pkl",
      f"ai4realnet/submissions/{submission_id}/98ceb866-5479-47e6-a735-81292de8ca65/0db72a40-43e8-477b-89b3-a7bd1224660a/serialised_state/0db72a40-43e8-477b-89b3-a7bd1224660a_step0140.pkl",
      f"ai4realnet/submissions/{submission_id}/98ceb866-5479-47e6-a735-81292de8ca65/0db72a40-43e8-477b-89b3-a7bd1224660a/serialised_state/0db72a40-43e8-477b-89b3-a7bd1224660a_step0070.pkl",
      f"ai4realnet/submissions/{submission_id}/98ceb866-5479-47e6-a735-81292de8ca65/0db72a40-43e8-477b-89b3-a7bd1224660a/serialised_state/0db72a40-43e8-477b-89b3-a7bd1224660a_step0030.pkl",
      f"ai4realnet/submissions/{submission_id}/98ceb866-5479-47e6-a735-81292de8ca65/0db72a40-43e8-477b-89b3-a7bd1224660a/serialised_state/0db72a40-43e8-477b-89b3-a7bd1224660a_step0010.pkl",
      f"ai4realnet/submissions/{submission_id}/98ceb866-5479-47e6-a735-81292de8ca65/0db72a40-43e8-477b-89b3-a7bd1224660a/serialised_state/0db72a40-43e8-477b-89b3-a7bd1224660a_step0170.pkl",
      f"ai4realnet/submissions/{submission_id}/98ceb866-5479-47e6-a735-81292de8ca65/0db72a40-43e8-477b-89b3-a7bd1224660a/serialised_state/0db72a40-43e8-477b-89b3-a7bd1224660a_step0000.pkl",
      f"ai4realnet/submissions/{submission_id}/98ceb866-5479-47e6-a735-81292de8ca65/0db72a40-43e8-477b-89b3-a7bd1224660a/serialised_state/0db72a40-43e8-477b-89b3-a7bd1224660a_step0090.pkl",
      f"ai4realnet/submissions/{submission_id}/98ceb866-5479-47e6-a735-81292de8ca65/0db72a40-43e8-477b-89b3-a7bd1224660a/serialised_state/0db72a40-43e8-477b-89b3-a7bd1224660a_step0150.pkl",
      f"ai4realnet/submissions/{submission_id}/98ceb866-5479-47e6-a735-81292de8ca65/0db72a40-43e8-477b-89b3-a7bd1224660a/serialised_state/0db72a40-43e8-477b-89b3-a7bd1224660a_step0110.pkl",
      f"ai4realnet/submissions/{submission_id}/98ceb866-5479-47e6-a735-81292de8ca65/0db72a40-43e8-477b-89b3-a7bd1224660a/serialised_state/0db72a40-43e8-477b-89b3-a7bd1224660a.pkl",
      f"ai4realnet/submissions/{submission_id}/98ceb866-5479-47e6-a735-81292de8ca65/0db72a40-43e8-477b-89b3-a7bd1224660a/serialised_state/0db72a40-43e8-477b-89b3-a7bd1224660a_step0080.pkl",
      f"ai4realnet/submissions/{submission_id}/98ceb866-5479-47e6-a735-81292de8ca65/0db72a40-43e8-477b-89b3-a7bd1224660a/serialised_state/0db72a40-43e8-477b-89b3-a7bd1224660a_step0130.pkl",
      f"ai4realnet/submissions/{submission_id}/98ceb866-5479-47e6-a735-81292de8ca65/0db72a40-43e8-477b-89b3-a7bd1224660a/serialised_state/0db72a40-43e8-477b-89b3-a7bd1224660a_step0050.pkl",
      f"ai4realnet/submissions/{submission_id}/98ceb866-5479-47e6-a735-81292de8ca65/0db72a40-43e8-477b-89b3-a7bd1224660a/serialised_state/0db72a40-43e8-477b-89b3-a7bd1224660a_step0060.pkl",
      f"ai4realnet/submissions/{submission_id}/98ceb866-5479-47e6-a735-81292de8ca65/0db72a40-43e8-477b-89b3-a7bd1224660a/serialised_state/0db72a40-43e8-477b-89b3-a7bd1224660a_step0120.pkl",
      f"ai4realnet/submissions/{submission_id}/98ceb866-5479-47e6-a735-81292de8ca65/0db72a40-43e8-477b-89b3-a7bd1224660a/serialised_state/0db72a40-43e8-477b-89b3-a7bd1224660a_step0040.pkl",
      f"ai4realnet/submissions/{submission_id}/98ceb866-5479-47e6-a735-81292de8ca65/0db72a40-43e8-477b-89b3-a7bd1224660a/event_logs/ActionEvents.discrete_action.tsv",
      f"ai4realnet/submissions/{submission_id}/98ceb866-5479-47e6-a735-81292de8ca65/0db72a40-43e8-477b-89b3-a7bd1224660a/event_logs/TrainMovementEvents.trains_arrived.tsv",
      f"ai4realnet/submissions/{submission_id}/98ceb866-5479-47e6-a735-81292de8ca65/0db72a40-43e8-477b-89b3-a7bd1224660a/event_logs/TrainMovementEvents.trains_positions.tsv",
      f"ai4realnet/submissions/{submission_id}/98ceb866-5479-47e6-a735-81292de8ca65/0db72a40-43e8-477b-89b3-a7bd1224660a/event_logs/TrainMovementEvents.trains_rewards_dones_infos.tsv"},
  }.items():
    listing = s3.list_objects_v2(
      Bucket='fab-demo-results',
      Prefix=f'ai4realnet/submissions/{submission_id}/{test_id}/{scenario_id}',
    )
    actual_keys = {c["Key"] for c in listing["Contents"]}
    assert len(listing["Contents"]) > 0
    assert actual_keys == expected_keys


@pytest.mark.usefixtures("test_containers_fixture")
@pytest.mark.integration
def test_runner_kpi_nf_045_railway():
  benchmark_id = "4b0be731-8371-4e4e-a673-b630187b0bb8"
  task_queue_name = 'Railway'  # Celery: queue name = task name
  test_id = "e075d4a7-5cda-4d3c-83ac-69a0db1d74dd"  # Celery: passed in "tests" key of kwargs when Celery task is submitted
  submission_data_url = "ghcr.io/flatland-association/flatland-baselines-deadlock-avoidance-heuristic:latest"  # Celery: passed in "submission_data_url" key of kwargs when Celery task is submitted

  def _verify_kpi_nf_045(test_results):
    assert len(test_results.body) == 1
    test_results = test_results.body[0]

    assert test_results.scenario_scorings[0].scorings[0].field_key == "network_impact_propagation"
    assert test_results.scenario_scorings[0].scorings[0].score == 0.8571428571428572
    assert test_results.scenario_scorings[0].scorings[1].field_key == "success_rate_1"
    assert test_results.scenario_scorings[0].scorings[1].score == 1.0
    assert test_results.scenario_scorings[0].scorings[2].field_key == "punctuality_1"
    assert test_results.scenario_scorings[0].scorings[2].score == 0.8571428571428571
    assert test_results.scenario_scorings[0].scorings[3].field_key == "success_rate_2"
    assert test_results.scenario_scorings[0].scorings[3].score == 1.0
    assert test_results.scenario_scorings[0].scorings[4].field_key == "punctuality_2"
    assert test_results.scenario_scorings[0].scorings[4].score == 0.8571428571428571

    assert test_results.scenario_scorings[1].scorings[0].score == 1
    assert test_results.scenario_scorings[1].scorings[1].field_key == "success_rate_1"
    assert test_results.scenario_scorings[1].scorings[1].score == 1.0
    assert test_results.scenario_scorings[1].scorings[2].field_key == "punctuality_1"
    assert test_results.scenario_scorings[1].scorings[2].score == 1
    assert test_results.scenario_scorings[1].scorings[3].field_key == "success_rate_2"
    assert test_results.scenario_scorings[1].scorings[3].score == 1.0
    assert test_results.scenario_scorings[1].scorings[4].field_key == "punctuality_2"
    assert test_results.scenario_scorings[1].scorings[4].score == 1

    assert test_results.scorings[0].field_key == "network_impact_propagation"
    assert test_results.scorings[0].score == 0.9285714285714286

  submission_id = _generic_run(benchmark_id, submission_data_url, task_queue_name, test_id, _verify_kpi_nf_045)

  s3 = s3_utils.get_boto_client("minioadmin", "minioadmin", "http://localhost:9000")
  for scenario_id, expected_key in {
    'bb6302f1-0dc2-43ed-976b-4e5d3126006a': {
      f"ai4realnet/submissions/{submission_id}/e075d4a7-5cda-4d3c-83ac-69a0db1d74dd/bb6302f1-0dc2-43ed-976b-4e5d3126006a/no_malfunction/serialised_state/bb6302f1-0dc2-43ed-976b-4e5d3126006a_step0045.pkl",
      f"ai4realnet/submissions/{submission_id}/e075d4a7-5cda-4d3c-83ac-69a0db1d74dd/bb6302f1-0dc2-43ed-976b-4e5d3126006a/no_malfunction/serialised_state/bb6302f1-0dc2-43ed-976b-4e5d3126006a_step0108.pkl",
      f"ai4realnet/submissions/{submission_id}/e075d4a7-5cda-4d3c-83ac-69a0db1d74dd/bb6302f1-0dc2-43ed-976b-4e5d3126006a/no_malfunction/serialised_state/bb6302f1-0dc2-43ed-976b-4e5d3126006a_step0052.pkl",
      f"ai4realnet/submissions/{submission_id}/e075d4a7-5cda-4d3c-83ac-69a0db1d74dd/bb6302f1-0dc2-43ed-976b-4e5d3126006a/no_malfunction/serialised_state/bb6302f1-0dc2-43ed-976b-4e5d3126006a_step0033.pkl",
      f"ai4realnet/submissions/{submission_id}/e075d4a7-5cda-4d3c-83ac-69a0db1d74dd/bb6302f1-0dc2-43ed-976b-4e5d3126006a/no_malfunction/serialised_state/bb6302f1-0dc2-43ed-976b-4e5d3126006a_step0016.pkl",
      f"ai4realnet/submissions/{submission_id}/e075d4a7-5cda-4d3c-83ac-69a0db1d74dd/bb6302f1-0dc2-43ed-976b-4e5d3126006a/no_malfunction/serialised_state/bb6302f1-0dc2-43ed-976b-4e5d3126006a_step0147.pkl",
      f"ai4realnet/submissions/{submission_id}/e075d4a7-5cda-4d3c-83ac-69a0db1d74dd/bb6302f1-0dc2-43ed-976b-4e5d3126006a/no_malfunction/serialised_state/bb6302f1-0dc2-43ed-976b-4e5d3126006a_step0121.pkl",
      f"ai4realnet/submissions/{submission_id}/e075d4a7-5cda-4d3c-83ac-69a0db1d74dd/bb6302f1-0dc2-43ed-976b-4e5d3126006a/no_malfunction/serialised_state/bb6302f1-0dc2-43ed-976b-4e5d3126006a_step0056.pkl",
      f"ai4realnet/submissions/{submission_id}/e075d4a7-5cda-4d3c-83ac-69a0db1d74dd/bb6302f1-0dc2-43ed-976b-4e5d3126006a/no_malfunction/serialised_state/bb6302f1-0dc2-43ed-976b-4e5d3126006a_step0095.pkl",
      f"ai4realnet/submissions/{submission_id}/e075d4a7-5cda-4d3c-83ac-69a0db1d74dd/bb6302f1-0dc2-43ed-976b-4e5d3126006a/no_malfunction/serialised_state/bb6302f1-0dc2-43ed-976b-4e5d3126006a_step0090.pkl",
      f"ai4realnet/submissions/{submission_id}/e075d4a7-5cda-4d3c-83ac-69a0db1d74dd/bb6302f1-0dc2-43ed-976b-4e5d3126006a/no_malfunction/serialised_state/bb6302f1-0dc2-43ed-976b-4e5d3126006a_step0117.pkl",
      f"ai4realnet/submissions/{submission_id}/e075d4a7-5cda-4d3c-83ac-69a0db1d74dd/bb6302f1-0dc2-43ed-976b-4e5d3126006a/no_malfunction/serialised_state/bb6302f1-0dc2-43ed-976b-4e5d3126006a_step0069.pkl",
      f"ai4realnet/submissions/{submission_id}/e075d4a7-5cda-4d3c-83ac-69a0db1d74dd/bb6302f1-0dc2-43ed-976b-4e5d3126006a/no_malfunction/serialised_state/bb6302f1-0dc2-43ed-976b-4e5d3126006a_step0053.pkl",
      f"ai4realnet/submissions/{submission_id}/e075d4a7-5cda-4d3c-83ac-69a0db1d74dd/bb6302f1-0dc2-43ed-976b-4e5d3126006a/no_malfunction/serialised_state/bb6302f1-0dc2-43ed-976b-4e5d3126006a_step0123.pkl",
      f"ai4realnet/submissions/{submission_id}/e075d4a7-5cda-4d3c-83ac-69a0db1d74dd/bb6302f1-0dc2-43ed-976b-4e5d3126006a/no_malfunction/serialised_state/bb6302f1-0dc2-43ed-976b-4e5d3126006a_step0012.pkl",
      f"ai4realnet/submissions/{submission_id}/e075d4a7-5cda-4d3c-83ac-69a0db1d74dd/bb6302f1-0dc2-43ed-976b-4e5d3126006a/no_malfunction/event_logs/ActionEvents.discrete_action.tsv",
      f"ai4realnet/submissions/{submission_id}/e075d4a7-5cda-4d3c-83ac-69a0db1d74dd/bb6302f1-0dc2-43ed-976b-4e5d3126006a/no_malfunction/event_logs/TrainMovementEvents.trains_arrived.tsv",
      f"ai4realnet/submissions/{submission_id}/e075d4a7-5cda-4d3c-83ac-69a0db1d74dd/bb6302f1-0dc2-43ed-976b-4e5d3126006a/no_malfunction/event_logs/TrainMovementEvents.trains_positions.tsv",
      f"ai4realnet/submissions/{submission_id}/e075d4a7-5cda-4d3c-83ac-69a0db1d74dd/bb6302f1-0dc2-43ed-976b-4e5d3126006a/no_malfunction/event_logs/TrainMovementEvents.trains_rewards_dones_infos.tsv",
      f"ai4realnet/submissions/{submission_id}/e075d4a7-5cda-4d3c-83ac-69a0db1d74dd/bb6302f1-0dc2-43ed-976b-4e5d3126006a/with_malfunction/serialised_state/bb6302f1-0dc2-43ed-976b-4e5d3126006a_step0070.pkl",
      f"ai4realnet/submissions/{submission_id}/e075d4a7-5cda-4d3c-83ac-69a0db1d74dd/bb6302f1-0dc2-43ed-976b-4e5d3126006a/with_malfunction/serialised_state/bb6302f1-0dc2-43ed-976b-4e5d3126006a_step0100.pkl",
      f"ai4realnet/submissions/{submission_id}/e075d4a7-5cda-4d3c-83ac-69a0db1d74dd/bb6302f1-0dc2-43ed-976b-4e5d3126006a/with_malfunction/serialised_state/bb6302f1-0dc2-43ed-976b-4e5d3126006a_step0080.pkl",
      f"ai4realnet/submissions/{submission_id}/e075d4a7-5cda-4d3c-83ac-69a0db1d74dd/bb6302f1-0dc2-43ed-976b-4e5d3126006a/with_malfunction/serialised_state/bb6302f1-0dc2-43ed-976b-4e5d3126006a_step0040.pkl",
      f"ai4realnet/submissions/{submission_id}/e075d4a7-5cda-4d3c-83ac-69a0db1d74dd/bb6302f1-0dc2-43ed-976b-4e5d3126006a/with_malfunction/serialised_state/bb6302f1-0dc2-43ed-976b-4e5d3126006a_step0180.pkl",
      f"ai4realnet/submissions/{submission_id}/e075d4a7-5cda-4d3c-83ac-69a0db1d74dd/bb6302f1-0dc2-43ed-976b-4e5d3126006a/with_malfunction/serialised_state/bb6302f1-0dc2-43ed-976b-4e5d3126006a_step0000.pkl",
      f"ai4realnet/submissions/{submission_id}/e075d4a7-5cda-4d3c-83ac-69a0db1d74dd/bb6302f1-0dc2-43ed-976b-4e5d3126006a/with_malfunction/serialised_state/bb6302f1-0dc2-43ed-976b-4e5d3126006a_step0140.pkl",
      f"ai4realnet/submissions/{submission_id}/e075d4a7-5cda-4d3c-83ac-69a0db1d74dd/bb6302f1-0dc2-43ed-976b-4e5d3126006a/with_malfunction/serialised_state/bb6302f1-0dc2-43ed-976b-4e5d3126006a_step0120.pkl",
      f"ai4realnet/submissions/{submission_id}/e075d4a7-5cda-4d3c-83ac-69a0db1d74dd/bb6302f1-0dc2-43ed-976b-4e5d3126006a/with_malfunction/serialised_state/bb6302f1-0dc2-43ed-976b-4e5d3126006a_step0010.pkl",
      f"ai4realnet/submissions/{submission_id}/e075d4a7-5cda-4d3c-83ac-69a0db1d74dd/bb6302f1-0dc2-43ed-976b-4e5d3126006a/with_malfunction/serialised_state/bb6302f1-0dc2-43ed-976b-4e5d3126006a_step0030.pkl",
      f"ai4realnet/submissions/{submission_id}/e075d4a7-5cda-4d3c-83ac-69a0db1d74dd/bb6302f1-0dc2-43ed-976b-4e5d3126006a/with_malfunction/serialised_state/bb6302f1-0dc2-43ed-976b-4e5d3126006a_step0130.pkl",
      f"ai4realnet/submissions/{submission_id}/e075d4a7-5cda-4d3c-83ac-69a0db1d74dd/bb6302f1-0dc2-43ed-976b-4e5d3126006a/with_malfunction/serialised_state/bb6302f1-0dc2-43ed-976b-4e5d3126006a_step0110.pkl",
      f"ai4realnet/submissions/{submission_id}/e075d4a7-5cda-4d3c-83ac-69a0db1d74dd/bb6302f1-0dc2-43ed-976b-4e5d3126006a/with_malfunction/serialised_state/bb6302f1-0dc2-43ed-976b-4e5d3126006a.pkl",
      f"ai4realnet/submissions/{submission_id}/e075d4a7-5cda-4d3c-83ac-69a0db1d74dd/bb6302f1-0dc2-43ed-976b-4e5d3126006a/with_malfunction/serialised_state/bb6302f1-0dc2-43ed-976b-4e5d3126006a_step0050.pkl",
      f"ai4realnet/submissions/{submission_id}/e075d4a7-5cda-4d3c-83ac-69a0db1d74dd/bb6302f1-0dc2-43ed-976b-4e5d3126006a/with_malfunction/serialised_state/bb6302f1-0dc2-43ed-976b-4e5d3126006a_step0060.pkl",
      f"ai4realnet/submissions/{submission_id}/e075d4a7-5cda-4d3c-83ac-69a0db1d74dd/bb6302f1-0dc2-43ed-976b-4e5d3126006a/with_malfunction/serialised_state/bb6302f1-0dc2-43ed-976b-4e5d3126006a_step0020.pkl",
      f"ai4realnet/submissions/{submission_id}/e075d4a7-5cda-4d3c-83ac-69a0db1d74dd/bb6302f1-0dc2-43ed-976b-4e5d3126006a/with_malfunction/serialised_state/bb6302f1-0dc2-43ed-976b-4e5d3126006a_step0150.pkl",
      f"ai4realnet/submissions/{submission_id}/e075d4a7-5cda-4d3c-83ac-69a0db1d74dd/bb6302f1-0dc2-43ed-976b-4e5d3126006a/with_malfunction/serialised_state/bb6302f1-0dc2-43ed-976b-4e5d3126006a_step0160.pkl",
      f"ai4realnet/submissions/{submission_id}/e075d4a7-5cda-4d3c-83ac-69a0db1d74dd/bb6302f1-0dc2-43ed-976b-4e5d3126006a/with_malfunction/serialised_state/bb6302f1-0dc2-43ed-976b-4e5d3126006a_step0170.pkl",
      f"ai4realnet/submissions/{submission_id}/e075d4a7-5cda-4d3c-83ac-69a0db1d74dd/bb6302f1-0dc2-43ed-976b-4e5d3126006a/with_malfunction/serialised_state/bb6302f1-0dc2-43ed-976b-4e5d3126006a_step0090.pkl",
      f"ai4realnet/submissions/{submission_id}/e075d4a7-5cda-4d3c-83ac-69a0db1d74dd/bb6302f1-0dc2-43ed-976b-4e5d3126006a/with_malfunction/event_logs/ActionEvents.discrete_action.tsv",
      f"ai4realnet/submissions/{submission_id}/e075d4a7-5cda-4d3c-83ac-69a0db1d74dd/bb6302f1-0dc2-43ed-976b-4e5d3126006a/with_malfunction/event_logs/TrainMovementEvents.trains_arrived.tsv",
      f"ai4realnet/submissions/{submission_id}/e075d4a7-5cda-4d3c-83ac-69a0db1d74dd/bb6302f1-0dc2-43ed-976b-4e5d3126006a/with_malfunction/event_logs/TrainMovementEvents.trains_positions.tsv",
      f"ai4realnet/submissions/{submission_id}/e075d4a7-5cda-4d3c-83ac-69a0db1d74dd/bb6302f1-0dc2-43ed-976b-4e5d3126006a/with_malfunction/event_logs/TrainMovementEvents.trains_rewards_dones_infos.tsv"},
    'f84dcf0c-4bde-460b-9139-ea76e3694267': {
      f"ai4realnet/submissions/{submission_id}/e075d4a7-5cda-4d3c-83ac-69a0db1d74dd/f84dcf0c-4bde-460b-9139-ea76e3694267/no_malfunction/serialised_state/f84dcf0c-4bde-460b-9139-ea76e3694267_step0045.pkl",
      f"ai4realnet/submissions/{submission_id}/e075d4a7-5cda-4d3c-83ac-69a0db1d74dd/f84dcf0c-4bde-460b-9139-ea76e3694267/no_malfunction/serialised_state/f84dcf0c-4bde-460b-9139-ea76e3694267_step0108.pkl",
      f"ai4realnet/submissions/{submission_id}/e075d4a7-5cda-4d3c-83ac-69a0db1d74dd/f84dcf0c-4bde-460b-9139-ea76e3694267/no_malfunction/serialised_state/f84dcf0c-4bde-460b-9139-ea76e3694267_step0052.pkl",
      f"ai4realnet/submissions/{submission_id}/e075d4a7-5cda-4d3c-83ac-69a0db1d74dd/f84dcf0c-4bde-460b-9139-ea76e3694267/no_malfunction/serialised_state/f84dcf0c-4bde-460b-9139-ea76e3694267_step0033.pkl",
      f"ai4realnet/submissions/{submission_id}/e075d4a7-5cda-4d3c-83ac-69a0db1d74dd/f84dcf0c-4bde-460b-9139-ea76e3694267/no_malfunction/serialised_state/f84dcf0c-4bde-460b-9139-ea76e3694267_step0016.pkl",
      f"ai4realnet/submissions/{submission_id}/e075d4a7-5cda-4d3c-83ac-69a0db1d74dd/f84dcf0c-4bde-460b-9139-ea76e3694267/no_malfunction/serialised_state/f84dcf0c-4bde-460b-9139-ea76e3694267_step0147.pkl",
      f"ai4realnet/submissions/{submission_id}/e075d4a7-5cda-4d3c-83ac-69a0db1d74dd/f84dcf0c-4bde-460b-9139-ea76e3694267/no_malfunction/serialised_state/f84dcf0c-4bde-460b-9139-ea76e3694267_step0121.pkl",
      f"ai4realnet/submissions/{submission_id}/e075d4a7-5cda-4d3c-83ac-69a0db1d74dd/f84dcf0c-4bde-460b-9139-ea76e3694267/no_malfunction/serialised_state/f84dcf0c-4bde-460b-9139-ea76e3694267_step0056.pkl",
      f"ai4realnet/submissions/{submission_id}/e075d4a7-5cda-4d3c-83ac-69a0db1d74dd/f84dcf0c-4bde-460b-9139-ea76e3694267/no_malfunction/serialised_state/f84dcf0c-4bde-460b-9139-ea76e3694267_step0095.pkl",
      f"ai4realnet/submissions/{submission_id}/e075d4a7-5cda-4d3c-83ac-69a0db1d74dd/f84dcf0c-4bde-460b-9139-ea76e3694267/no_malfunction/serialised_state/f84dcf0c-4bde-460b-9139-ea76e3694267_step0090.pkl",
      f"ai4realnet/submissions/{submission_id}/e075d4a7-5cda-4d3c-83ac-69a0db1d74dd/f84dcf0c-4bde-460b-9139-ea76e3694267/no_malfunction/serialised_state/f84dcf0c-4bde-460b-9139-ea76e3694267_step0117.pkl",
      f"ai4realnet/submissions/{submission_id}/e075d4a7-5cda-4d3c-83ac-69a0db1d74dd/f84dcf0c-4bde-460b-9139-ea76e3694267/no_malfunction/serialised_state/f84dcf0c-4bde-460b-9139-ea76e3694267_step0069.pkl",
      f"ai4realnet/submissions/{submission_id}/e075d4a7-5cda-4d3c-83ac-69a0db1d74dd/f84dcf0c-4bde-460b-9139-ea76e3694267/no_malfunction/serialised_state/f84dcf0c-4bde-460b-9139-ea76e3694267_step0053.pkl",
      f"ai4realnet/submissions/{submission_id}/e075d4a7-5cda-4d3c-83ac-69a0db1d74dd/f84dcf0c-4bde-460b-9139-ea76e3694267/no_malfunction/serialised_state/f84dcf0c-4bde-460b-9139-ea76e3694267_step0123.pkl",
      f"ai4realnet/submissions/{submission_id}/e075d4a7-5cda-4d3c-83ac-69a0db1d74dd/f84dcf0c-4bde-460b-9139-ea76e3694267/no_malfunction/serialised_state/f84dcf0c-4bde-460b-9139-ea76e3694267_step0012.pkl",
      f"ai4realnet/submissions/{submission_id}/e075d4a7-5cda-4d3c-83ac-69a0db1d74dd/f84dcf0c-4bde-460b-9139-ea76e3694267/no_malfunction/event_logs/ActionEvents.discrete_action.tsv",
      f"ai4realnet/submissions/{submission_id}/e075d4a7-5cda-4d3c-83ac-69a0db1d74dd/f84dcf0c-4bde-460b-9139-ea76e3694267/no_malfunction/event_logs/TrainMovementEvents.trains_arrived.tsv",
      f"ai4realnet/submissions/{submission_id}/e075d4a7-5cda-4d3c-83ac-69a0db1d74dd/f84dcf0c-4bde-460b-9139-ea76e3694267/no_malfunction/event_logs/TrainMovementEvents.trains_positions.tsv",
      f"ai4realnet/submissions/{submission_id}/e075d4a7-5cda-4d3c-83ac-69a0db1d74dd/f84dcf0c-4bde-460b-9139-ea76e3694267/no_malfunction/event_logs/TrainMovementEvents.trains_rewards_dones_infos.tsv",
      f"ai4realnet/submissions/{submission_id}/e075d4a7-5cda-4d3c-83ac-69a0db1d74dd/f84dcf0c-4bde-460b-9139-ea76e3694267/with_malfunction/serialised_state/f84dcf0c-4bde-460b-9139-ea76e3694267_step0070.pkl",
      f"ai4realnet/submissions/{submission_id}/e075d4a7-5cda-4d3c-83ac-69a0db1d74dd/f84dcf0c-4bde-460b-9139-ea76e3694267/with_malfunction/serialised_state/f84dcf0c-4bde-460b-9139-ea76e3694267_step0100.pkl",
      f"ai4realnet/submissions/{submission_id}/e075d4a7-5cda-4d3c-83ac-69a0db1d74dd/f84dcf0c-4bde-460b-9139-ea76e3694267/with_malfunction/serialised_state/f84dcf0c-4bde-460b-9139-ea76e3694267_step0080.pkl",
      f"ai4realnet/submissions/{submission_id}/e075d4a7-5cda-4d3c-83ac-69a0db1d74dd/f84dcf0c-4bde-460b-9139-ea76e3694267/with_malfunction/serialised_state/f84dcf0c-4bde-460b-9139-ea76e3694267_step0040.pkl",
      f"ai4realnet/submissions/{submission_id}/e075d4a7-5cda-4d3c-83ac-69a0db1d74dd/f84dcf0c-4bde-460b-9139-ea76e3694267/with_malfunction/serialised_state/f84dcf0c-4bde-460b-9139-ea76e3694267_step0180.pkl",
      f"ai4realnet/submissions/{submission_id}/e075d4a7-5cda-4d3c-83ac-69a0db1d74dd/f84dcf0c-4bde-460b-9139-ea76e3694267/with_malfunction/serialised_state/f84dcf0c-4bde-460b-9139-ea76e3694267_step0000.pkl",
      f"ai4realnet/submissions/{submission_id}/e075d4a7-5cda-4d3c-83ac-69a0db1d74dd/f84dcf0c-4bde-460b-9139-ea76e3694267/with_malfunction/serialised_state/f84dcf0c-4bde-460b-9139-ea76e3694267_step0140.pkl",
      f"ai4realnet/submissions/{submission_id}/e075d4a7-5cda-4d3c-83ac-69a0db1d74dd/f84dcf0c-4bde-460b-9139-ea76e3694267/with_malfunction/serialised_state/f84dcf0c-4bde-460b-9139-ea76e3694267_step0120.pkl",
      f"ai4realnet/submissions/{submission_id}/e075d4a7-5cda-4d3c-83ac-69a0db1d74dd/f84dcf0c-4bde-460b-9139-ea76e3694267/with_malfunction/serialised_state/f84dcf0c-4bde-460b-9139-ea76e3694267_step0010.pkl",
      f"ai4realnet/submissions/{submission_id}/e075d4a7-5cda-4d3c-83ac-69a0db1d74dd/f84dcf0c-4bde-460b-9139-ea76e3694267/with_malfunction/serialised_state/f84dcf0c-4bde-460b-9139-ea76e3694267_step0030.pkl",
      f"ai4realnet/submissions/{submission_id}/e075d4a7-5cda-4d3c-83ac-69a0db1d74dd/f84dcf0c-4bde-460b-9139-ea76e3694267/with_malfunction/serialised_state/f84dcf0c-4bde-460b-9139-ea76e3694267_step0130.pkl",
      f"ai4realnet/submissions/{submission_id}/e075d4a7-5cda-4d3c-83ac-69a0db1d74dd/f84dcf0c-4bde-460b-9139-ea76e3694267/with_malfunction/serialised_state/f84dcf0c-4bde-460b-9139-ea76e3694267_step0110.pkl",
      f"ai4realnet/submissions/{submission_id}/e075d4a7-5cda-4d3c-83ac-69a0db1d74dd/f84dcf0c-4bde-460b-9139-ea76e3694267/with_malfunction/serialised_state/f84dcf0c-4bde-460b-9139-ea76e3694267.pkl",
      f"ai4realnet/submissions/{submission_id}/e075d4a7-5cda-4d3c-83ac-69a0db1d74dd/f84dcf0c-4bde-460b-9139-ea76e3694267/with_malfunction/serialised_state/f84dcf0c-4bde-460b-9139-ea76e3694267_step0050.pkl",
      f"ai4realnet/submissions/{submission_id}/e075d4a7-5cda-4d3c-83ac-69a0db1d74dd/f84dcf0c-4bde-460b-9139-ea76e3694267/with_malfunction/serialised_state/f84dcf0c-4bde-460b-9139-ea76e3694267_step0060.pkl",
      f"ai4realnet/submissions/{submission_id}/e075d4a7-5cda-4d3c-83ac-69a0db1d74dd/f84dcf0c-4bde-460b-9139-ea76e3694267/with_malfunction/serialised_state/f84dcf0c-4bde-460b-9139-ea76e3694267_step0020.pkl",
      f"ai4realnet/submissions/{submission_id}/e075d4a7-5cda-4d3c-83ac-69a0db1d74dd/f84dcf0c-4bde-460b-9139-ea76e3694267/with_malfunction/serialised_state/f84dcf0c-4bde-460b-9139-ea76e3694267_step0150.pkl",
      f"ai4realnet/submissions/{submission_id}/e075d4a7-5cda-4d3c-83ac-69a0db1d74dd/f84dcf0c-4bde-460b-9139-ea76e3694267/with_malfunction/serialised_state/f84dcf0c-4bde-460b-9139-ea76e3694267_step0160.pkl",
      f"ai4realnet/submissions/{submission_id}/e075d4a7-5cda-4d3c-83ac-69a0db1d74dd/f84dcf0c-4bde-460b-9139-ea76e3694267/with_malfunction/serialised_state/f84dcf0c-4bde-460b-9139-ea76e3694267_step0170.pkl",
      f"ai4realnet/submissions/{submission_id}/e075d4a7-5cda-4d3c-83ac-69a0db1d74dd/f84dcf0c-4bde-460b-9139-ea76e3694267/with_malfunction/serialised_state/f84dcf0c-4bde-460b-9139-ea76e3694267_step0090.pkl",
      f"ai4realnet/submissions/{submission_id}/e075d4a7-5cda-4d3c-83ac-69a0db1d74dd/f84dcf0c-4bde-460b-9139-ea76e3694267/with_malfunction/event_logs/ActionEvents.discrete_action.tsv",
      f"ai4realnet/submissions/{submission_id}/e075d4a7-5cda-4d3c-83ac-69a0db1d74dd/f84dcf0c-4bde-460b-9139-ea76e3694267/with_malfunction/event_logs/TrainMovementEvents.trains_arrived.tsv",
      f"ai4realnet/submissions/{submission_id}/e075d4a7-5cda-4d3c-83ac-69a0db1d74dd/f84dcf0c-4bde-460b-9139-ea76e3694267/with_malfunction/event_logs/TrainMovementEvents.trains_positions.tsv",
      f"ai4realnet/submissions/{submission_id}/e075d4a7-5cda-4d3c-83ac-69a0db1d74dd/f84dcf0c-4bde-460b-9139-ea76e3694267/with_malfunction/event_logs/TrainMovementEvents.trains_rewards_dones_infos.tsv"}
  }.items():
    listing = s3.list_objects_v2(
      Bucket='fab-demo-results',
      Prefix=f'ai4realnet/submissions/{submission_id}/{test_id}/{scenario_id}',
    )
    print(listing)
    assert len(listing["Contents"]) > 0


def _generic_run(benchmark_id, submission_data_url, task_queue_name, test_id, verify):
  try:
    token = backend_application_flow(
      client_id='fab-client-credentials',
      client_secret='top-secret',
      token_url='http://localhost:8081/realms/flatland/protocol/openid-connect/token',
    )
    logger.info(token)
    fab = DefaultApi(ApiClient(configuration=Configuration(host="http://localhost:8000", access_token=token["access_token"])))
    ret = fab.submissions_post(SubmissionsPostRequest(
      name="fancy",
      benchmark_id=benchmark_id,
      submission_data_url=submission_data_url,
      code_repository="",
      test_ids=[test_id]
    ))

    logger.info(f"submission posted {ret}")
    submission_id = ret.body.id

    run_task(task_queue_name, submission_id, submission_data_url, tests=[test_id])

    token = backend_application_flow(
      client_id='fab-client-credentials',
      client_secret='top-secret',
      token_url='http://localhost:8081/realms/flatland/protocol/openid-connect/token',
    )
    logger.info(token)
    fab = DefaultApi(ApiClient(configuration=Configuration(host="http://localhost:8000", access_token=token["access_token"])))

    test_results = fab.results_submissions_submission_id_tests_test_ids_get(
      submission_id=submission_id,
      test_ids=[test_id])
    logger.info("results downloaed")
    _pretty_print(test_results)
    verify(test_results)
    return submission_id

  except BaseException as e:
    exec_with_logging(["docker", "ps"])
    debug = []
    try:
      logger.info("/ Logs from docker compose")

      stdo, stderr = exec_with_logging(["docker", "compose", "--profile", "full", "logs", ],
                                       log_level_stdout=logging.INFO,
                                       log_level_stderr=logging.WARN,
                                       collect=True)
      debug += stdo
      debug += stderr
      logger.info("\\ Logs from docker compose")
    except:
      logger.warning("Could not fetch logs from docker compose")
    raise Exception(str(e) + ": " + '\n'.join(debug)) from e


# https://stackoverflow.com/questions/21953835/run-subprocess-and-print-output-to-logging
def exec_with_logging(exec_args: List[str], log_level_stdout=logging.DEBUG, log_level_stderr=logging.WARN, collect: bool = False):
  logger.debug(f"/ Start %s", exec_args)
  try:
    proc = subprocess.Popen(exec_args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    stdout, stderr = proc.communicate()
    stdo = log_subprocess_output(TextIOWrapper(BytesIO(stdout)), level=log_level_stdout, label=str(exec_args), collect=collect)
    stde = log_subprocess_output(TextIOWrapper(BytesIO(stderr)), level=log_level_stderr, label=str(exec_args), collect=collect)
    logger.debug("\\ End %s", exec_args)
    return stdo, stde
  except (OSError, subprocess.CalledProcessError) as exception:
    logger.error(stderr)
    raise RuntimeError(f"Failed to run {exec_args}. Stdout={stdout}. Stderr={stderr}") from exception


# https://stackoverflow.com/questions/21953835/run-subprocess-and-print-output-to-logging
def log_subprocess_output(pipe, level=logging.DEBUG, label="", collect: bool = False) -> Optional[List[str]]:
  s = []
  for line in pipe.readlines():
    logger.log(level, "[from subprocess %s] %s", label, line)
    if collect:
      s.append(line)
  if collect:
    return s
  return None


# https://stackoverflow.com/questions/36588126/uuid-is-not-json-serializable
class UUIDEncoder(json.JSONEncoder):
  def default(self, obj):
    if isinstance(obj, uuid.UUID):
      # if the obj is uuid, we simply return the value of uuid
      return obj.hex
    return json.JSONEncoder.default(self, obj)


def _pretty_print(submissions):
  print(json.dumps(submissions.to_dict(), indent=4, cls=UUIDEncoder))
