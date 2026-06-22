import json
import os
from time import sleep
from uuid import UUID

import pytest

from ai4realnet_orchestrators.fab_oauth_utils import backend_application_flow
from ai4realnet_orchestrators.railway.orchestrator import railway_orchestrator
from fab_clientlib import DefaultApi, ApiClient, Configuration, SubmissionsPostRequest


@pytest.mark.skip(reason="run manually")
def test_get_submissions():
  fab = _get_fab()
  test_runner = _get_test_runner()

  benchmark_id = test_runner.benchmark_id
  submissions = fab.submissions_get(
    benchmark_ids=[benchmark_id],
    # submitted_by="service-account-ai4realnet-client-credentials"
    submitted_by="9d63bdc9-8438-49e2-afc4-53d4de1bfcee"
  )
  _pretty_print(submissions)


@pytest.mark.skip(reason="run manually")
def test_submit(
  submission_name="S046",
  test="KPI-PF-026"  # Punctuality,
  # test="KPI-NF-045"# Network Impact Propagation,
):
  fab = _get_fab()
  _pretty_print(fab.health_live_get())
  test_runner = _get_test_runner(test)
  benchmark_id = test_runner.benchmark_id
  test_id = test_runner.test_id

  submitted = fab.submissions_post(
    SubmissionsPostRequest(
      benchmark_id=benchmark_id,
      name=submission_name,
      # TODO use released version
      submission_data_url="ghcr.io/flatland-association/flatland-baselines-deadlock-avoidance-heuristic:latest",
      test_ids=[test_id])
  )

  _pretty_print(submitted)

  submission_id = submitted.body.id
  fab.results_submissions_submission_ids_get(submission_ids=[submission_id])
  results = fab.results_submissions_submission_ids_get([submission_id])
  test_scorings = results.body[0].test_scorings[0]
  assert len(test_scorings.scorings) == 1
  test_scoring = test_scorings.scorings[0]
  print(test_scoring)
  while test_scoring.score is None:
    print(".")
    fab.results_submissions_submission_ids_get(submission_ids=[submission_id])
    results = fab.results_submissions_submission_ids_get([submission_id])
    test_scoring = results.body[0].test_scorings[0].scorings[0]
    # print(test_scoring)
    sleep(2)
  print("scored")
  print(test_scoring)

  patched = fab.submissions_submission_ids_patch(submission_ids=[submission_id])
  _pretty_print(patched)


def _get_test_runner(label: str = "KPI-PF-026"):
  # KPI-PF-026: Punctuality (Railway)
  # "98ceb866-5479-47e6-a735-81292de8ca65": TestRunner_KPI_PF_026_Railway(
  if label == "KPI-PF-026":
    print(
      "https://ai4realnet-int.flatland.cloud/suites/0ca46887-897a-463f-bf83-c6cd6269a977/3b1bdca6-ed90-4938-bd63-fd657aa7dcd7/tests/98ceb866-5479-47e6-a735-81292de8ca65")
    return railway_orchestrator.test_runners["98ceb866-5479-47e6-a735-81292de8ca65"]
  # KPI-AF-029: AI Response time (Railway)
  # "1e226684-a836-468d-9929-b95bbf2f88dc": TestRunner_KPI_AF_029_Railway(
  # test_runner = railway_orchestrator.test_runners["1e226684-a836-468d-9929-b95bbf2f88dc"]
  # KPI-NF-045: Network Impact Propagation (Railway)
  if label == "KPI-NF-045":
    print(
      "https://ai4realnet-int.flatland.cloud/suites/0ca46887-897a-463f-bf83-c6cd6269a977/4b0be731-8371-4e4e-a673-b630187b0bb8/tests/e075d4a7-5cda-4d3c-83ac-69a0db1d74dd")
    # "e075d4a7-5cda-4d3c-83ac-69a0db1d74dd": TestRunner_KPI_NF_045_Railway(
    return railway_orchestrator.test_runners["e075d4a7-5cda-4d3c-83ac-69a0db1d74dd"]
  # KPI-AF-051: AI-Agent Scalability Testing (Railway)
  if label == "KPI-AF-051":
    print(
      "https://ai4realnet-int.flatland.cloud/suites/0ca46887-897a-463f-bf83-c6cd6269a977/16706c82-75df-4969-932d-a7f5c941eca2/tests/b2e91a79-1390-414f-bf5d-8a6fd93c6080")
    return railway_orchestrator.test_runners["b2e91a79-1390-414f-bf5d-8a6fd93c6080"]


def _get_fab():
  FAB_API_URL = os.environ.get("FAB_API_URL", "https://ai4realnet-int.flatland.cloud:8000")
  CLIENT_ID = os.environ.get("CLIENT_ID", 'ai4realnet-client-credentials')
  CLIENT_SECRET = os.environ.get("CLIENT_SECRET")
  TOKEN_URL = os.environ.get("TOKEN_URL", "https://keycloak.flatland.cloud/realms/flatland/protocol/openid-connect/token")
  token = backend_application_flow(CLIENT_ID, CLIENT_SECRET, TOKEN_URL)
  print(token)
  fab = DefaultApi(ApiClient(configuration=Configuration(host=FAB_API_URL, access_token=token["access_token"])))
  return fab


# https://stackoverflow.com/questions/36588126/uuid-is-not-json-serializable
class UUIDEncoder(json.JSONEncoder):
  def default(self, obj):
    if isinstance(obj, UUID):
      # if the obj is uuid, we simply return the value of uuid
      return obj.hex
    return json.JSONEncoder.default(self, obj)


def _pretty_print(submissions):
  print(json.dumps(submissions.to_dict(), indent=4, cls=UUIDEncoder))
