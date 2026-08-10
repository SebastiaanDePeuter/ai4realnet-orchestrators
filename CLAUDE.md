# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

AI4REALNET Validation Campaign Hub Orchestrator: domain-specific orchestrator and test-runner implementations that integrate with the Flatland Association's Validation Campaign Hub (FAB, [flatland-benchmarks](https://github.com/flatland-association/flatland-benchmarks)). Each domain (`railway`, `atm`, `power_grid`) runs as a Celery worker consuming a domain-named queue, dispatches submissions to `TestRunner` implementations, and uploads results back to FAB via [fab-clientlib](https://pypi.org/project/fab-clientlib/) (OAuth2 client-credentials flow). Trunk-based development: `main` is trunk, feature branches are short-lived, PRs merge directly into `main`.

Three loop types (mirrors FAB's own `CLOSED`/`INTERACTIVE`/`OFFLINE` `tests.loop` column): **closed-loop** (orchestrator runs end-to-end, uploads results, closes submission), **interactive-loop** (orchestrator uploads partial results automatically, a Human Factors Researcher completes/closes the submission manually), **offline-loop** (no queue message at all — results are uploaded manually via the FAB UI or fab-clientlib CLI). A message is sent to the domain queue for closed-loop and interactive-loop, but never for offline-loop.

## Commands

```bash
# Install shared deps + one domain's extra deps, e.g. railway:
python -m pip install -r requirements.txt -r requirements-dev.txt -r ai4realnet_orchestrators/railway/requirements.txt

# Unit tests (excludes domains requiring their own extra/heavy deps):
pytest ai4realnet_orchestrators/ --ignore=ai4realnet_orchestrators/railway --ignore=ai4realnet_orchestrators/power_grid --ignore=ai4realnet_orchestrators/atm
# Run a single test:
pytest ai4realnet_orchestrators/test_orchestrator.py::test_orchestrator

# Railway domain unit tests (excludes the docker-compose-based integration test):
python -m pip install -r requirements.txt -r requirements-dev.txt -r ai4realnet_orchestrators/railway/requirements.txt
pytest -s -m "not integration" ai4realnet_orchestrators/railway

# Railway integration test — needs a sibling flatland-benchmarks checkout (see below) and Docker:
pytest -s -m integration ai4realnet_orchestrators/railway

# Start a domain orchestrator locally (see each domain's README for full env var list):
python -m celery -A ai4realnet_orchestrators.railway.orchestrator worker -l info -n orchestrator@%n -Q ${DOMAIN}
```

`power_grid` is its own `uv`-managed subproject (`ai4realnet_orchestrators/power_grid/pyproject.toml` + `uv.lock`, `requires-python = ">=3.10"`) — its dependencies (`grid2op`, `lightsim2grid`, `stable-baselines3`, plus two `git+https://...@main`/`@master` research-code deps) are declared there, not in the shared root `requirements.txt`.

**PowerGrid orchestrator must run with Celery's solo pool** (`-P solo`, not the default prefork pool) — `pypowsybl` (used by `lightsim2grid`) spawns processes that interfere with Celery's own multiprocessing and deadlock otherwise (`ai4realnet_orchestrators/power_grid/README.md`).

`pytest.ini` sets `python_classes = !TestRunner` — pytest's default `Test*`-prefix collection would otherwise try to collect the imported `TestRunner` base class itself (from `ai4realnet_orchestrators/test_runner.py`) as a test class.

## Architecture

### `Orchestrator`/`TestRunner` base classes (`ai4realnet_orchestrators/{orchestrator,test_runner}.py`)

`Orchestrator.run(submission_id, submission_data_url, tests, fab=None)` looks up each `test_id` in its `test_runners: Dict[str, TestRunner]` map, calls `.init()` then `.run()` on it, and POSTs results to FAB via `fab.results_submissions_submission_id_tests_test_ids_post(...)`. Any failure is wrapped as `TaskExecutionError`. A `TestRunner` subclass need only implement `run_scenario(scenario_id, submission_id) -> dict` (default field name `primary`); `TestRunner.run()` iterates `self.scenario_ids` and calls it once per scenario, unless overridden entirely.

### Per-domain wiring (`ai4realnet_orchestrators/{railway,atm,power_grid}/`)

Each domain defines its own `celery.Celery(broker=BROKER_URL, backend=BACKEND_URL, queue=BENCHMARK_ID, broker_use_ssl={...})` app and registers a single bound task (name matching the domain, e.g. `"Railway"`) that calls into an `Orchestrator` instance mapping `test_id` (UUID) → `TestRunner` instance (`atm/orchestrator.py` builds this map inline; railway delegates to `railway/orchestrator_definitions.py`'s `railway_orchestrator`). This mirrors flatland-benchmarks' own Celery queue/task naming convention (queue name = task name = `BENCHMARK_ID`, or a `queue` override on a single-test submission for `CAMPAIGN`-style KPI routing).

### Railway integration test (`ai4realnet_orchestrators/railway/integrationtests/`)

`conftest.py`'s `test_containers_fixture` is a near-duplicate of flatland-benchmarks' own `test_common/docker_compose_fixture.py`, but **always runs an explicit, unconditional `docker compose build`** (via `basic._run_command(cmd=[...] + ["build"])`) before `basic.start()` — unlike the shared fixture, there's no `build=` toggle here, and `pull_policy`/`image:` switches on the Compose services are irrelevant to that explicit `build` subcommand. flatland-benchmarks' CI works around this via two additive Compose overrides (`docker-compose.ci-prebuilt-backend.yml`/`-frontend.yml`, one per service) that each `!reset`-clear `build:` for just that one service, included independently whenever that specific service's prebuilt image is confirmed to exist — so reusing only `fab-backend` or only `fab-frontend` doesn't force the other one's build-skip along with it (see that repo's `CLAUDE.md`, "CI image reuse and caching"). This repo's own `.github/workflows/checks.yaml` mirrors that same mechanism in its `test-ai4realnet-orchestrator-railway-integration` job, via its own copy of flatland-benchmarks' `.github/actions/determine-image-reuse-strategy` composite action (`repo-root: flatland-benchmarks`, pointing at the nested checkout) — kept in sync by hand, since a cross-repo `uses:` reference back to flatland-benchmarks was rejected as untestable from either repo's sandbox. That action checks `fab-backend`/`fab-frontend`/the orchestrator-deps image independently of each other (three separate `docker manifest inspect` calls) so one image's hash changing doesn't force the other two to rebuild too; this repo's own `checks.yaml` passes it the same `backend-hash-paths`/`frontend-hash-paths`/`orchestrator-deps-hash-paths` inputs (defined once, as YAML anchors, in its top-level `env:` block) that flatland-benchmarks' `checks.yml` passes from its `build-images` matrix — the two repos' literal path lists must still match by hand, same as the composite-action copies themselves.

This repo's own `integrationtests/Dockerfile` also has a `deps`/`full` split (mirroring flatland-benchmarks' orchestrator Dockerfile), with `full`'s `FROM ${RAILWAY_DEPS_IMAGE}` defaulting to the local `deps` stage *name* (`ARG RAILWAY_DEPS_IMAGE=deps`, declared before the first `FROM`) rather than a published `:latest` tag — there is no publish workflow for this test-only image to default to. flatland-benchmarks' `build-images` job builds+pushes just the `deps` target as `ghcr.io/flatland-association/ai4realnet-orchestrator-railway:ci-deps-<hash>`; this job's own call to `.github/actions/determine-railway-deps-reuse` (a composite action, kept independent of the `determine-image-reuse-strategy` step above so a missing tag here never gates whether `fab-backend`/`fab-frontend`/orchestrator-deps get reused — its own copy lives in this repo too, alongside `determine-image-reuse-strategy`) checks/pulls that tag and passes it through as the `RAILWAY_DEPS_IMAGE` build arg via `docker-compose.yml`.

### Cross-repo relationship with flatland-benchmarks

This repo is normally consumed by flatland-benchmarks as an external clone (`evaluation/ai4realnet_orchestrators` there — gitignored, cloned per its `ai4realnet-orchestrators-ref` env var; see flatland-benchmarks' own `CLAUDE.md`). This repo's own CI (`.github/workflows/checks.yaml`) runs the reverse direction: it checks out flatland-benchmarks (ref set via this repo's `flatland-benchmarks-ref` env var) into a nested `flatland-benchmarks/` directory, then checks itself out again at `flatland-benchmarks/evaluation/ai4realnet_orchestrators` to reproduce that exact layout, and runs the railway integration test against flatland-benchmarks' real `docker-compose.yml`. Both `*-ref` env vars are normally pinned to a released version/tag on each side (each marked `# DEPENDENCY SWITCH` in the respective workflow file) — when cross-repo work is being tested before either side has released, both are deliberately pointed at matching feature-branch names instead, and should be reverted to released refs once that work merges.

## Testing conventions

- Unit tests use [`mockito`](https://mockito-python.readthedocs.io/) (`mock()`/`when()`/`verify()`), not `unittest.mock`.
- The `integration` pytest marker (declared in `pytest.ini`) gates tests requiring the flatland-benchmarks docker-compose stack; run everything else with `-m "not integration"`.
