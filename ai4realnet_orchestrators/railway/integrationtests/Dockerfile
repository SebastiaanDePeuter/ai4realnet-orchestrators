# Unset (default): full's FROM below resolves the "deps" stage name to the local stage defined here, i.e.
# today's behavior — build deps then extend it. Set (by flatland-benchmarks' docker-compose.yml, via CI once a
# matching ci-deps-<hash> tag has been pulled): resolves to that external image instead, skipping the local
# deps build entirely. See flatland-benchmarks' CLAUDE.md ("CI image reuse and caching") for the full mechanism.
ARG RAILWAY_DEPS_IMAGE=deps

FROM ghcr.io/flatland-association/flatland-rl:latest-py3.12 AS deps

# This makes output not buffer and return immediately, nice for seeing results in stdout
ENV PYTHONUNBUFFERED=1

# Install Docker
RUN apt-get update && curl -fsSL https://get.docker.com | sh


# setup user and workdir
RUN addgroup runner && adduser --uid 1000 --system --disabled-password --home /app --ingroup runner runner
WORKDIR /app

RUN apt-get update \
 && apt-get install -y sudo

RUN usermod -a -G sudo runner

RUN echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers

# configure runner
USER runner

RUN sudo ls /tmp
RUN sudo chown -R runner /app

# Copied to a scratch path rather than ./ai4realnet_orchestrators/ (which full's whole-tree ADD below
# populates fresh) so this stage's narrow, hash-friendly copy isn't redundantly overwritten by identical bytes.
ADD ./evaluation/ai4realnet_orchestrators/requirements.txt /tmp/deps-requirements/requirements.txt
ADD ./evaluation/ai4realnet_orchestrators/ai4realnet_orchestrators/railway/requirements.txt /tmp/deps-requirements/railway-requirements.txt

RUN python -m pip install -U -r /tmp/deps-requirements/requirements.txt -r /tmp/deps-requirements/railway-requirements.txt


FROM ${RAILWAY_DEPS_IMAGE} AS full

ADD ./evaluation/ai4realnet_orchestrators ./ai4realnet_orchestrators/

RUN find .
