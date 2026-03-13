# Powergrid usecase
1. Install ai4realnet_orchestrator following the repo's README
2. Install powergrid requirements `pip install -r power_grid_requirements.txt`
3. Launch the orchestrator:

```
export DOMAIN="PowerGrid"
export BACKEND_URL=rpc://
export BROKER_URL=amqps://<USER - get it from Flatland>:<PW - get it from Flatland>@rabbitmq-int.flatland.cloud:5671//
export CLIENT_ID=<get it from Flatland>
export CLIENT_SECRET=<get it from Flatland>
export FAB_API_URL=https://ai4realnet-int.flatland.cloud:8000
export RABBITMQ_KEYFILE=.../certs/tls.key # get it from Flatland
export RABBITMQ_CERTFILE=.../certs/tls.crt # get it from Flatland
export RABBITMQ_CA_CERTS=.../certs/ca.crt # get it from Flatland

export BENCHMARK_ID=4b0be731-8371-4e4e-a673-b630187b0bb8

python -m celery -A ai4realnet_orchestrators.power_grid.orchestrator worker -P solo -n orchestrator@%n -Q ${DOMAIN} --logfile=$PWD/power-grid-orchestrator.log --pidfile=$PWD/power-grid-orchestrator.pid --detach
```
We use celery solo pool implementation (*-P solo* CLI parameter) instead of the default implementation because pypowsybl creates processes which interfere with celery multiprocessing and creates deadlocks. 

**Troubleshooting**

The --detach parameter can be unstable in combination with python loggers.
In case detach does not work, you can create a service with systemd or use a command like *nohup* :
```
python -m celery -A ai4realnet_orchestrators.power_grid.orchestrator worker \
-P solo -n orchestrator@%n -Q ${DOMAIN} --logfile=/opt/ai4realnet/ai4realnet-orchestrators/power-grid-orchestrator.log --pidfile=/opt/ai4realnet/ai4realnet-orchestrators/power-grid-orchestrator.pid \
> /opt/ai4realnet/ai4realnet-orchestrators/power-grid-orchestrator.log 2>&1 &

# Don't forge to detach the process if used in a bash file
disown
```
