#!/bin/bash
export CARLA_ROOT=~/CARLA
# export CARLA_SERVER=${CARLA_ROOT}/CarlaUE4.sh
export PYTHONPATH=$PYTHONPATH:${CARLA_ROOT}/PythonAPI
export PYTHONPATH=$PYTHONPATH:${CARLA_ROOT}/PythonAPI/carla
export PYTHONPATH=$PYTHONPATH:$CARLA_ROOT/PythonAPI/carla/dist/carla-0.9.10-py3.7-linux-x86_64.egg
export PYTHONPATH=$PYTHONPATH:leaderboard
export PYTHONPATH=$PYTHONPATH:leaderboard/team_code
export PYTHONPATH=$PYTHONPATH:scenario_runner

export LEADERBOARD_ROOT=leaderboard
export CHALLENGE_TRACK_CODENAME=SENSORS
export DEBUG_CHALLENGE=0
export REPETITIONS=1 
export RESUME=True

export WEATHER=HardRainNoon
#export WEATHER=HardRainNoon # ClearNoon, ClearSunset, CloudyNoon, CloudySunset, HardRainNoon, HardRainSunset, MidRainSunset, MidRainyNoon, SoftRainNoon, SoftRainSunset, WetCloudyNoon, WetCloudySunset, WetNoon, WetSunset
export SCENARIO_ROUTE=03
export ROUTE_TYPE=long
export SAVE_PATH=/media/fachri/banyak/endtoend/data/ADVERSARIAL/${WEATHER}-new/Town${SCENARIO_ROUTE}_${ROUTE_TYPE}-1 # ADVERSARIAL NORMAL
export ROUTES=leaderboard/data/all_routes/routes_town${SCENARIO_ROUTE}_${ROUTE_TYPE}.xml #lihat di /leaderboard/data/all_routes
export SCENARIOS=leaderboard/data/scenarios/town${SCENARIO_ROUTE}_all_scenarios_new.json #look at leaderboard/data/scenarios  town05_all_scenarios OR no_scenarios.json old(more npc)/new(less npc)
export PORT=2002 # same as the carla server port
export TM_PORT=2052 # port for traffic manager, required when spawning multiple servers/clients
export CHECKPOINT_ENDPOINT=${SAVE_PATH}/eval_result.json # results file
export TEAM_AGENT=leaderboard/team_code/expert.py

python3 ${LEADERBOARD_ROOT}/leaderboard/leaderboard_evaluator.py \
--scenarios=${SCENARIOS}  \
--routes=${ROUTES} \
--repetitions=${REPETITIONS} \
--track=${CHALLENGE_TRACK_CODENAME} \
--checkpoint=${CHECKPOINT_ENDPOINT} \
--agent=${TEAM_AGENT} \
--debug=${DEBUG_CHALLENGE} \
--record=${RECORD_PATH} \
--resume=${RESUME} \
--port=${PORT} \
--trafficManagerPort=${TM_PORT} \
--weather=${WEATHER} #buat ganti2 weather

