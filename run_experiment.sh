#!/bin/bash

trap 'exit' SIGINT SIGTERM SIGHUP SIGQUIT

if [ -z "$1" ]
then
  printf "Iterates the feedback loop experiment over many parameters configs using the grid search.\n"
  printf "Parameters inside this scripts are configurable: step, usage, adherence."
  printf "\n\n"
  printf "Usage: run_experiment.sh <pipeline_name>\n\n"
  printf "See ./experiment.yml for more details\n"
  exit
fi

for w in {1..9}.0
do
  export w &&
  mldev --config .mldev/config.yaml run -f ./experiment.yml --no-commit $1
done
