#!/usr/bin/env bash

day=0
time=$((3600*24*${day}))

pids=$(ps -e -o pid,etimes,command,user | grep wandb-service | grep -v grep | awk -v time="$time" -v user="$USER" '{if($2>time&&$4==user) print $1}')
for pid in ${pids}; do
	echo ${pid} 
	kill -9 "${pid}";
done
