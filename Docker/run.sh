#!/bin/bash
username=$(id -u -n)
uid=$(id -u)
gid=$(id -g)
title=$1
container_num=$2
container_name="${username}_${title}_${container_num}"
ROOT="/work"
WORKING_DIR="${ROOT}/$title"

docker run --rm --gpus all -it \
   --shm-size=16gb \
   --name "$container_name" \
   --workdir "$WORKING_DIR" \
   -u "$uid":"$gid" \
   $title \
   bash


