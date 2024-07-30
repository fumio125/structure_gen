#!/bin/bash
username=$(id -u -n)
uid=$(id -u)
groupname=$(id -g -n)
gid=$(id -g)
tag="latest"
title=$1

docker build --build-arg USERNAME=${username} \
       --build-arg UID=${uid} \
       --build-arg GROUPNAME=${groupname} \
       --build-arg GID=${gid} \
       -t $title \
       -f ./Docker/Dockerfile .



