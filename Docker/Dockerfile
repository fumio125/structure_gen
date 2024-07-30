FROM nvidia/cuda:11.6.2-cudnn8-devel-ubuntu20.04

ARG USERNAME
ARG UID
ARG GROUPNAME
ARG GID
ARG USE_SHELL=/bin/bash
ARG PASSWORD=$USERNAME
ARG ROOT_PASSWORD='su'

MAINTAINER $USERNAME

#if stop build at timezone setting, comentout this line
ENV DEBIAN_FRONTEND=noninteractive

RUN apt update -y && \
    apt upgrade -y

RUN apt install -y tzdata
RUN apt install -y git vim sudo libgl1-mesa-dev

#set root password
RUN echo 'root:'$ROOT_PASSWORD | chpasswd

#make group and user
RUN groupadd -g $GID $GROUPNAME && \
    useradd -u $UID -g $GROUPNAME -G sudo -m -s $USE_SHELL $USERNAME && \
    echo $PASSWORD':'$PASSWORD | chpasswd $USERNAME

#install packages
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    python3 \
    python3-pip \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext-dev \
    libgl1-mesa-dev

#install pytorch
RUN python3 -m pip install --upgrade pip && python3 -m pip install \
    torch==1.13.1+cu116 \
    torchvision==0.14.1+cu116 \
    torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu116






