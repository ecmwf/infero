#!/bin/bash

CONDA_DIR=${HOME}/miniconda3
ENV_NAME=infero_env

# create the env
ENV_DIR=${CONDA_DIR}/envs/${ENV_NAME}
if [ ! -d ${ENV_DIR} ]; then

  conda env create -f conda_req.yml
  source ${CONDA_DIR}/bin/activate ${ENV_NAME}

fi

# activate the environment
source ${CONDA_DIR}/bin/activate ${ENV_NAME}
