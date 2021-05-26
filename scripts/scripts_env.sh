#!/bin/bash

CONDA_PREFIX=${CONDA_PREFIX:-${HOME}/miniconda3}
ENV_NAME=infero_env2

# create the env
ENV_DIR=${CONDA_PREFIX}/envs/${ENV_NAME}
if [ ! -d ${ENV_DIR} ]; then
  conda env create -n ${ENV_NAME} -f conda_req.yml
  source ${CONDA_PREFIX}/bin/activate ${ENV_NAME}
fi

# activate the environment
source ${CONDA_PREFIX}/bin/activate ${ENV_NAME}
