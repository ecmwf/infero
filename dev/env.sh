#!/bin/bash

# initialise module environment if it is not
if [[ ! $(command -v module > /dev/null 2>&1) ]]; then
  . /usr/local/apps/module/init/bash
fi

module load cmake

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

# ======= Basic config =========
ROOT_DIR=${INFERO_HOME:-$HOME}

ROOT_SRC_DIR=${ROOT_DIR}/local
ROOT_BUILD_DIR=${ROOT_DIR}/builds

WITH_MPI=OFF
WITH_ONNX_RUNTIME=OFF
WITH_TFLITE_RUNTIME=ON
WITH_TRT=OFF

BUILD_NPROCS=8
# ===============================


# ======== other configs ========

# ECBUILD
ECBUILD_BRANCH=develop
ECBUILD_SRC_DIR=${ROOT_SRC_DIR}/ecbuild
ECBUILD_BUILD_DIR=${ECBUILD_SRC_DIR}
ECBUILD_BUILD_EXE=${ECBUILD_BUILD_DIR}/bin/ecbuild

# ECKIT
ECKIT_BRANCH=develop
ECKIT_SRC_DIR=${ROOT_SRC_DIR}/eckit
ECKIT_BUILD_DIR=${ROOT_BUILD_DIR}/eckit

# ONNX runtime
ONNXRT_SOURCE_DIR=${ROOT_SRC_DIR}/onnx_rt
ONNXRT_BUILD_DIR=${ONNXRT_SOURCE_DIR}/build

# TFLITE
TFLITE_SOURCE_DIR=${ROOT_SRC_DIR}/tensorflow
TFLITE_BUILD_DIR=${ROOT_BUILD_DIR}/tflite

# TENSORRT
# (NB TensorRT must to be downloaded separately)
TRT_SOURCE_DIR=${ROOT_SRC_DIR}/TensorRT-8.0.0.3
TRT_BUILD_DIR=${ROOT_SRC_DIR}/TensorRT-8.0.0.3

# INFERO
INFERO_SRC_DIR=$(dirname ${SCRIPT_DIR})
INFERO_BUILD_DIR=${ROOT_BUILD_DIR}/infero
# ===============================


## ====== get miniconda env ======
#CONDA_DIR=${ROOT_DIR}/miniconda
#CONDA_ENV=infero_test_env
#
#if [ ! -d ${CONDA_DIR} ]; then
#  
#  # get miniconda
#  cd ${ROOT_DIR}
#  wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh
#  sh Miniconda3-latest-Linux-x86_64.sh -b -p ${CONDA_DIR}
# 
#  # export path
#  export PATH=${CONDA_DIR}/bin:${PATH}
#
#  # create infero env
#  conda create -y -n ${CONDA_ENV} python=3
#
#  # install cmake
#  source activate ${CONDA_ENV}
#  pip install cmake
#
#fi
#
## activate env
#export PATH=${CONDA_DIR}/bin:${PATH}
#source activate ${CONDA_ENV}
## ===============================



