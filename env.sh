#!/bin/bash

# ======= Basic config =========
CONDA_DIR=${HOME}/miniconda3
ENV_NAME=ml_exec_env2

ROOT_SRC_DIR=${HOME}/local2
ROOT_BUILD_DIR=${HOME}/builds2

BUILD_NPROCS=8
# ===============================


# ======== other configs ========

# ECBUILD
ECBUILD_TAG=2021.03.0
ECBUILD_SRC_DIR=${ROOT_SRC_DIR}/ecbuild
ECBUILD_BUILD_DIR=${ROOT_BUILD_DIR}/ecbuild

# ECKIT
ECKIT_TAG=2021.03.0
ECKIT_SRC_DIR=${ROOT_SRC_DIR}/eckit
ECKIT_BUILD_DIR=${ROOT_BUILD_DIR}/eckit

# ONNX runtime
WITH_ONNX_RUNTIME=ON
ONNXRT_SOURCE_DIR=${ROOT_SRC_DIR}/onnx_rt
ONNXRT_BUILD_DIR=${ONNXRT_SOURCE_DIR}/build

# TFLITE
WITH_TFLITE_RUNTIME=ON
TFLITE_SOURCE_DIR=${ROOT_SRC_DIR}/tensorflow
TFLITE_BUILD_DIR=${ROOT_BUILD_DIR}/tflite

# TENSORRT
# (NB TensorRT must to be downloaded separately)
WITH_TRT=ON
TRT_SOURCE_DIR=${ROOT_SRC_DIR}/TensorRT-8.0.0.3
TRT_BUILD_DIR=${ROOT_SRC_DIR}/TensorRT-8.0.0.3

# MLEXEC
SRC_DIR=$(pwd)
BUILD_DIR=${ROOT_BUILD_DIR}/mlexec
# ===============================

# ============ ENV ==============
ENV_DIR=${CONDA_DIR}/envs/${ENV_NAME}
if [ ! -d ${ENV_DIR} ]; then

  echo "Creating conda env ${ENV_NAME}"
  conda create  -y -n ${ENV_NAME} python=3.8
  conda activate ${ENV_NAME}

  echo "Installing conda deps.."
  conda install -y cmake
  conda install -y matplotlib
  pip install keras2onnx
  pip install keras
  pip install tensorflow==2.3.1

else

  # activate the environment
  source ${CONDA_DIR}/bin/activate ${ENV_NAME}

fi

