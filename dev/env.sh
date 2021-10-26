#!/bin/bash

# # initialise module environment if it is not
# if [[ ! $(command -v module > /dev/null 2>&1) ]]; then
#   . /usr/local/apps/module/init/bash
# fi
# module load cmake

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

# ======= Basic config =========
INFERO_VERBOSE_COMPILATION=${INFERO_VERBOSE_COMPILATION:-"0"}
ROOT_DIR=${INFERO_HOME:-$HOME}

ROOT_SRC_DIR=${ROOT_DIR}/local
ROOT_BUILD_DIR=${ROOT_DIR}/builds

WITH_MPI=OFF
WITH_ONNX_RUNTIME=ON
WITH_TF_C_RUNTIME=ON
WITH_TFLITE_RUNTIME=OFF
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

# TF_C
TF_C_SOURCE_DIR=${ROOT_SRC_DIR}/tensorflow_c
TF_C_BUILD_DIR=${ROOT_SRC_DIR}/tensorflow_c

# TENSORRT
# (NB TensorRT must to be downloaded separately)
TRT_SOURCE_DIR=${ROOT_SRC_DIR}/TensorRT-8.0.0.3
TRT_BUILD_DIR=${ROOT_SRC_DIR}/TensorRT-8.0.0.3

# INFERO
INFERO_SRC_DIR=$(dirname ${SCRIPT_DIR})
INFERO_BUILD_DIR=${ROOT_BUILD_DIR}/infero
# ===============================


