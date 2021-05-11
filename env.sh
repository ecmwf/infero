#!/bin/bash

# ======= Basic config =========
ROOT_SRC_DIR=${HPCPERM}/local
ROOT_BUILD_DIR=${HPCPERM}/builds

WITH_ONNX_RUNTIME=ON
WITH_TFLITE_RUNTIME=ON
WITH_TRT=

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
ONNXRT_SOURCE_DIR=${ROOT_SRC_DIR}/onnx_rt
ONNXRT_BUILD_DIR=${ONNXRT_SOURCE_DIR}/build

# TFLITE
TFLITE_SOURCE_DIR=${ROOT_SRC_DIR}/tensorflow
TFLITE_BUILD_DIR=${ROOT_BUILD_DIR}/tflite

# TENSORRT
# (NB TensorRT must to be downloaded separately)
#WITH_TRT=ON
TRT_SOURCE_DIR=${ROOT_SRC_DIR}/TensorRT-8.0.0.3
TRT_BUILD_DIR=${ROOT_SRC_DIR}/TensorRT-8.0.0.3

# INFERO
SRC_DIR=$(pwd)
BUILD_DIR=${ROOT_BUILD_DIR}/infero
# ===============================

