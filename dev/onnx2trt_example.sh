#!/bin/bash

# example script to generate a TensorRT engine from onnx model

TRT_PATH=${1:-${HOME}/TensorRT-8.0.0.3}
ONNX_MODEL_PATH=$2
TRT_MODEL_PATH=${3:-model.trt}

export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:${TRT_PATH}/lib

echo "Converting onnx model ${ONNX_PATH} ..."

# ========== tcyclone model
#${TRT_PATH}/bin/trtexec --explicitBatch --onnx=${ONNX_MODEL_PATH} \
#  --minShapes=input:1x200x200x17 \
#  --optShapes=input:1x200x200x17 \
#  --maxShapes=input:1x200x200x17 \
#  --saveEngine=${TRT_MODEL_PATH}

# ========== orographic drag model
${TRT_PATH}/bin/trtexec --onnx=${ONNX_MODEL_PATH} \
  --minShapes=dense_21_input:8x191 \
  --optShapes=dense_21_input:8x191 \
  --maxShapes=dense_21_input:8x191 \
  --saveEngine=${TRT_MODEL_PATH}

echo "all done."
