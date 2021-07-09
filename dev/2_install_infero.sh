#!/bin/bash


# load the environment
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
source ${SCRIPT_DIR}/env.sh

# make build dir
if [ ! -e ${INFERO_BUILD_DIR} ]; then
  echo "Creating dir ${INFERO_BUILD_DIR}.."
  mkdir -p ${INFERO_BUILD_DIR}
fi

echo "cd in ${INFERO_BUILD_DIR}.."
cd ${INFERO_BUILD_DIR}


echo "Building Infero.."
${ECBUILD_BUILD_EXE} \
-Deckit_ROOT=${HOME}/builds2/eckit \
\
-DENABLE_MPI=${WITH_MPI} \
\
-DENABLE_TF_LITE=${WITH_TFLITE_RUNTIME} \
-DTENSORFLOWLITE_PATH=${TFLITE_SOURCE_DIR} \
-DTENSORFLOWLITE_ROOT=${TFLITE_BUILD_DIR} \
\
-DENABLE_ONNX=${WITH_ONNX_RUNTIME} \
-DONNX_ROOT=${ONNXRT_SOURCE_DIR} \
\
-DENABLE_TENSORRT=${WITH_TRT} \
${INFERO_SRC_DIR} \
\
&& \
\
make

echo "all done."

