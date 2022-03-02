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
INFERO_CMAKE_CMD="${ECBUILD_BUILD_EXE} \
\
-Deckit_ROOT=${ECKIT_BUILD_DIR} \
-DENABLE_MPI=${WITH_MPI} \
\
-DENABLE_TF_LITE=${WITH_TFLITE_RUNTIME} \
-DTENSORFLOWLITE_PATH=${TFLITE_SOURCE_DIR} \
-DTENSORFLOWLITE_ROOT=${TFLITE_BUILD_DIR} \
\
-DENABLE_TF_C=${WITH_TF_C_RUNTIME} \
-DTENSORFLOWC_ROOT=${TF_C_BUILD_DIR} \
\
-DENABLE_ONNX=${WITH_ONNX_RUNTIME} \
-DONNX_ROOT=${ONNXRT_BUILD_DIR} \
\
-DENABLE_TENSORRT=${WITH_TRT} \
-DTENSORRT_ROOT=${TRT_BUILD_DIR} \
\
${INFERO_SRC_DIR}"

if [ ${INFERO_VERBOSE_COMPILATION} == "0" ]; then
  INFERO_MAKE_CMD="make" 
else
  INFERO_MAKE_CMD="make VERBOSE=1" 
fi

#execute cmake and make
${INFERO_CMAKE_CMD} && ${INFERO_MAKE_CMD}

echo "all done."
