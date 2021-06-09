#!/bin/bash


# load the environment
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
source ${SCRIPT_DIR}/env.sh


# ============= ECBUILD =============
if [ ! -d ${ECBUILD_SRC_DIR} ]; then

  # clone ECBUILD
  echo "Creating dir ${ECBUILD_SRC_DIR}.."
  mkdir -p ${ECBUILD_SRC_DIR}

  echo "Cloning ecbuild into ${ECBUILD_SRC_DIR}.."
  git clone https://github.com/ecmwf/ecbuild.git ${ECBUILD_SRC_DIR}
  cd ${ECBUILD_SRC_DIR}
  git checkout tags/${ECBUILD_TAG} -b ${ECBUILD_TAG}-branch

else
  echo "Directory ${ECBUILD_SRC_DIR} already exist!"
fi


# ============= ECKIT =============
if [ ! -d ${ECKIT_SRC_DIR} ]; then

  # clone ECKIT
  echo "Creating dir ${ECKIT_SRC_DIR}.."
  mkdir -p ${ECKIT_SRC_DIR}

  echo "Cloning ecbuild into ${ECKIT_SRC_DIR}.."
  git clone https://github.com/ecmwf/eckit.git ${ECKIT_SRC_DIR}
  cd ${ECKIT_SRC_DIR}
  git checkout tags/${ECKIT_TAG} -b ${ECKIT_TAG}-branch
else
    echo "Directory ${ECKIT_SRC_DIR} already exist!"
fi

if [ ! -d ${ECKIT_BUILD_DIR} ]; then

  echo "Building eckit in ${ECKIT_BUILD_DIR}.."
  if [ ! -e ${ECKIT_BUILD_DIR} ]; then
    echo "Creating dir ${ECKIT_BUILD_DIR}.."
    mkdir -p ${ECKIT_BUILD_DIR}
  fi

  cd ${ECKIT_BUILD_DIR}
  export PATH=${ECBUILD_SRC_DIR}/bin:$PATH
  cmake ${ECKIT_SRC_DIR}
  make -j${BUILD_NPROCS}

else
  echo "Directory ${ECKIT_BUILD_DIR} already exist!"
fi
# ====================================

# ====== Clone and build ONNXRT ======
if [ ! -d ${ONNXRT_SOURCE_DIR} ]; then

  # clone ONNXRT
  echo "Creating dir ${ONNXRT_SOURCE_DIR}.."
  mkdir -p ${ONNXRT_SOURCE_DIR}

  echo "cloning ONNXRT in ${ONNXRT_SOURCE_DIR}.."
  git clone --recursive https://github.com/Microsoft/onnxruntime ${ONNXRT_SOURCE_DIR}

  # build ONNXRT
  echo "cd in ${ONNXRT_SOURCE_DIR}.."
  cd ${ONNXRT_SOURCE_DIR}

  echo "Building ONNX.."
  ./build.sh --config Release --build_shared_lib --parallel ${BUILD_NPROCS}

else
    echo "Directory ${ONNXRT_SOURCE_DIR} already exist!"
fi
# =============================================

# =========== Clone Tensorflow-LITE ===========
if [ ! -d ${TFLITE_SOURCE_DIR} ]; then

  # clone ONNXRT
  echo "Creating dir ${TFLITE_SOURCE_DIR}.."
  mkdir -p ${TFLITE_SOURCE_DIR}

  echo "cloning TFLITE in ${TFLITE_SOURCE_DIR}.."
  git clone https://github.com/tensorflow/tensorflow.git ${TFLITE_SOURCE_DIR}

else
    echo "Directory ${TFLITE_SOURCE_DIR} already exist!"
fi
# =============================================

# =========== Build Tensorflow-LITE ===========
if [ ! -d ${TFLITE_BUILD_DIR} ]; then

  echo "Building TFLITE in ${TFLITE_BUILD_DIR}.."
  if [ ! -e ${TFLITE_BUILD_DIR} ]; then
    echo "Creating dir ${TFLITE_BUILD_DIR}.."
    mkdir -p ${TFLITE_BUILD_DIR}
  fi

  cd ${TFLITE_BUILD_DIR}
  cmake -DBUILD_SHARED_LIBS=ON ${TFLITE_SOURCE_DIR}/tensorflow/lite
  make -j${BUILD_NPROCS}

else
    echo "Directory ${TFLITE_BUILD_DIR} already exist!"
fi
# =============================================

# ================= TensorRT ==================
if [[ ! -d ${TRT_SOURCE_DIR} && ${WITH_TRT} == ON ]] ; then
  echo "WITH_TRT=ON but TRT source dir ${TRT_SOURCE_DIR} is NOT FOUND!"
  exit 1
fi

if [[ ! -d ${TRT_BUILD_DIR} && ${WITH_TRT} == ON ]] ; then
  echo "WITH_TRT=ON but TRT build dir ${TRT_BUILD_DIR} is NOT FOUND!"
  exit 1
fi
# =============================================

echo "all done."

