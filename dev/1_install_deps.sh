#!/bin/bash


# load the environment
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
source ${SCRIPT_DIR}/env.sh

echo "ROOT_DIR = ${ROOT_DIR}"
echo "ROOT_SRC_DIR = ${ROOT_SRC_DIR}"
echo "ROOT_BUILD_DIR = ${ROOT_BUILD_DIR}"

# ============= ECBUILD =============
if [ ! -d ${ECBUILD_SRC_DIR} ]; then

  # clone ECBUILD
  echo "Creating dir ${ECBUILD_SRC_DIR}.."
  mkdir -p ${ECBUILD_SRC_DIR}

  echo "Cloning ecbuild into ${ECBUILD_SRC_DIR}.."
  git clone https://github.com/ecmwf/ecbuild.git ${ECBUILD_SRC_DIR}
  cd ${ECBUILD_SRC_DIR}
  git checkout ${ECBUILD_BRANCH}

else
  echo "Directory ${ECBUILD_SRC_DIR} already exist!"
fi

if [ ! -d ${ROOT_INSTALL_DIR} ]; then
   mkdir -p ${ROOT_INSTALL_DIR} 
fi

# ============= ECKIT =============
if [ ! -d ${ECKIT_SRC_DIR} ]; then

  # clone ECKIT
  echo "Creating dir ${ECKIT_SRC_DIR}.."
  mkdir -p ${ECKIT_SRC_DIR}

  echo "Cloning ecbuild into ${ECKIT_SRC_DIR}.."
  git clone https://github.com/ecmwf/eckit.git ${ECKIT_SRC_DIR}
  cd ${ECKIT_SRC_DIR}
  git checkout ${ECKIT_BRANCH}
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
  cmake -DCMAKE_INSTALL_PREFIX=${ROOT_INSTALL_DIR} ${ECKIT_SRC_DIR} && \
  make -j${BUILD_NPROCS} && \
  make install

else
  echo "Directory ${ECKIT_BUILD_DIR} already exist!"
fi
# ====================================


# ============= FCKIT =============
if [ ! -d ${FCKIT_SRC_DIR} ]; then

  # clone FCKIT
  echo "Creating dir ${FCKIT_SRC_DIR}.."
  mkdir -p ${FCKIT_SRC_DIR}

  echo "Cloning ecbuild into ${FCKIT_SRC_DIR}.."
  git clone https://github.com/ecmwf/fckit.git ${FCKIT_SRC_DIR}
  cd ${FCKIT_SRC_DIR}
  git checkout ${FCKIT_BRANCH}
else
    echo "Directory ${FCKIT_SRC_DIR} already exist!"
fi

if [ ! -d ${FCKIT_BUILD_DIR} ]; then

  echo "Building FCKIT in ${FCKIT_BUILD_DIR}.."
  if [ ! -e ${FCKIT_BUILD_DIR} ]; then
    echo "Creating dir ${FCKIT_BUILD_DIR}.."
    mkdir -p ${FCKIT_BUILD_DIR}
  fi

  cd ${FCKIT_BUILD_DIR}
  export PATH=${ECBUILD_SRC_DIR}/bin:$PATH
  
  cmake -DCMAKE_INSTALL_PREFIX=${ROOT_INSTALL_DIR} ${FCKIT_SRC_DIR}  && \
  make -j${BUILD_NPROCS}  && \
  make install

else
  echo "Directory ${FCKIT_BUILD_DIR} already exist!"
fi
# ====================================


# ====== Get the pre-built ONNXRT ======
if [ ! -d ${ONNXRT_SOURCE_DIR} ] && [ ${WITH_ONNX_RUNTIME} == ON ]; then

  ONNX_URL=https://github.com/microsoft/onnxruntime/releases/download
  ONNX_TARFILE=onnxruntime-linux-x64-${ONNX_VERSION}.tgz

  echo "Creating dir ${ONNXRT_SOURCE_DIR}.."
  mkdir -p ${ONNXRT_SOURCE_DIR}

  echo "Downloading ONNXRT in ${ONNXRT_SOURCE_DIR}"
  wget ${ONNX_URL}/v${ONNX_VERSION}/${ONNX_TARFILE} -P ${ONNXRT_SOURCE_DIR}
  #cd ${ROOT_INSTALL_DIR}
  tar xzvf ${ONNXRT_SOURCE_DIR}/${ONNX_TARFILE} --strip-components=1 -C ${ROOT_INSTALL_DIR}
else
    echo "Skipping ${ONNXRT_SOURCE_DIR}.."
fi
# =============================================


# =============== TF C-API ====================
if [ ! -d ${TF_C_SOURCE_DIR} ] && [ ${WITH_TF_C_RUNTIME} == ON ]; then

  TFC_URL=https://storage.googleapis.com/tensorflow/libtensorflow
  TFC_VERSION=2.6.0
  TFC_TARFILE=libtensorflow-cpu-linux-x86_64-${TFC_VERSION}.tar.gz

  # clone ONNXRT
  echo "Creating dir ${TF_C_SOURCE_DIR}.."
  mkdir -p ${TF_C_SOURCE_DIR}

  echo "Downloading TF C-API in ${TF_C_SOURCE_DIR}"
  wget ${TFC_URL}/${TFC_TARFILE} -P ${TF_C_SOURCE_DIR}
  #cd ${ROOT_INSTALL_DIR}
  tar xzvf ${TF_C_SOURCE_DIR}/${TFC_TARFILE} -C ${ROOT_INSTALL_DIR}

else
    echo "Skipping ${TF_C_SOURCE_DIR}.."
fi
# =============================================



# =========== Clone Tensorflow-LITE ===========
if [ ! -d ${TFLITE_SOURCE_DIR} ] && [ ${WITH_TFLITE_RUNTIME} == ON ]; then

  # clone ONNXRT
  echo "Creating dir ${TFLITE_SOURCE_DIR}.."
  mkdir -p ${TFLITE_SOURCE_DIR}

  echo "cloning TFLITE in ${TFLITE_SOURCE_DIR}.."
  git clone https://github.com/tensorflow/tensorflow.git ${TFLITE_SOURCE_DIR}

else
    echo "Skipping ${TFLITE_SOURCE_DIR}.."
fi
# =============================================

# =========== Build Tensorflow-LITE ===========
if [ ! -d ${TFLITE_BUILD_DIR} ] && [ ${WITH_TFLITE_RUNTIME} == ON ]; then

  echo "Building TFLITE in ${TFLITE_BUILD_DIR}.."
  if [ ! -e ${TFLITE_BUILD_DIR} ]; then
    echo "Creating dir ${TFLITE_BUILD_DIR}.."
    mkdir -p ${TFLITE_BUILD_DIR}
  fi

  cd ${TFLITE_BUILD_DIR}
  cmake -DCMAKE_INSTALL_PREFIX=${ROOT_INSTALL_DIR} \
        -DBUILD_SHARED_LIBS=ON ${TFLITE_SOURCE_DIR}/tensorflow/lite && \
  make -j${BUILD_NPROCS}

else
    echo "Skipping ${TFLITE_BUILD_DIR}.."
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

