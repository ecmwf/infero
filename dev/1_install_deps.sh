#!/bin/bash

set -e


# load the environment
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
source ${SCRIPT_DIR}/env.sh

echo "ROOT_DIR = ${ROOT_DIR}"
echo "ROOT_SRC_DIR = ${ROOT_SRC_DIR}"
echo "ROOT_BUILD_DIR = ${ROOT_BUILD_DIR}"

if [ ! -d ${ROOT_INSTALL_DIR} ]; then
   mkdir -p ${ROOT_INSTALL_DIR} 
fi


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


echo "Building eckit in ${ECKIT_BUILD_DIR}.."
if [ ! -d ${ECKIT_BUILD_DIR} ]; then
  echo "Creating dir ${ECKIT_BUILD_DIR}.."
  mkdir -p ${ECKIT_BUILD_DIR}

  cd ${ECKIT_BUILD_DIR}
  
  cmake \
  -DCMAKE_INSTALL_PREFIX=${ROOT_INSTALL_DIR} \
  -Decbuild_ROOT=${ECBUILD_SRC_DIR} \
  -DENABLE_MPI=ON \
  ${ECKIT_SRC_DIR}
  
  make -j${BUILD_NPROCS} && make install

else
  echo "Directory ${ECKIT_BUILD_DIR} already exist, building.."
  cd ${ECKIT_BUILD_DIR}
  make -j${BUILD_NPROCS} && make install
fi
# ====================================


# ============= FCKIT =============
if [ ${WITH_FORTRAN} == ON ]; then

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

    echo "Building FCKIT in ${FCKIT_BUILD_DIR}.."
    if [ ! -d ${FCKIT_BUILD_DIR} ]; then

        echo "Creating dir ${FCKIT_BUILD_DIR}.."
        mkdir -p ${FCKIT_BUILD_DIR}

        cd ${FCKIT_BUILD_DIR}
        cmake \
        -DCMAKE_INSTALL_PREFIX=${ROOT_INSTALL_DIR} \
        -Decbuild_ROOT=${ECBUILD_SRC_DIR} \
        ${FCKIT_SRC_DIR}

        make -j${BUILD_NPROCS} && make install

    else
        echo "Directory ${FCKIT_BUILD_DIR} already exist, building.."
        cd ${FCKIT_BUILD_DIR}
        make -j${BUILD_NPROCS} && make install
    fi
fi    
# ====================================


# ====== Get the pre-built ONNXRT ======
if [ ${WITH_ONNX_RUNTIME} == ON ]; then    
    if [ ! -d ${ONNX_SOURCE_DIR} ]; then

        echo "Creating dir ${ONNX_SOURCE_DIR}.."
        mkdir -p ${ONNX_SOURCE_DIR}

        echo "Downloading ONNXRT in ${ONNX_SOURCE_DIR}"
        wget ${ONNX_URL}/v${ONNX_VERSION}/${ONNX_TARFILE} -P ${ONNX_SOURCE_DIR}
        tar xzvf ${ONNX_SOURCE_DIR}/${ONNX_TARFILE} --strip-components=1 -C ${ROOT_INSTALL_DIR}
    else
        echo "ONNXRT sources ${ONNX_SOURCE_DIR} exists. Trying to install.."
        tar xzvf ${ONNX_SOURCE_DIR}/${ONNX_TARFILE} --strip-components=1 -C ${ROOT_INSTALL_DIR}
    fi    
fi
# =============================================


# =============== TF C-API ====================
if [ ${WITH_TFC_RUNTIME} == ON ]; then
    if [ ! -d ${TFC_SOURCE_DIR} ]; then
        echo "Creating dir ${TFC_SOURCE_DIR}.."
        mkdir -p ${TFC_SOURCE_DIR}

        echo "Downloading TF C-API in ${TFC_SOURCE_DIR}"
        wget ${TFC_URL}/${TFC_TARFILE} -P ${TFC_SOURCE_DIR}
        tar xzvf ${TFC_SOURCE_DIR}/${TFC_TARFILE} -C ${ROOT_INSTALL_DIR}

    else
        echo "TF_C sources ${TFC_SOURCE_DIR} exists. Trying to install.."
        tar xzvf ${TFC_SOURCE_DIR}/${TFC_TARFILE} -C ${ROOT_INSTALL_DIR}
    fi
fi    
# =============================================


# ================ TFLITE =====================
if [ ${WITH_TFLITE_RUNTIME} == ON ]; then
    if [ ! -d ${TFLITE_SOURCE_DIR} ]; then
    
        echo "Creating dir ${TFLITE_SOURCE_DIR}.."
        mkdir -p ${TFLITE_SOURCE_DIR}

        echo "cloning TFLITE in ${TFLITE_SOURCE_DIR}.."
        git clone https://github.com/tensorflow/tensorflow.git ${TFLITE_SOURCE_DIR}

    else
        echo "Skipping ${TFLITE_SOURCE_DIR}.."
    fi

    echo "Building TFLITE in ${TFLITE_BUILD_DIR}.."
    if [ ! -d ${TFLITE_BUILD_DIR} ]; then
        
        echo "Creating dir ${TFLITE_BUILD_DIR}.."
        mkdir -p ${TFLITE_BUILD_DIR}

        cd ${TFLITE_BUILD_DIR}
        cmake \
        -DCMAKE_INSTALL_PREFIX=${ROOT_INSTALL_DIR} \
        -DBUILD_SHARED_LIBS=ON ${TFLITE_SOURCE_DIR}/tensorflow/lite
                
        make -j${BUILD_NPROCS}
        
        # NOTE: DO NOT INSTALL..

    else
        echo "Directory ${TFLITE_BUILD_DIR} already exist, building.."
        cd ${TFLITE_BUILD_DIR}
        make -j${BUILD_NPROCS}
    fi
fi    


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

