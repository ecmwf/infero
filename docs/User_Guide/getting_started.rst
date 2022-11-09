.. _getting_started:

Getting Started
===============

Install
-------

Build dependencies
``````````````````

Compilation dependencies

 * C/C++ compiler
 * Fortran 90 compiler
 * *CMake* > 3.16 --- For use and installation: `<http://www.cmake.org/>`__
 * *ecbuild* --- ECMWF library of CMake macros: `<https://github.com/ecmwf/ecbuild>`__

Runtime dependencies:

 * *eckit*: `<https://github.com/ecmwf/eckit>`__

Optional runtime dependencies:

  * *TensorFlow Lite* : `<https://github.com/tensorflow/tensorflow.git>`__
  * *TensorFlow C-API* : `<https://www.tensorflow.org/install/lang_c>`__
  * *ONNX-Runtime* : `<https://github.com/Microsoft/onnxruntime>`__
  * *TensorRT* : `<https://developer.nvidia.com/tensorrt>`__

Step-by-Step Installation scripts
`````````````````````````````````
Utility installation scripts are provided in the /dev directory and can be used for default installation of Infero.

 * env.sh : defines installation environment
 * 1_install_deps.sh : installs dependencies
 * 2_install_infero.sh : installs Infero

Installation environment can also be customised by editing the following variables in the *env.sh* script:

+----------------------------+-------------------------------+-------------------------------+
|          Variable          |          Description          |            Default            |
+----------------------------+-------------------------------+-------------------------------+
|INFERO_VERBOSE_COMPILATION  |       Verbose flag            |              0                |
+----------------------------+-------------------------------+-------------------------------+
|ROOT_DIR                    |      Infero root path         |           ${HOME}             |
+----------------------------+-------------------------------+-------------------------------+
|ROOT_SRC_DIR                |      Sources root path        |       ${ROOT_DIR}/local       |
+----------------------------+-------------------------------+-------------------------------+
|ROOT_BUILD_DIR              |      build root path          |       ${ROOT_DIR}/builds      |
+----------------------------+-------------------------------+-------------------------------+
|ROOT_INSTALL_DIR            |      install root path        |       ${ROOT_DIR}/installs    |
+----------------------------+-------------------------------+-------------------------------+
|WITH_MPI                    |      Use MPI functionalities  |             OFF               |
+----------------------------+-------------------------------+-------------------------------+
|WITH_FCKIT                  |      Use FCKIT (Fortran API)  |             ON                |
+----------------------------+-------------------------------+-------------------------------+
|WITH_ONNX_RUNTIME           |      ONNX runtime             |             ON                |
+----------------------------+-------------------------------+-------------------------------+
|WITH_TFC_RUNTIME            |      Tensorflow C-API         |             ON                |
+----------------------------+-------------------------------+-------------------------------+
|TFC_GPU                     |      TEnsorflow C-API (GPU)   |             1                 |
+----------------------------+-------------------------------+-------------------------------+
|WITH_TFLITE_RUNTIME         |      TensorFlow TFlite        |             OFF               |
+----------------------------+-------------------------------+-------------------------------+
|WITH_TRT                    |      TensorRT                 |             OFF               |
+----------------------------+-------------------------------+-------------------------------+
|ENABLE_TESTS                |      Build Infero tests       |             ON                |
+----------------------------+-------------------------------+-------------------------------+
|BUILD_NPROCS                |      Num procs for building   |              8                |
+----------------------------+-------------------------------+-------------------------------+


Configuration
`````````````


Run Tests
---------

Tests can be run from the script:

 * dev/3_run_tests.sh : run Infero tests