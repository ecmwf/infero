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


Installation scripts
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


Manual Installation
```````````````````
Infero employs an out-of-source build/install based on CMake.
To manually invoke cmake, make sure that ecbuild is installed and the ecbuild executable script is found

.. code-block:: bash

   which ecbuild

Now proceed with installation as follows:

.. code-block:: bash

   # Environment --- Edit as needed
   srcdir=$(pwd)
   builddir=build
   installdir=$HOME/local

.. code-block:: console

   # 1. Create the build directory:
   mkdir $builddir
   cd $builddir

.. code-block:: console

   # 2. Run CMake
   ecbuild --prefix=$installdir -- -DECKIT_PATH=<path/to/eckit/install> $srcdir

.. code-block:: console

   # 3. Compile / Install
   make -j10
   make install


Useful cmake arguments:
 - -DENABLE_TF_LITE=ON
 - -DTENSORFLOWLITE_PATH=</path/to/tensorflow/sources>
 - -DTENSORFLOWLITE_ROOT=</path/to/tflite/root/dir>
 - -DENABLE_TF_C=ON
 - -DTENSORFLOWC_ROOT=</path/to/tf_c/root/dir>
 - -DENABLE_ONNX=ON
 - -DONNX_ROOT=</path/to/onnxruntime/root/dir>
 - -DENABLE_TENSORRT=ON
 - -DTENSORRT_ROOT=</path/to/tensorRT/root/dir>



Run Tests
---------

Tests can be run from the script:

 * dev/3_run_tests.sh : run Infero tests