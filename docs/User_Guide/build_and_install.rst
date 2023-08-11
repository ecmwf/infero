.. _build_and_install:

Build & Install
===============


Build dependencies
------------------

Compilation dependencies

 * C/C++ compiler
 * Fortran 90 compiler
 * `CMake <http://www.cmake.org/>`__ > 3.16
 * `ecbuild <https://github.com/ecmwf/ecbuild>`__ --- ECMWF library of CMake macros

Runtime dependencies:

 * `eckit <https://github.com/ecmwf/eckit>`__ --- ECMWF C++ toolkit

Optional runtime dependencies:

 * `TensorFlow Lite <https://github.com/tensorflow/tensorflow.git>`__
 * `TensorFlow C-API <https://www.tensorflow.org/install/lang_c>`__
 * `ONNX-Runtime <https://github.com/Microsoft/onnxruntime>`__
 * `TensorRT <https://developer.nvidia.com/tensorrt>`__


Installation scripts
--------------------
Utility installation scripts are provided in the /dev directory and can be used for default installation of Infero.

 * env.sh : defines installation environment
 * 1_install_deps.sh : installs dependencies
 * 2_install_infero.sh : installs Infero

Installation environment can also be customised by editing the following variables in the *env.sh* script:

+----------------------------+-------------------------------+-------------------------------+
|          Variable          |          Description          |            Default            |
+============================+===============================+===============================+
|INFERO_VERBOSE_COMPILATION  |      Verbose flag             |              0                |
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
-------------------
This installation procedure gives more control on the building/installation process.
Infero employs an out-of-source build/install based on CMake. To manually invoke cmake,
make sure that ecbuild is installed and the ecbuild executable script is found.

.. code-block:: console

   which ecbuild

Now proceed with installation as follows:

.. code-block:: bash

   # Environment --- Edit as needed
   srcdir=$(pwd)
   builddir=build
   installdir=$HOME/local

Create the build directory:

.. code-block:: console

   mkdir $builddir
   cd $builddir

Run CMake:

.. code-block:: console

   ecbuild --prefix=$installdir -- -DECKIT_PATH=<path/to/eckit/install> $srcdir

Compile and Install:

.. code-block:: console

   make -j10
   make install

Useful Cmake arguments:

+-----------------------------------+---------------------------------+
|             Variable              |            Description          |
+===================================+=================================+
| -DENABLE_TESTS                    |   Enable Infero tests           |
+-----------------------------------+---------------------------------+
| -DCMAKE_INSTALL_PREFIX            |   Installation root path        |
+-----------------------------------+---------------------------------+
| -DCMAKE_Fortran_MODULE_DIRECTORY  |   Fortran module path           |
+-----------------------------------+---------------------------------+
| -Deckit_ROOT                      |   eckit root path               |
+-----------------------------------+---------------------------------+
| -DENABLE_MPI                      |   Enable MPI                    |
+-----------------------------------+---------------------------------+
| -DENABLE_FCKIT                    |   Enable fckit                  |
+-----------------------------------+---------------------------------+
| -DFCKIT_ROOT                      |   fckit root path               |
+-----------------------------------+---------------------------------+
| -DENABLE_TF_LITE                  |   Enable Tensorflow lite        |
+-----------------------------------+---------------------------------+
| -DTENSORFLOWLITE_PATH             |   TensorFlow lite sources path  |
+-----------------------------------+---------------------------------+
| -DTENSORFLOWLITE_ROOT             |   TensorFlow lite root path     |
+-----------------------------------+---------------------------------+
| -DENABLE_TF_C                     |   Enable TensorFlow C-API       |
+-----------------------------------+---------------------------------+
| -DTENSORFLOWC_ROOT                |   TensorFlow C-API root path    |
+-----------------------------------+---------------------------------+
| -DENABLE_ONNX                     |   Enable onnx-runtime           |
+-----------------------------------+---------------------------------+
| -DONNX_ROOT                       |   ONNX-runtime root path        |
+-----------------------------------+---------------------------------+
| -DENABLE_TENSORRT                 |   Enable tensor-rt              |
+-----------------------------------+---------------------------------+
| -DTENSORRT_ROOT                   |   TensorRT root path            |
+-----------------------------------+---------------------------------+


Run Tests
---------

Tests can be run from the script:

.. code-block:: console

   dev/3_run_tests.sh

Note: The following environment variables can also be set when running tests:

- *INFERO_TEST_NPROCS*: number of processors to use for each regression test (when MPI is enabled)
- *INFERO_TEST_TOL*: overrides the error tolerance on tests at runtime