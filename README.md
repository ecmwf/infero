# Infero

## Description
Infero runs a pre-trained ML model for inference. It can be deployed 
on a HPC system without the need for high-level python libraries 
(e.g. TensorFlow, PyTorch, etc..)

### Disclaimer
This software is still under heavy development and not yet ready for operational use

### Requirements

Build dependencies:

- C/C++ compiler
- Fortran 90 compiler
- CMake > 3.16 --- For use and installation see http://www.cmake.org/
- ecbuild --- ECMWF library of CMake macros (https://github.com/ecmwf/ecbuild)

Runtime dependencies:
  - eckit (https://github.com/ecmwf/eckit)

Optional runtime dependencies:
  - TensorFlow Lite (https://github.com/tensorflow/tensorflow.git)
  - ONNX-Runtime (https://github.com/Microsoft/onnxruntime)
  - TensorRT (https://developer.nvidia.com/tensorrt)

### Installation

Infero employs an out-of-source build/install based on CMake.
Make sure ecbuild is installed and the ecbuild executable script is found ( `which ecbuild` ).
Now proceed with installation as follows:

```bash
# Environment --- Edit as needed
srcdir=$(pwd)
builddir=build
installdir=$HOME/local  

# 1. Create the build directory:
mkdir $builddir
cd $builddir

# 2. Run CMake
ecbuild --prefix=$installdir -- -DECKIT_PATH=<path/to/eckit/install> $srcdir

# 3. Compile / Install
make -j10
make install
```
Useful cmake arguments:
 - -DENABLE_TF_LITE=ON
 - -DTENSORFLOWLITE_PATH=</path/to/tensorflow/sources>
 - -DTENSORFLOWLITE_ROOT=</path/to/tflite/root/dir>
 - -DENABLE_ONNX=ON
 - -DONNX_ROOT=</path/to/onnxruntime/root/dir>
 - -DENABLE_TENSORRT=ON
 - -DTENSORRT_ROOT=</path/to/tensorRT/root/dir>

### Installation (python scripts)
Create and source a conda environment containing all the required dependencies:

> cd scripts
> 
> source scripts_env.sh
