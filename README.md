# Infero

## Description
Infero runs a pre-trained ML model on an input sample. It can be deployed 
on a HPC system without the need for high-level python libraries 
(e.g. TensorFlow, PyTorch, etc..)

### Installation
Requirements:
  - cmake > 3.16
  - C++ compiler

Source the environment:
> source env.sh

For a custom configuration, the environment 
variables in *env.sh* can be edited. 

Root variables below:

 - ${ROOT_SRC_DIR}: Root source path for dependencies
 - ${ROOT_BUILD_DIR}: Root build path
 - ${BUILD_NPROCS}: number of processes for building

Supported backend runtime libraries
 - {WITH_ONNX_RUNTIME}: with ONNX runtime library
 - {WITH_TFLITE_RUNTIME}: with TfLite library
 - {WITH_TRT}: with TensorRT library

Install this package
> ./install.sh

### Installation (python scripts)
Create and source a conda environment with all the required 
dependencies in it:

> source scripts_env.sh

### User Guide
WIP..
