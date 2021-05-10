# Cyclone

## Description
Infero runs a pre-trained ML model on an input sample.
It is deployed on a HPC node without the need for high-level 
python libraries (e.g. Tensorflow, pytorch, etc..)

### Installation
Create a conda environment with all the required 
dependencies in it:

> source env.sh

For a custom configuration, the following environment 
variables (in env.sh) can be edited

 - CONDA_DIR: conda root directory
 - ENV_NAME: name of conda environment to create
 - ROOT_SRC_DIR: Root source path for dependencies
 - ROOT_BUILD_DIR: Root build path for dependencies
 - BUILD_NPROCS: number of processes for building

Install this package
> ./install.sh
 
### User Guide
WIP..
