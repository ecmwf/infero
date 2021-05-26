# (C) Copyright 2011- ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation nor
# does it submit to any jurisdiction.

# - Try to find TensorFlowLite
# Once done this will define
#
#  TENSORRT_FOUND         - found TRT
#  TENSORRT_INCLUDE_DIRS  - the TRT include directories
#  TENSORRT_LIBRARIES     - the TRT libraries
#
# The following paths will be searched with priority if set in CMake or env
#
#  TENSORRT_PATH          - prefix path of the TRT installation
#  TENSORRT_ROOT              - Set this variable to the root installation

# Search with priority for TENSORRT_PATH if given as CMake or env var
find_path(TENSORRT_INCLUDE_DIR NvInfer.h
          HINTS $ENV{TENSORRT_ROOT}/include/
                ${TENSORRT_ROOT}/include/
          PATHS ${TENSORRT_PATH} ENV TENSORRT_PATH
          PATH_SUFFIXES include NO_DEFAULT_PATH)

find_path(TENSORRT_INCLUDE_COMMON_DIR common.h
          HINTS $ENV{TENSORRT_ROOT}/targets/x86_64-linux-gnu/samples/common
                ${TENSORRT_ROOT}/targets/x86_64-linux-gnu/samples/common
          PATHS ${TENSORRT_PATH} ENV TENSORRT_PATH
          PATH_SUFFIXES include NO_DEFAULT_PATH)


# Search with priority for TENSORRT_PATH if given as CMake or env var
find_library(TENSORRT_LIB_infer
            NAMES nvinfer
            HINTS $ENV{TENSORRT_ROOT}/lib ${TENSORRT_ROOT}/lib
            PATHS ${TENSORRT_PATH} ENV TENSORRT_PATH
            PATH_SUFFIXES lib64 lib NO_DEFAULT_PATH)

find_library(TENSORRT_LIB_nvonnxparser
            NAMES nvonnxparser
            HINTS $ENV{TENSORRT_ROOT}/lib ${TENSORRT_ROOT}/lib
            PATHS ${TENSORRT_PATH} ENV TENSORRT_PATH
            PATH_SUFFIXES lib64 lib NO_DEFAULT_PATH)

find_library(TENSORRT_LIB_nvparsers
            NAMES nvparsers
            HINTS $ENV{TENSORRT_ROOT}/lib ${TENSORRT_ROOT}/lib
            PATHS ${TENSORRT_PATH} ENV TENSORRT_PATH
            PATH_SUFFIXES lib64 lib NO_DEFAULT_PATH)

find_library(TENSORRT_LIB_nvinfer_plugin
            NAMES nvinfer_plugin
            HINTS $ENV{TENSORRT_ROOT}/lib ${TENSORRT_ROOT}/lib
            PATHS ${TENSORRT_PATH} ENV TENSORRT_PATH
            PATH_SUFFIXES lib64 lib NO_DEFAULT_PATH)


find_package(CUDA)

set(TENSORRT_LIB
    ${TENSORRT_LIB_infer}
    ${TENSORRT_LIB_nvonnxparser}
    ${TENSORRT_LIB_nvparsers}
    ${TENSORRT_LIB_nvinfer_plugin}
)

set(TENSORRT_LIBRARIES
    ${CUDA_LIBRARIES}
    ${TENSORRT_LIB}
    )

set(TENSORRT_INCLUDE_DIRS
    ${CUDA_INCLUDE_DIRS}
    ${TENSORRT_INCLUDE_DIR}
    ${TENSORRT_INCLUDE_COMMON_DIR}
)

include(FindPackageHandleStandardArgs)

# handle the QUIET and REQUIRED arguments and set TENSORRT_FOUND to TRUE
# if all listed variables are TRUE
# Note: capitalisation of the package name must be the same as in the file name
find_package_handle_standard_args(TensorRT DEFAULT_MSG TENSORRT_LIB TENSORRT_INCLUDE_DIR)

mark_as_advanced(
    TENSORRT_INCLUDE_DIR
    TENSORRT_LIB
    TENSORRT_LIB_infer
    TENSORRT_LIB_nvonnxparser
    TENSORRT_LIB_nvparsers
    TENSORRT_LIB_nvinfer_plugin
    TENSORRT_LIB_nvrtc
    )
