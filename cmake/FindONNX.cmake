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
#  ONNX_FOUND         - found ONNX
#  ONNX_INCLUDE_DIRS  - the ONNX include directories
#  ONNX_LIBRARIES     - the ONNX libraries
#
# The following paths will be searched with priority if set in CMake or env
#
#  ONNX_PATH          - prefix path of the ONNX installation
#  ONNX_ROOT              - Set this variable to the root installation

# Search with priority for ONNX_PATH if given as CMake or env var
find_path(ONNX_INCLUDE_DIR onnxruntime_cxx_api.h
          HINTS $ENV{ONNX_ROOT}/include/onnxruntime/core/session/
                ${ONNX_ROOT}/include/onnxruntime/core/session/

          PATHS ${ONNX_PATH} ENV ONNX_PATH
          PATH_SUFFIXES include NO_DEFAULT_PATH)

find_path(ONNX_INCLUDE_DIR onnxruntime_cxx_api.h PATH_SUFFIXES include )

# Search with priority for ONNX_PATH if given as CMake or env var
find_library(ONNX_LIB onnxruntime
            HINTS $ENV{ONNX_ROOT}/build/Linux/Release ${ONNX_ROOT}/build/Linux/Release
            PATHS ${ONNX_PATH} ENV ONNX_PATH
            PATH_SUFFIXES lib64 lib NO_DEFAULT_PATH)

find_library( ONNX_LIB onnxruntime PATH_SUFFIXES lib64 lib )

set( ONNX_LIBRARIES    ${ONNX_LIB} )
set( ONNX_INCLUDE_DIRS ${ONNX_INCLUDE_DIR})

include(FindPackageHandleStandardArgs)

# handle the QUIET and REQUIRED arguments and set ONNX_FOUND to TRUE
# if all listed variables are TRUE
# Note: capitalisation of the package name must be the same as in the file name
find_package_handle_standard_args(ONNX DEFAULT_MSG ONNX_LIB ONNX_INCLUDE_DIR)

mark_as_advanced(ONNX_INCLUDE_DIR ONNX_LIB)
