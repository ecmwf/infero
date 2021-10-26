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
#  TENSORFLOWC_FOUND         - found TensorflowC
#  TENSORFLOWC_INCLUDE_DIRS  - the TensorflowC include directories
#  TENSORFLOWC_LIBRARIES     - the TensorflowC libraries
#
# The following paths will be searched with priority if set in CMake or env
#
#  TENSORFLOWC_PATH          - prefix path of the TensorflowC installation
#  TENSORFLOWC_ROOT          - Set this variable to the root installation

# Search with priority for TENSORFLOWC_PATH if given as CMake or env var
find_path(TENSORFLOWC_INCLUDE_DIR tensorflow/c/c_api.h
          HINTS $ENV{TENSORFLOWC_ROOT} ${TENSORFLOWC_ROOT}
          PATHS ${TENSORFLOWC_PATH} ENV TENSORFLOWC_PATH
          PATH_SUFFIXES include NO_DEFAULT_PATH)

find_path(TENSORFLOWC_INCLUDE_DIR tensorflow/c/c_api.h PATH_SUFFIXES include )

# Search with priority for TENSORFLOWC_PATH if given as CMake or env var
find_library(TENSORFLOWC_LIB tensorflow
            HINTS $ENV{TENSORFLOWC_ROOT} ${TENSORFLOWC_ROOT}
            PATHS ${TENSORFLOWC_PATH} ENV TENSORFLOWC_PATH
            PATH_SUFFIXES lib64 lib NO_DEFAULT_PATH)

find_library( TENSORFLOWC_LIB tensorflow PATH_SUFFIXES lib64 lib )

set( TENSORFLOWC_LIBRARIES    ${TENSORFLOWC_LIB} )
set( TENSORFLOWC_INCLUDE_DIRS ${TENSORFLOWC_INCLUDE_DIR})

include(FindPackageHandleStandardArgs)

# handle the QUIET and REQUIRED arguments and set TENSORFLOWC_FOUND to TRUE
# if all listed variables are TRUE
# Note: capitalisation of the package name must be the same as in the file name
find_package_handle_standard_args(TensorflowC DEFAULT_MSG TENSORFLOWC_LIB TENSORFLOWC_INCLUDE_DIR)

mark_as_advanced(TENSORFLOWC_INCLUDE_DIR TENSORFLOWC_LIB)
