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
#  TENSORFLOWLITE_FOUND         - found TensorFlowLite
#  TENSORFLOWLITE_INCLUDE_DIRS  - the TensorFlowLite include directories
#  TENSORFLOWLITE_LIBRARIES     - the TensorFlowLite libraries
#
# The following paths will be searched with priority if set in CMake or env
#
#  TENSORFLOWLITE_PATH          - prefix path of the TensorFlowLite installation
#  TENSORFLOWLITE_ROOT              - Set this variable to the root installation

# Search with priority for TENSORFLOWLITE_PATH if given as CMake or env var

find_path(TENSORFLOWLITE_INCLUDE_DIR tensorflow/lite/model.h
          HINTS $ENV{TENSORFLOWLITE_ROOT} ${TENSORFLOWLITE_ROOT}
          PATHS ${TENSORFLOWLITE_PATH} ENV TENSORFLOWLITE_PATH
          PATH_SUFFIXES include NO_DEFAULT_PATH)

find_path(TENSORFLOWLITE_INCLUDE_DIR tensorflow/lite/model.h PATH_SUFFIXES include )

find_path(TENSORFLOWLITE_FLATBUFFERS_DIR flatbuffers/flatbuffers.h
          HINTS $ENV{TENSORFLOWLITE_ROOT} ${TENSORFLOWLITE_ROOT}
          PATHS ${TENSORFLOWLITE_PATH} ENV TENSORFLOWLITE_PATH
          PATH_SUFFIXES include NO_DEFAULT_PATH)

find_path(TENSORFLOWLITE_FLATBUFFERS_DIR flatbuffers/flatbuffers.h PATH_SUFFIXES include )

# Search with priority for TENSORFLOWLITE_PATH if given as CMake or env var
find_library(TENSORFLOW_LITE_LIB tensorflow-lite
            HINTS $ENV{TENSORFLOWLITE_ROOT} ${TENSORFLOWLITE_ROOT}
            PATHS ${TENSORFLOWLITE_PATH} ENV TENSORFLOWLITE_PATH
            PATH_SUFFIXES lib64 lib NO_DEFAULT_PATH)

find_library( TENSORFLOW_LITE_LIB tensorflow-lite PATH_SUFFIXES lib64 lib )

set( TENSORFLOWLITE_LIBRARIES    ${TENSORFLOW_LITE_LIB} )
set( TENSORFLOWLITE_INCLUDE_DIRS ${TENSORFLOWLITE_INCLUDE_DIR} ${TENSORFLOWLITE_FLATBUFFERS_DIR})

include(FindPackageHandleStandardArgs)

# handle the QUIET and REQUIRED arguments and set TENSORFLOWLITE_FOUND to TRUE
# if all listed variables are TRUE
# Note: capitalisation of the package name must be the same as in the file name
find_package_handle_standard_args(TensorflowLite DEFAULT_MSG TENSORFLOW_LITE_LIB TENSORFLOWLITE_INCLUDE_DIR)

mark_as_advanced(TENSORFLOWLITE_INCLUDE_DIR TENSORFLOW_LITE_LIB)
