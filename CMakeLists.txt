# (C) Copyright 2011- ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation nor
# does it submit to any jurisdiction.

########################################################################################################################

cmake_minimum_required( VERSION 3.16 FATAL_ERROR )

find_package( ecbuild 3.6 REQUIRED HINTS ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/../ecbuild)

project( infero LANGUAGES CXX )

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

########################################################################################################################
### dependencies and options

set( PERSISTENT_NAMESPACE "eckit" CACHE INTERNAL "" ) # needed for generating .b files for persistent support

ecbuild_find_package( NAME eckit  VERSION  1.16 REQUIRED )

ecbuild_add_option( FEATURE BUILD_TOOLS
                    DEFAULT ON
                    DESCRIPTION "Build the command line tools" )

ecbuild_add_option( FEATURE FORTRAN
                    DESCRIPTION "Provide Fortran bindings" )
if(HAVE_FORTRAN)
    ecbuild_enable_fortran( REQUIRED MODULE_DIRECTORY ${PROJECT_BINARY_DIR}/module )
endif()


############## MPI
ecbuild_add_option(DEFAULT OFF
                   FEATURE MPI
                   DESCRIPTION "Use system MPI libraries"
                   REQUIRED_PACKAGES "MPI COMPONENTS CXX C" )

############## TFlite
ecbuild_add_option( FEATURE TF_LITE
                    DEFAULT OFF
                    DESCRIPTION "TensorflowLite engine"
                    REQUIRED_PACKAGES TensorflowLite )

############## TF_C
ecbuild_add_option( FEATURE TF_C
                    DEFAULT OFF
                    DESCRIPTION "Tensorflow C engine"
                    REQUIRED_PACKAGES TensorflowC )

############## ONNX
ecbuild_add_option( FEATURE ONNX
                    DEFAULT OFF
                    DESCRIPTION "ONNX engine"
                    REQUIRED_PACKAGES ONNX )

############## TensorRT
ecbuild_add_option( FEATURE TENSORRT
                    DEFAULT OFF
                    DESCRIPTION "TensorRT engine"
                    REQUIRED_PACKAGES TensorRT )

find_package(ZLIB REQUIRED) # for cnpy

ecbuild_info("MPI_FOUND ${MPI_FOUND}")
ecbuild_info("MPI_CXX_LIBRARIES ${MPI_CXX_LIBRARIES}")
ecbuild_info("MPI_CXX_INCLUDE_DIRS ${MPI_CXX_INCLUDE_DIRS}")

ecbuild_info("TENSORFLOWLITE_FOUND ${TENSORFLOWLITE_FOUND}")
ecbuild_info("TENSORFLOWLITE_LIBRARIES ${TENSORFLOWLITE_LIBRARIES}")
ecbuild_info("TENSORFLOWLITE_INCLUDE_DIRS ${TENSORFLOWLITE_INCLUDE_DIRS}")

ecbuild_info("TENSORFLOWC_FOUND ${TENSORFLOWC_FOUND}")
ecbuild_info("TENSORFLOWC_LIBRARIES ${TENSORFLOWC_LIBRARIES}")
ecbuild_info("TENSORFLOWC_INCLUDE_DIRS ${TENSORFLOWC_INCLUDE_DIRS}")

ecbuild_info("ONNX_FOUND ${ONNX_FOUND}")
ecbuild_info("ONNX_LIBRARIES ${ONNX_LIBRARIES}")
ecbuild_info("ONNX_INCLUDE_DIRS ${ONNX_INCLUDE_DIRS}")

ecbuild_info("TENSORRT_FOUND ${TENSORRT_FOUND}")
ecbuild_info("TENSORRT_LIBRARIES ${TENSORRT_LIBRARIES}")
ecbuild_info("TENSORRT_INCLUDE_DIRS ${TENSORRT_INCLUDE_DIRS}")

include(cmake/compiler_warnings.cmake) # optionally handle compiler specific warnings

add_subdirectory( contrib )
add_subdirectory( src )
add_subdirectory( tests )

############################################################################################
# finalize

ecbuild_pkgconfig( NAME infero
                   DESCRIPTION "A lower level API for ML Inference in Operations"
                   LIBRARIES infero )

ecbuild_install_project( NAME ${PROJECT_NAME} )

ecbuild_print_summary()
