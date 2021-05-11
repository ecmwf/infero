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

ecbuild_add_option( FEATURE TF_LITE
                    DESCRIPTION "TensorflowLite engine"
                    REQUIRED_PACKAGES TensorflowLite )

# ecbuild_add_option( FEATURE ONNX
#                     DESCRIPTION "ONNX engine"
#                     REQUIRED_PACKAGES ONNX )

# ecbuild_add_option( FEATURE TENSOR_RT
#                     DESCRIPTION "TensorRT engine"
#                     REQUIRED_PACKAGES TensorRT )

# ONNX
# if(${WITH_ONNX_RUNTIME})
#     set(HAVE_ONNX ON)
#     set(INCL_REL include/onnxruntime/core/session)
#     set(ONNX_INCLUDE_DIR ${ONNXRT_SOURCE_DIR}/${INCL_REL})
#     set(ONNX_LIB_DIR ${ONNXRT_BUILD_DIR}/Linux/Release)
#     list(APPEND INCLUDE_DIRS_ALL ${ONNX_INCLUDE_DIR})
#     list(APPEND LINK_DIRS_ALL ${ONNX_LIB_DIR})
#     add_definitions(-DHAVE_ONNX)
# endif()


# # TFLITE
# if(${WITH_TFLITE_RUNTIME})
#     set(HAVE_TFLITE ON)
#     set(TFLITE_SCHEMA_DIR "${TFLITE_SOURCE_DIR}/schema")
#     set(TFLITE_FLATBUFFERS_DIR "${TFLITE_BUILD_DIR}/flatbuffers/include")
#     list(APPEND INCLUDE_DIRS_ALL ${TFLITE_SOURCE_DIR})
#     list(APPEND INCLUDE_DIRS_ALL ${TFLITE_SCHEMA_DIR})
#     list(APPEND INCLUDE_DIRS_ALL ${TFLITE_FLATBUFFERS_DIR})
#     list(APPEND LINK_DIRS_ALL ${TFLITE_BUILD_DIR})
#     add_definitions(-DHAVE_TFLITE)
# endif()


# # TRT
# if(${WITH_TRT})
#     set(HAVE_TRT ON)
#     set(TRT_TRG "x86_64-linux-gnu")
#     set(TRT_COM ${TRT_SOURCE_DIR}/targets/${TRT_TRG}/samples/common)
#     set(CUDA_INCLUDE_DIR /usr/local/cuda/include)
#     set(CUDA_LIB_DIR /usr/local/cuda-10.2/targets/x86_64-linux/lib)
#     list(APPEND INCLUDE_DIRS_ALL ${TRT_SOURCE_DIR}/include)
#     list(APPEND INCLUDE_DIRS_ALL ${TRT_COM})
#     list(APPEND INCLUDE_DIRS_ALL ${CUDA_INCLUDE_DIR})
#     list(APPEND LINK_DIRS_ALL ${TRT_BUILD_DIR}/lib)
#     list(APPEND LINK_DIRS_ALL ${CUDA_LIB_DIR})
#     add_definitions(-DHAVE_TRT)
# endif()

ecbuild_info("TENSORFLOWLITE_LIBRARIES ${TENSORFLOWLITE_LIBRARIES}")
ecbuild_info("TENSORFLOWLITE_INCLUDE_DIRS ${TENSORFLOWLITE_INCLUDE_DIRS}")

include(cmake/compiler_warnings.cmake) # optionally handle compiler specific warnings

add_subdirectory( src )

ecbuild_add_executable( TARGET infero.x
    SOURCES   src/infero/infero.cpp
    CONDITION HAVE_BUILD_TOOLS
    INCLUDES  ${eckit_INCLUDE_DIRS}
    LIBS      infero )

############################################################################################
# finalize

ecbuild_pkgconfig( NAME infero
                   DESCRIPTION "A lower level API for ML Inference in Operations"
                   LIBRARIES infero )

ecbuild_install_project( NAME ${PROJECT_NAME} )

ecbuild_print_summary()
