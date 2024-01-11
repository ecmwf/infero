# (C) Copyright 1996- ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

list(APPEND infero_srcs    
    InferenceModel.h
    InferenceModel.cc
    ModelStatistics.h
    ModelStatistics.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/../Configurable.h
    ${CMAKE_CURRENT_SOURCE_DIR}/../Configurable.cc
)

### support for ONNX
if(HAVE_ONNX)
    list(APPEND infero_srcs
        InferenceModelONNX.h
        InferenceModelONNX.cc
        )
    list(APPEND onnx_libs  ${ONNX_LIBRARIES} )
    add_definitions(-DHAVE_ONNX)
endif()

### support for Tensorflow C API
if(HAVE_TF_C)
    list(APPEND infero_srcs
        InferenceModelTFC.h
        InferenceModelTFC.cc
    )
    list(APPEND tensorflow_c_libs ${TENSORFLOWC_LIBRARIES})
    add_definitions(-DHAVE_TF_C)
endif()


### support for TensorflowLite
if(HAVE_TF_LITE)
    list(APPEND infero_srcs
        InferenceModelTFlite.h
        InferenceModelTFlite.cc
    )
    list(APPEND tensorflow_lite_libs ${TENSORFLOWLITE_LIBRARIES})
    add_definitions(-DHAVE_TFLITE)
endif()


### support for TensorRT
if(HAVE_TENSORRT)
    list(APPEND infero_srcs
        InferenceModelTRT.h
        InferenceModelTRT.cc
    )

    set(tensor_rt_libs ${TENSORRT_LIBRARIES})

    # TODO check this...
    list(APPEND infero_srcs
        ${TENSORRT_INCLUDE_COMMON_DIR}/logger.h
        ${TENSORRT_INCLUDE_COMMON_DIR}/logger.cpp
        )

    add_definitions(-DHAVE_TENSORRT)
endif()



ecbuild_add_library(
    TARGET infero
    
    INSTALL_HEADERS LISTED

    HEADER_DESTINATION
        ${INSTALL_INCLUDE_DIR}/infero

    SOURCES
        ${infero_srcs}

    PUBLIC_INCLUDES
       $<BUILD_INTERFACE:${PROJECT_BINARY_DIR}/src>
       $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/src>
       $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/contrib>

    PRIVATE_INCLUDES
        "${eckit_INCLUDE_DIRS}"
        "${TENSORFLOWC_INCLUDE_DIRS}"
        "${TENSORFLOWLITE_INCLUDE_DIRS}"
        "${ONNX_INCLUDE_DIRS}"
        "${TENSORRT_INCLUDE_DIRS}"

    PRIVATE_LIBS
        ${ZLIB_LIBRARIES}        
        eckit
        eckit_option
        eckit_mpi
        "${tensorflow_c_libs}"
        "${tensorflow_lite_libs}"
        "${onnx_libs}"
        "${tensor_rt_libs}"

    PUBLIC_LIBS
        cnpy
)
