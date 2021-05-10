
PROJECT(Infero CXX)
cmake_minimum_required(VERSION 3.5)


SET(CMAKE_BUILD_TYPE Release)
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread -std=c++11")


##################### env #####################
set(ECKIT_SRC_DIR $ENV{ECKIT_SRC_DIR})
set(ECKIT_BUILD_DIR $ENV{ECKIT_BUILD_DIR})

set(WITH_ONNX_RUNTIME $ENV{WITH_ONNX_RUNTIME})
set(ONNXRT_SOURCE_DIR $ENV{ONNXRT_SOURCE_DIR})
set(ONNXRT_BUILD_DIR $ENV{ONNXRT_BUILD_DIR})

set(WITH_TFLITE_RUNTIME $ENV{WITH_TFLITE_RUNTIME})
set(TFLITE_SOURCE_DIR $ENV{TFLITE_SOURCE_DIR})
set(TFLITE_BUILD_DIR $ENV{TFLITE_BUILD_DIR})

set(WITH_TRT $ENV{WITH_TRT})
set(TRT_SOURCE_DIR $ENV{TRT_SOURCE_DIR})
set(TRT_BUILD_DIR $ENV{TRT_BUILD_DIR})
###############################################

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)


# include directories
set(INCLUDE_DIRS_ALL
    ${CMAKE_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/src
    )

# link dirs
set(LINK_DIRS_ALL "")

# ECKIT
set(ECKIT_INCLUDE_DIR
    ${ECKIT_SRC_DIR}/src
    ${ECKIT_BUILD_DIR}/src
    )

set(ECKIT_LIB_DIR
    ${ECKIT_BUILD_DIR}/lib
    )

list(APPEND INCLUDE_DIRS_ALL ${ECKIT_INCLUDE_DIR})
list(APPEND LINK_DIRS_ALL ${ECKIT_LIB_DIR})


# ONNX
if(${WITH_ONNX_RUNTIME})

    set(HAVE_ONNX ON)
    set(INCL_REL include/onnxruntime/core/session)
    set(ONNX_INCLUDE_DIR ${ONNXRT_SOURCE_DIR}/${INCL_REL})
    set(ONNX_LIB_DIR ${ONNXRT_BUILD_DIR}/Linux/Release)

    list(APPEND INCLUDE_DIRS_ALL ${ONNX_INCLUDE_DIR})

    list(APPEND LINK_DIRS_ALL ${ONNX_LIB_DIR})

    add_definitions(-DHAVE_ONNX)

endif()


# TFLITE
if(${WITH_TFLITE_RUNTIME})

    set(HAVE_TFLITE ON)
    set(TFLITE_SCHEMA_DIR "${TFLITE_SOURCE_DIR}/schema")
    set(TFLITE_FLATBUFFERS_DIR "${TFLITE_BUILD_DIR}/flatbuffers/include")

    list(APPEND INCLUDE_DIRS_ALL ${TFLITE_SOURCE_DIR})
    list(APPEND INCLUDE_DIRS_ALL ${TFLITE_SCHEMA_DIR})
    list(APPEND INCLUDE_DIRS_ALL ${TFLITE_FLATBUFFERS_DIR})

    list(APPEND LINK_DIRS_ALL ${TFLITE_BUILD_DIR})

    add_definitions(-DHAVE_TFLITE)

endif()


# TRT
if(${WITH_TRT})

    set(HAVE_TRT ON)
    set(TRT_TRG "x86_64-linux-gnu")
    set(TRT_COM ${TRT_SOURCE_DIR}/targets/${TRT_TRG}/samples/common)
    set(CUDA_INCLUDE_DIR /usr/local/cuda/include)
    set(CUDA_LIB_DIR /usr/local/cuda-10.2/targets/x86_64-linux/lib)

    list(APPEND INCLUDE_DIRS_ALL ${TRT_SOURCE_DIR}/include)
    list(APPEND INCLUDE_DIRS_ALL ${TRT_COM})
    list(APPEND INCLUDE_DIRS_ALL ${CUDA_INCLUDE_DIR})

    list(APPEND LINK_DIRS_ALL ${TRT_BUILD_DIR}/lib)
    list(APPEND LINK_DIRS_ALL ${CUDA_LIB_DIR})

    add_definitions(-DHAVE_TRT)

endif()


message("WITH_ONNX_RUNTIME ${WITH_ONNX_RUNTIME}")
message("HAVE_ONNX ${HAVE_ONNX}")

message("WITH_TFLITE_RUNTIME ${WITH_TFLITE_RUNTIME}")
message("HAVE_TFLITE ${HAVE_TFLITE}")

message("WITH_TRT ${WITH_TRT}")
message("HAVE_TRT ${HAVE_TRT}")

message("INCLUDE_DIRS_ALL ${INCLUDE_DIRS_ALL}")
message("LINK_DIRS_ALL ${LINK_DIRS_ALL}")


INCLUDE_DIRECTORIES(${INCLUDE_DIRS_ALL})
LINK_DIRECTORIES(${LINK_DIRS_ALL})


####################### build library #######################
set(SOURCE_FILES
    ${CMAKE_SOURCE_DIR}/src/input_types/input_data.h
    ${CMAKE_SOURCE_DIR}/src/input_types/input_data.cpp

    ${CMAKE_SOURCE_DIR}/src/ml_engines/engine.h
    ${CMAKE_SOURCE_DIR}/src/ml_engines/engine.cpp

    ${CMAKE_SOURCE_DIR}/src/prediction.h
    ${CMAKE_SOURCE_DIR}/src/prediction.cpp

    ${CMAKE_SOURCE_DIR}/src/clustering/clustering.h
    ${CMAKE_SOURCE_DIR}/src/clustering/clustering.cpp
    ${CMAKE_SOURCE_DIR}/src/clustering/clustering_dbscan.h
    ${CMAKE_SOURCE_DIR}/src/clustering/clustering_dbscan.cpp

    ${CMAKE_SOURCE_DIR}/third_party/DBSCAN/dbscan.h
    ${CMAKE_SOURCE_DIR}/third_party/DBSCAN/dbscan.cpp

    ${CMAKE_SOURCE_DIR}/third_party/cnpy/cnpy.h
    ${CMAKE_SOURCE_DIR}/third_party/cnpy/cnpy.cpp
    )

if(${WITH_ONNX_RUNTIME})
    list(APPEND SOURCE_FILES
        ${CMAKE_SOURCE_DIR}/src/ml_engines/engine_onnx.h
        ${CMAKE_SOURCE_DIR}/src/ml_engines/engine_onnx.cpp
        )
endif()

if(${WITH_TFLITE_RUNTIME})
    list(APPEND SOURCE_FILES
        ${CMAKE_SOURCE_DIR}/src/ml_engines/engine_tflite.h
        ${CMAKE_SOURCE_DIR}/src/ml_engines/engine_tflite.cpp
        )
endif()

if(${WITH_TRT})
    list(APPEND SOURCE_FILES
        ${CMAKE_SOURCE_DIR}/src/ml_engines/engine_trt.h
        ${CMAKE_SOURCE_DIR}/src/ml_engines/engine_trt.cpp
        )

    # TODO check this...
    list(APPEND SOURCE_FILES
        ${TRT_COM}/logger.h
        ${TRT_COM}/logger.cpp
        )
endif()

#message("SOURCE_FILES ${SOURCE_FILES}")

add_library(Infero SHARED ${SOURCE_FILES})

target_link_libraries(Infero
    eckit
    eckit_option
)

#################### build the executable ####################
add_executable(infero
    ${CMAKE_SOURCE_DIR}/src/infero.cpp
)

target_link_libraries(infero
    Infero
)

if(${WITH_ONNX_RUNTIME})
    target_link_libraries(infero
        onnxruntime
        )
endif()

if(${WITH_TFLITE_RUNTIME})
    target_link_libraries(infero
        tensorflow-lite
        )
endif()

if(${WITH_TRT})
    target_link_libraries(infero
        nvinfer
        nvonnxparser
        nvparsers
        nvinfer_plugin
        nvrtc
        cudart
        )
endif()
###############################################################

##################### ONNX2TRTconverter #######################
if(${WITH_TRT})

    add_executable(onnx2trt
        ${CMAKE_SOURCE_DIR}/src/onnx2trt.cpp
    )

    target_link_libraries(onnx2trt

        eckit
        eckit_option
        Infero

        nvinfer
        nvonnxparser
        nvparsers
        nvinfer_plugin
        nvrtc
        cudart
        )
endif()
###############################################################










