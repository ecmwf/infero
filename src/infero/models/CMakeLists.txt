
list(APPEND infero_srcs    
    InferenceModel.h
    InferenceModel.cc
    ${PROJECT_SOURCE_DIR}/src/infero/infero_utils.h
    ${PROJECT_SOURCE_DIR}/contrib/cnpy/cnpy.h
    ${PROJECT_SOURCE_DIR}/contrib/cnpy/cnpy.cpp
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


if(HAVE_MPI)
    add_definitions(-DHAVE_MPI)
endif()

ecbuild_add_library(

    TARGET infero

    INSTALL_HEADERS LISTED

    HEADER_DESTINATION
        ${INSTALL_INCLUDE_DIR}/infero

    # GENERATED
    #     infero_version.cc

    SOURCES
        ${infero_srcs}

    PUBLIC_INCLUDES
       $<BUILD_INTERFACE:${PROJECT_BINARY_DIR}/src>
       $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/src>
       $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/contrib>

    PRIVATE_INCLUDES
        "${eckit_INCLUDE_DIRS}"
        "${MPI_INCLUDE_DIRS}"
        "${TENSORFLOWLITE_INCLUDE_DIRS}"
        "${ONNX_INCLUDE_DIRS}"
        "${TENSORRT_INCLUDE_DIRS}"        

    PRIVATE_LIBS
        ${ZLIB_LIBRARIES}        
        eckit
        eckit_option
        eckit_mpi
        "${tensorflow_lite_libs}"
        "${onnx_libs}"
        "${tensor_rt_libs}"

)
