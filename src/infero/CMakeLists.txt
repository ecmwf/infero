add_subdirectory(api)
add_subdirectory(clustering)

list(APPEND infero_srcs
    MLTensor.h
    MLTensor.cc
    ml_engines/MLEngine.h
    ml_engines/MLEngine.cc
    ${PROJECT_SOURCE_DIR}/contrib/cnpy/cnpy.h
    ${PROJECT_SOURCE_DIR}/contrib/cnpy/cnpy.cpp
)

### support for ONNX
if(HAVE_ONNX)
    list(APPEND infero_srcs
        ml_engines/MLEngineONNX.h
        ml_engines/MLEngineONNX.cc
        )
    list(APPEND onnx_libs  ${ONNX_LIBRARIES} )
    add_definitions(-DHAVE_ONNX)
endif()


### support for TensorflowLite
if(HAVE_TF_LITE)
    list(APPEND infero_srcs
        ml_engines/MLEngineTFlite.h
        ml_engines/MLEngineTFlite.cc
    )
    list(APPEND tensorflow_lite_libs ${TENSORFLOWLITE_LIBRARIES})
    add_definitions(-DHAVE_TFLITE)
endif()


### support for TensorRT
if(HAVE_TENSORRT)
    list(APPEND infero_srcs
        ml_engines/MLEngineTRT.h
        ml_engines/MLEngineTRT.cc
    )

    set(tensor_rt_libs ${TENSORRT_LIBRARIES})

    # TODO check this...
    list(APPEND infero_srcs
        ${TENSORRT_INCLUDE_COMMON_DIR}/logger.h
        ${TENSORRT_INCLUDE_COMMON_DIR}/logger.cpp
        )
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
        "${TENSORFLOWLITE_INCLUDE_DIRS}"
        "${ONNX_INCLUDE_DIRS}"
        "${CUDA_INCLUDE_DIRS}"
        "${TENSORRT_INCLUDE_DIRS}"

    PRIVATE_LIBS
        ${ZLIB_LIBRARIES}
        eckit
        eckit_option
        "${tensorflow_lite_libs}"        
        "${onnx_libs}"
        ${CUDA_LIBRARIES}
        "${tensor_rt_libs}"
)

ecbuild_add_executable( TARGET infero.x
    SOURCES   infero.cc
    CONDITION HAVE_BUILD_TOOLS
    INCLUDES  ${eckit_INCLUDE_DIRS}
    LIBS      infero eckit eckit_option )
