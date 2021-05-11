list(APPEND infero_srcs
    input_types/input_data.h
    input_types/input_data.cpp
    ml_engines/engine.h
    ml_engines/engine.cpp
    prediction.h
    prediction.cpp
    # clustering/clustering.h
    # clustering/clustering.cpp
    # clustering/clustering_dbscan.h
    # clustering/clustering_dbscan.cpp
    # contrib/DBSCAN/dbscan.h
    # contrib/DBSCAN/dbscan.cpp
    ../../contrib/cnpy/cnpy.h
    ../../contrib/cnpy/cnpy.cpp
)

if(HAVE_ONNX)
    list(APPEND infero_srcs
        ml_engines/engine_onnx.h
        ml_engines/engine_onnx.cpp
        )
    list(APPEND onnx_libs  onnxruntime )
endif()

if(HAVE_TF_LITE)
    list(APPEND infero_srcs
        ml_engines/engine_tflite.h
        ml_engines/engine_tflite.cpp
    )
    list(APPEND tensorflow_lite_libs ${TENSORFLOWLITE_LIBRARIES})
endif()

if(HAVE_TENSOR_RT)
    list(APPEND infero_srcs
        ml_engines/engine_trt.h
        ml_engines/engine_trt.cpp
    )

    list(APPEND tensor_rt_libs 
        nvinfer
        nvonnxparser
        nvparsers
        nvinfer_plugin
        nvrtc
        cudart
    )

    # # TODO check this...
    # list(APPEND SOURCE_FILES
    #     ${TRT_COM}/logger.h
    #     ${TRT_COM}/logger.cpp
    #     )
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

    PRIVATE_LIBS
        eckit
        eckit_option
        "${tensorflow_lite_libs}"
        "${tensor_rt_libs}"
        "${onnx_libs}"
)
