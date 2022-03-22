list(APPEND tests_types_ "")
list(APPEND tests_models_ "")

if (HAVE_ONNX)
    list(APPEND tests_types_ onnx)
    list(APPEND tests_models_ cyclone_model_200x200.onnx)
endif()

if (HAVE_TF_C)
    list(APPEND tests_types_ tf_c)
    list(APPEND tests_models_ cyclone_model_200x200_tf)
endif()

if (HAVE_TF_LITE)
    list(APPEND tests_types_ tflite)
    list(APPEND tests_models_ cyclone_model_200x200.tflite)
endif()

if (HAVE_TENSORRT)
    list(APPEND tests_types_ tensorrt)
    list(APPEND tests_models_ cyclone_model_200x200.trt)
endif()

message("MIMO test models: ${tests_models_}")

list( LENGTH tests_types_ _count )
math( EXPR _count "${_count}-1" )

foreach( _i RANGE ${_count} )

    list( GET tests_types_ ${_i} type_ )
    list( GET tests_models_ ${_i} model_ )

    # Different models use different tensor names
    if (${type_} MATCHES "tf_c")
        set(tensor_names_ "--input_layer=serving_default_input_1 --output_layer=StatefulPartitionedCall")
    else()
        set(tensor_names_ "")
    endif()

    ecbuild_configure_file(
        test_tcyclone_template.sh.in
        test_tcyclone_${type_}.sh @ONLY
    )

    ecbuild_add_test(
        TYPE     SCRIPT
        COMMAND  test_tcyclone_${type_}.sh
    )

endforeach()




