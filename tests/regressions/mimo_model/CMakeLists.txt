
list(APPEND tests_types_ "")
list(APPEND tests_models_ "")


if (HAVE_ONNX)   
    list(APPEND tests_types_ onnx)
    list(APPEND tests_models_ mimo_model.onnx)
endif()


if (HAVE_TF_C)
    list(APPEND tests_types_ tf_c)
    list(APPEND tests_models_ mimo_model_tf)
endif()


if (HAVE_TF_LITE)
    list(APPEND tests_types_ tflite)
    list(APPEND tests_models_ mimo_model.tflite)
endif()

if (HAVE_TENSORRT)
    list(APPEND tests_types_ tensorrt)
    list(APPEND tests_models_ mimo_model.trt)
endif()

message("MIMO test models: ${tests_models_}")

list( LENGTH tests_types_ _count )
math( EXPR _count "${_count}-1" )

foreach( _i RANGE ${_count} )

    list( GET tests_types_ ${_i} type_ )
    list( GET tests_models_ ${_i} model_ )

    # Different models use different tensor names
    if (${type_} MATCHES "tf_c")
        set(input_name_1 "serving_default_input_1")
        set(input_name_2 "serving_default_input_2")
        set(output_name_1 "StatefulPartitionedCall")
    else()
        set(input_name_1 "input_1")
        set(input_name_2 "input_2")
        set(output_name_1 "dense_6")
    endif()

    ecbuild_configure_file(
        test_mimo_template.sh.in
        test_mimo_${type_}.sh @ONLY
    )

    ecbuild_add_test(
        TYPE     SCRIPT
        COMMAND  test_mimo_${type_}.sh
    )
endforeach()

