list(APPEND tests_types_ "")
list(APPEND tests_models_ "")
list(APPEND tests_layers_ "init_")

if (HAVE_ONNX)
    list(APPEND tests_types_ onnx)
    list(APPEND tests_models_ model.onnx)
    list(APPEND tests_layers_ "")
endif()

if (HAVE_TF_C)
    list(APPEND tests_types_ tf_c)
    list(APPEND tests_models_ model_tf)
    list(APPEND tests_layers_ "--input_layer=serving_default_dense_21_input --output_layer=StatefulPartitionedCall")
endif()

if (HAVE_TF_LITE)
    list(APPEND tests_types_ tflite)
    list(APPEND tests_models_ model.tflite)
    list(APPEND tests_layers_ "")
endif()

if (HAVE_TENSORRT)
    list(APPEND tests_types_ tensorrt)
    list(APPEND tests_models_ model.trt)
    list(APPEND tests_layers_ "")
endif()

list(POP_FRONT tests_layers_)

message("MIMO test models: ${tests_models_}")

list( LENGTH tests_types_ _count )

math( EXPR _count "${_count}-1" )

foreach( _i RANGE ${_count} )

    list( GET tests_types_ ${_i} type_ )
    list( GET tests_models_ ${_i} model_ )
    list( GET tests_layers_ ${_i} layers_ )

    ecbuild_configure_file(
        test_oro_drag_template.sh.in
        test_oro_drag_${type_}.sh @ONLY
    )

    ecbuild_add_test(
        TYPE     SCRIPT
        COMMAND  test_oro_drag_${type_}.sh
    )

endforeach()








