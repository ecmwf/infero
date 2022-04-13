# ----------------- Runners ----------------
ecbuild_add_executable( TARGET runner_mimo
   SOURCES   runner_mimo.c
   CONDITION HAVE_BUILD_TOOLS
   INCLUDES  ${eckit_INCLUDE_DIRS}
   LIBS      inferoapi
   NOINSTALL
)

set_target_properties( runner_mimo
    PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)

ecbuild_add_executable( TARGET runner_mimo_f90
   SOURCES   runner_mimo.F90
   CONDITION HAVE_BUILD_TOOLS
   INCLUDES  ${eckit_INCLUDE_DIRS}
   LIBS      inferof
   NOINSTALL
)

set_target_properties( runner_mimo_f90
    PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)

# ----------------- tests ------------------
list(APPEND tests_types_ "")
list(APPEND tests_models_ "")
list(APPEND tests_layers_ "")

if (HAVE_ONNX)   
    list(APPEND tests_types_ onnx)
    list(APPEND tests_models_ mimo_model.onnx)
    list(APPEND tests_layers_ "input_1 input_2 dense_6")
endif()

if (HAVE_TF_C)
    list(APPEND tests_types_ tf_c)
    list(APPEND tests_models_ mimo_model_tf)
    list(APPEND tests_layers_ "serving_default_input_1 serving_default_input_2 StatefulPartitionedCall")
endif()

if (HAVE_TF_LITE)
    list(APPEND tests_types_ tflite)
    list(APPEND tests_models_ mimo_model.tflite)
    list(APPEND tests_layers_ "input_1 input_2 dense_6")
endif()

if (HAVE_TENSORRT)
    list(APPEND tests_types_ tensorrt)
    list(APPEND tests_models_ mimo_model.trt)
    list(APPEND tests_layers_ "input_1 input_2 dense_6")
endif()

message("MIMO test models: ${tests_models_}")

list( LENGTH tests_types_ _count )
math( EXPR _count "${_count}-1" )

if(NOT ${_count} EQUAL "-1")
    foreach( _i RANGE ${_count} )

        list( GET tests_types_ ${_i} type_ )
        list( GET tests_models_ ${_i} model_ )
        list( GET tests_layers_ ${_i} layers_ )

        # C-runner
        SET(runner_exe "runner_mimo")
        ecbuild_configure_file(
            test_mimo_template.sh.in
            test_mimo_${type_}_c.sh @ONLY
        )

        ecbuild_add_test(
            TYPE     SCRIPT
            COMMAND  test_mimo_${type_}_c.sh
        )

        # F90-runner
        SET(runner_exe "runner_mimo_f90")
        ecbuild_configure_file(
            test_mimo_template.sh.in
            test_mimo_${type_}_f90.sh @ONLY
        )

        ecbuild_add_test(
            TYPE     SCRIPT
            COMMAND  test_mimo_${type_}_f90.sh
        )

    endforeach()
endif()

