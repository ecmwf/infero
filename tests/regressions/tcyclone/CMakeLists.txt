# ----------------- Runners ----------------
ecbuild_add_executable( TARGET runner_tcyclone
   SOURCES   runner_tcyclone.c
   CONDITION HAVE_BUILD_TOOLS
   INCLUDES  ${eckit_INCLUDE_DIRS}
   LIBS      inferoapi
   NOINSTALL
)

set_target_properties( runner_tcyclone
    PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)

ecbuild_add_executable( TARGET runner_tcyclone_f90
   SOURCES   runner_tcyclone.F90
   CONDITION HAVE_BUILD_TOOLS
   INCLUDES  ${eckit_INCLUDE_DIRS}
   LIBS      inferof
   NOINSTALL
)

set_target_properties( runner_tcyclone_f90
    PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)

# ----------------- tests ------------------
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

message("TCYCLONE test models: ${tests_models_}")

list( LENGTH tests_types_ _count )
math( EXPR _count "${_count}-1" )

if(NOT ${_count} EQUAL "-1")
    foreach( _i RANGE ${_count} )

        list( GET tests_types_ ${_i} type_ )
        list( GET tests_models_ ${_i} model_ )

        # Different models use different tensor names
        if (${type_} MATCHES "tf_c")
            set(tensor_names_ "--input_layer=serving_default_input_1 --output_layer=StatefulPartitionedCall")
        else()
            set(tensor_names_ "")
        endif()

        # C-runner
        SET(runner_exe "runner_tcyclone")
        SET(input_data_filename "cyclone_input_200x200_ctensor.csv")
        SET(ref_data_filename "cyclone_output_200x200_ctensor.csv")
        ecbuild_configure_file(
            test_tcyclone_template.sh.in
            test_tcyclone_${type_}_c.sh @ONLY
        )

        ecbuild_add_test(
            TYPE     SCRIPT
            COMMAND  test_tcyclone_${type_}_c.sh
        )

        # F90-runner
        SET(runner_exe "runner_tcyclone_f90")
        SET(input_data_filename "cyclone_input_200x200_ftensor.csv")
        SET(ref_data_filename "cyclone_output_200x200_ftensor.csv")
        ecbuild_configure_file(
            test_tcyclone_template.sh.in
            test_tcyclone_${type_}_f90.sh @ONLY
        )

        ecbuild_add_test(
            TYPE     SCRIPT
            COMMAND  test_tcyclone_${type_}_f90.sh
        )

    endforeach()
endif()



