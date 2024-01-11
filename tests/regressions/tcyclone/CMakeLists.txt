# (C) Copyright 1996- ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

# ----------------- Runners ----------------
ecbuild_add_executable( TARGET runner_tcyclone
   SOURCES   runner_tcyclone.c
   CONDITION HAVE_TOOLS
   INCLUDES  ${eckit_INCLUDE_DIRS}
   LIBS      inferoapi
   NOINSTALL
)

set_target_properties( runner_tcyclone
    PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)


if(infero_HAVE_FORTRAN_APIS)
    ecbuild_add_executable( TARGET runner_tcyclone_f90
       SOURCES   runner_tcyclone.F90
       CONDITION HAVE_TOOLS
       INCLUDES  ${eckit_INCLUDE_DIRS}
       LIBS      inferof
       NOINSTALL
    )

    set_target_properties( runner_tcyclone_f90
        PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
    )
endif()

# ----------------- tests ------------------
list(APPEND tests_types_ "")
list(APPEND tests_models_ "")
list(APPEND tests_batches_ "")

if (HAVE_ONNX)
    list(APPEND tests_types_ onnx)
    list(APPEND tests_models_ cyclone_model_200x200.onnx)
    list(APPEND tests_batches_ 32)
endif()

if (HAVE_TF_C)
    list(APPEND tests_types_ tf_c)
    list(APPEND tests_models_ cyclone_model_200x200_tf)
    list(APPEND tests_batches_ 32)
endif()

if (HAVE_TF_LITE)
    list(APPEND tests_types_ tflite)
    list(APPEND tests_models_ cyclone_model_200x200.tflite)
    list(APPEND tests_batches_ 1)
endif()

if (HAVE_TENSORRT)
    list(APPEND tests_types_ tensorrt)
    list(APPEND tests_models_ cyclone_model_200x200.trt)
    list(APPEND tests_batches_ 32)
endif()

message("TCYCLONE test models: ${tests_models_}")

list( LENGTH tests_types_ _count )
math( EXPR _count "${_count}-1" )

if(NOT ${_count} EQUAL "-1")
    foreach( _i RANGE ${_count} )

        list( GET tests_types_ ${_i} type_ )
        list( GET tests_models_ ${_i} model_ )
        list( GET tests_batches_ ${_i} batch_ )

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
        if(infero_HAVE_FORTRAN_APIS)
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
        endif()

    endforeach()
endif()



