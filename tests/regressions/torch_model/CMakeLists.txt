# (C) Copyright 1996- ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

# ----------------- Runners ----------------
ecbuild_add_executable( TARGET runner_torch
   SOURCES   runner_torch.c
   CONDITION HAVE_TOOLS
   INCLUDES  ${eckit_INCLUDE_DIRS}
   LIBS      inferoapi
   NOINSTALL
)
set_target_properties( runner_torch
    PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)

ecbuild_add_executable( TARGET runner_torch_mimo
   SOURCES   runner_torch_mimo.c
   CONDITION HAVE_TOOLS
   INCLUDES  ${eckit_INCLUDE_DIRS}
   LIBS      inferoapi
   NOINSTALL
)
set_target_properties( runner_torch_mimo
    PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)

if(infero_HAVE_FORTRAN_APIS)
    ecbuild_add_executable( TARGET runner_torch_mimo_F90
       SOURCES   runner_torch_mimo.F90
       CONDITION HAVE_TOOLS
       INCLUDES  ${eckit_INCLUDE_DIRS}
       LIBS      inferof
       NOINSTALL
    )

    set_target_properties( runner_torch_mimo_F90
        PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
    )
endif()




# ----------------- tests ------------------
list(APPEND tests_names_ "")
list(APPEND tests_types_ "")
list(APPEND tests_models_ "")
list(APPEND tests_exe_ "")

if (HAVE_TORCH)

    # model
    list(APPEND tests_names_ single_input_single_output)
    list(APPEND tests_types_ torch)
    list(APPEND tests_models_ torch_model.pt)
    list(APPEND tests_exe_ runner_torch)
    
    # model mimo
    list(APPEND tests_names_ multi_input_multi_output)
    list(APPEND tests_types_ torch)
    list(APPEND tests_models_ torch_mimo.pt)
    list(APPEND tests_exe_ runner_torch_mimo)


    # model mimo
    list(APPEND tests_names_ multi_input_multi_output_fortran)
    list(APPEND tests_types_ torch)
    list(APPEND tests_models_ torch_mimo.pt)
    list(APPEND tests_exe_ runner_torch_mimo_F90)

endif()

message("TORCH test models: ${tests_models_} ")

list( LENGTH tests_types_ _count )
math( EXPR _count "${_count}-1" )

if(NOT ${_count} EQUAL "-1")
    foreach( _i RANGE ${_count} )

        list( GET tests_names_ ${_i} name_ )
        list( GET tests_types_ ${_i} type_ )
        list( GET tests_models_ ${_i} model_ )
        list( GET tests_exe_ ${_i} exe_ )

        ecbuild_configure_file(
            test_torch_template.sh.in
            test_torch_${name_}.sh @ONLY
        )

        ecbuild_add_test(
            TYPE     SCRIPT
            COMMAND  test_torch_${name_}.sh
        )

    endforeach()
endif()








