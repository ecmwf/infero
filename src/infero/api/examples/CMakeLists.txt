# (C) Copyright 1996- ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

ecbuild_add_executable( TARGET infero_runner_tcyclone_ctensor
    SOURCES   runner_tcyclone.c
    CONDITION HAVE_TOOLS
    INCLUDES  ${eckit_INCLUDE_DIRS}
    LIBS      inferoapi
    NOINSTALL
)

set_target_properties( infero_runner_tcyclone_ctensor
    PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)

ecbuild_add_executable( TARGET infero_runner_mimo
   SOURCES   runner_mimo.c
   CONDITION HAVE_TOOLS
   INCLUDES  ${eckit_INCLUDE_DIRS}
   LIBS      inferoapi
   NOINSTALL
)
set_target_properties( infero_runner_mimo
    PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)


if(HAVE_FORTRAN)

    ecbuild_add_executable( TARGET infero_runner_oro_drag_file
       SOURCES   runner_oro_drag_file.F90
       CONDITION HAVE_TOOLS
       INCLUDES  ${eckit_INCLUDE_DIRS}
       LIBS      inferof
       NOINSTALL
    )
    set_target_properties( infero_runner_oro_drag_file
        PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
    )

    ecbuild_add_executable( TARGET infero_runner_oro_drag
       SOURCES   runner_oro_drag.F90
       CONDITION HAVE_TOOLS
       INCLUDES  ${eckit_INCLUDE_DIRS}
       LIBS      inferof
       NOINSTALL
    )
    set_target_properties( infero_runner_oro_drag
        PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
    )

    ecbuild_add_executable( TARGET infero_runner_tcyclone
       SOURCES   runner_tcyclone.F90
       CONDITION HAVE_TOOLS
       INCLUDES  ${eckit_INCLUDE_DIRS}
       LIBS      inferof
       NOINSTALL
    )
    set_target_properties( infero_runner_tcyclone
        PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
    )

    ecbuild_add_executable( TARGET infero_runner_mimo_f90
       SOURCES   runner_mimo.F90
       CONDITION HAVE_TOOLS
       INCLUDES  ${eckit_INCLUDE_DIRS}
       LIBS      inferof
       NOINSTALL
    )
    set_target_properties( infero_runner_mimo_f90
        PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
    )

endif()
