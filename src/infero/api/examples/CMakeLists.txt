# ecbuild_add_executable( TARGET runner_oro_drag
#    SOURCES   runner_oro_drag.F90
#    CONDITION HAVE_BUILD_TOOLS
#    INCLUDES  ${eckit_INCLUDE_DIRS}
#    LIBS      inferof )

# ecbuild_add_executable( TARGET runner_oro_drag_file
#    SOURCES   runner_oro_drag_file.F90
#    CONDITION HAVE_BUILD_TOOLS
#    INCLUDES  ${eckit_INCLUDE_DIRS}
#    LIBS      inferof )

ecbuild_add_executable( TARGET runner_tcyclone
   SOURCES   runner_tcyclone.F90
   CONDITION HAVE_BUILD_TOOLS
   INCLUDES  ${eckit_INCLUDE_DIRS}
   LIBS      inferof ) 

ecbuild_add_executable( TARGET runner_tcyclone_ctensor
    SOURCES   runner_tcyclone.c
    CONDITION HAVE_BUILD_TOOLS
    INCLUDES  ${eckit_INCLUDE_DIRS}
    LIBS      inferoapi )

ecbuild_add_executable( TARGET runner_mimo
   SOURCES   runner_mimo.c
   CONDITION HAVE_BUILD_TOOLS
   INCLUDES  ${eckit_INCLUDE_DIRS}
   LIBS      inferoapi )

ecbuild_add_executable( TARGET runner_mimo_f90
   SOURCES   runner_mimo.F90
   CONDITION HAVE_BUILD_TOOLS
   INCLUDES  ${eckit_INCLUDE_DIRS}
   LIBS      inferof )