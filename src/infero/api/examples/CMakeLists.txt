#ecbuild_add_executable( TARGET runner
#    SOURCES   runner.c
#    CONDITION HAVE_BUILD_TOOLS
#    INCLUDES  ${eckit_INCLUDE_DIRS}
#    LIBS      inferoapi )

ecbuild_add_executable( TARGET runner_oro_drag
    SOURCES   runner_oro_drag.F90
    CONDITION HAVE_BUILD_TOOLS
    INCLUDES  ${eckit_INCLUDE_DIRS}
    LIBS      inferof )

ecbuild_add_executable( TARGET runner_tcyclone
    SOURCES   runner_tcyclone.F90
    CONDITION HAVE_BUILD_TOOLS
    INCLUDES  ${eckit_INCLUDE_DIRS}
    LIBS      inferof )

