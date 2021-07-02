add_subdirectory(api)
add_subdirectory(clustering)
add_subdirectory(models)


# infero.x executable
ecbuild_add_executable( TARGET infero.x
    SOURCES   infero.cc
    CONDITION HAVE_BUILD_TOOLS
    INCLUDES  ${eckit_INCLUDE_DIRS}
    LIBS      infero eckit eckit_option )

# infero.x executable
ecbuild_add_executable( TARGET infero_par.x
    SOURCES   infero_par.cc
    CONDITION HAVE_BUILD_TOOLS
    INCLUDES  ${eckit_INCLUDE_DIRS} ${MPI_INCLUDE_DIRS}
    LIBS      infero eckit eckit_option eckit_mpi)
